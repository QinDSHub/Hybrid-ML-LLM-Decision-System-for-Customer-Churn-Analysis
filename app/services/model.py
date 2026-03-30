from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from app.services.embeddings import BaseEmbedder, build_embedder
from app.services.feature_engineering import NUMERIC_COLUMNS, TEXT_COLUMNS, normalize_car_level_label, normalize_repair_type_string


@dataclass
class ModelConfig:
    numeric_columns: list[str] = field(default_factory=lambda: NUMERIC_COLUMNS.copy())
    text_columns: list[str] = field(default_factory=lambda: TEXT_COLUMNS.copy())
    text_weight: float = 0.3
    numeric_weight: float = 0.7
    knn_k: int = 10
    positive_threshold: float = 0.4
    embedding_provider: str = 'openai'
    openai_model: str = 'text-embedding-3-small'
    hash_dimension: int = 256
    embedding_dimension: int | None = None
    feature_version: str = 'v2-refactor'


class HybridKNNChurnModel:
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self._embedder = embedder

        self.scalers: dict[str, Any] = {}
        self.train_embeddings: np.ndarray | None = None
        self.train_labels: np.ndarray | None = None
        self.train_vins: list[str] = []
        self.training_texts: list[str] = []
        self.created_at_utc: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.train_embeddings is not None and self.train_labels is not None and bool(self.scalers)

    def _get_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = build_embedder(
                self.config.embedding_provider,
                openai_model=self.config.openai_model,
                openai_api_key=self.openai_api_key,
                openai_base_url=self.openai_base_url,
                hash_dimension=self.config.hash_dimension,
            )
        return self._embedder

    def _choose_scaler(self, series: pd.Series) -> Any:
        values = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(float).to_numpy()
        values_2d = values.reshape(-1, 1)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        skewness = pd.Series(values).skew()

        if iqr > 0 and (float(values.max()) - float(values.min())) / iqr > 10:
            scaler = RobustScaler()
        elif abs(float(0.0 if pd.isna(skewness) else skewness)) > 1:
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            scaler = StandardScaler()
        scaler.fit(values_2d)
        return scaler

    def _fit_scalers(self, frame: pd.DataFrame) -> None:
        self.scalers = {}
        for column in self.config.numeric_columns:
            self.scalers[column] = self._choose_scaler(frame[column])

    def _scale_numeric_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.scalers:
            raise RuntimeError('模型尚未拟合，数值 scaler 不存在。')

        columns: list[np.ndarray] = []
        for column in self.config.numeric_columns:
            if column not in frame.columns:
                raise KeyError(f'缺少数值特征列: {column}')
            values = pd.to_numeric(frame[column], errors='coerce').fillna(0.0).astype(float).to_numpy().reshape(-1, 1)
            transformed = self.scalers[column].transform(values).ravel().astype(np.float32)
            columns.append(transformed)

        return np.vstack(columns).T.astype(np.float32)

    def _compose_text_features(self, frame: pd.DataFrame) -> list[str]:
        missing = [column for column in self.config.text_columns if column not in frame.columns]
        if missing:
            raise KeyError(f'缺少文本特征列: {missing}')

        text_frame = frame.copy()
        text_frame['member_level'] = text_frame['member_level'].fillna('无').astype(str).map(lambda x: f'会员卡：{x}')
        text_frame['owner_type'] = text_frame['owner_type'].fillna('个人').astype(str).map(lambda x: f'用户性质：{x}')
        text_frame['car_mode'] = text_frame['car_mode'].fillna('未知车型').astype(str).map(lambda x: f'汽车型号：{x}')
        text_frame['car_level'] = text_frame['car_level'].map(normalize_car_level_label)
        text_frame['last_repair_type'] = text_frame['last_repair_type'].map(normalize_repair_type_string).map(
            lambda x: f'上次进店类型：{x}'
        )
        text_frame['all_repair_types'] = text_frame['all_repair_types'].map(normalize_repair_type_string).map(
            lambda x: f'历史进店类型：{x}'
        )

        return (
            text_frame[
                ['last_repair_type', 'all_repair_types', 'owner_type', 'car_mode', 'car_level', 'member_level']
            ]
            .astype(str)
            .apply(lambda row: '，'.join(row.tolist()), axis=1)
            .tolist()
        )

    def _normalize_vectors(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return (matrix / norms).astype(np.float32)

    def _build_feature_embeddings(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        numeric_embeddings = self._scale_numeric_frame(frame)
        texts = self._compose_text_features(frame)
        text_embeddings = self._get_embedder().embed(texts).astype(np.float32)

        if self.config.embedding_dimension is None and text_embeddings.size:
            self.config.embedding_dimension = int(text_embeddings.shape[1])

        weighted = np.hstack(
            [
                text_embeddings * float(self.config.text_weight),
                numeric_embeddings * float(self.config.numeric_weight),
            ]
        )
        final_embedding = self._normalize_vectors(weighted)
        return final_embedding, texts

    def fit(self, frame: pd.DataFrame, label_col: str = 'churn_label') -> 'HybridKNNChurnModel':
        if label_col not in frame.columns:
            raise KeyError(f'训练数据中缺少标签列: {label_col}')

        self._fit_scalers(frame)
        embeddings, texts = self._build_feature_embeddings(frame)
        self.train_embeddings = embeddings
        self.train_labels = pd.to_numeric(frame[label_col], errors='coerce').fillna(0).astype(int).to_numpy()
        self.train_vins = frame['VIN'].astype(str).tolist() if 'VIN' in frame.columns else [f'train_{i}' for i in range(len(frame))]
        self.training_texts = texts
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        return self

    def _predict_single(self, vector: np.ndarray, *, vin: str, return_neighbors: int = 5) -> dict[str, Any]:
        if not self.is_fitted or self.train_embeddings is None or self.train_labels is None:
            raise RuntimeError('模型尚未训练或加载。')

        effective_k = min(int(self.config.knn_k), len(self.train_labels))
        similarities = self.train_embeddings @ vector
        top_indices = np.argsort(similarities)[-effective_k:][::-1]
        top_labels = self.train_labels[top_indices]
        top_scores = similarities[top_indices]

        positive_neighbors = int(top_labels.sum())
        churn_score = positive_neighbors / float(effective_k)
        pred_label = int(churn_score >= float(self.config.positive_threshold))

        neighbors = []
        for idx, similarity, label in zip(top_indices[:return_neighbors], top_scores[:return_neighbors], top_labels[:return_neighbors]):
            neighbors.append(
                {
                    'vin': self.train_vins[int(idx)],
                    'label': int(label),
                    'similarity': float(similarity),
                }
            )

        return {
            'vin': vin,
            'pred_label': pred_label,
            'churn_score': float(churn_score),
            'positive_neighbors': positive_neighbors,
            'neighbor_count': int(effective_k),
            'top_neighbors': neighbors,
        }

    def predict_records(self, frame: pd.DataFrame, *, return_neighbors: int = 5) -> list[dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError('模型尚未训练或加载。')

        embeddings, _ = self._build_feature_embeddings(frame)
        vins = frame['VIN'].astype(str).tolist() if 'VIN' in frame.columns else [f'query_{i}' for i in range(len(frame))]
        return [self._predict_single(vector, vin=vin, return_neighbors=return_neighbors) for vin, vector in zip(vins, embeddings)]

    def predict_dataframe(self, frame: pd.DataFrame, *, return_neighbors: int = 5) -> pd.DataFrame:
        records = self.predict_records(frame, return_neighbors=return_neighbors)
        return pd.DataFrame(records)

    def save(self, output_path: str | Path) -> Path:
        if not self.is_fitted:
            raise RuntimeError('模型尚未训练，无法保存。')
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'config': asdict(self.config),
            'scalers': self.scalers,
            'train_embeddings': self.train_embeddings,
            'train_labels': self.train_labels,
            'train_vins': self.train_vins,
            'training_texts': self.training_texts,
            'created_at_utc': self.created_at_utc,
        }
        joblib.dump(payload, output)
        return output

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
    ) -> 'HybridKNNChurnModel':
        payload = joblib.load(model_path)
        model = cls(
            ModelConfig(**payload['config']),
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
        model.scalers = payload['scalers']
        model.train_embeddings = payload['train_embeddings']
        model.train_labels = payload['train_labels']
        model.train_vins = payload['train_vins']
        model.training_texts = payload.get('training_texts', [])
        model.created_at_utc = payload.get('created_at_utc')
        return model


def compute_validation_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_score))
    else:
        metrics['roc_auc'] = None
    return metrics


def train_and_evaluate(
    cleaned_df: pd.DataFrame,
    artifact_dir: str | Path,
    *,
    config: ModelConfig | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> tuple[HybridKNNChurnModel, dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    if 'dataset' in cleaned_df.columns and cleaned_df['dataset'].eq('valid').any():
        train_df = cleaned_df[cleaned_df['dataset'] == 'train'].reset_index(drop=True)
        valid_df = cleaned_df[cleaned_df['dataset'] == 'valid'].reset_index(drop=True)
    else:
        train_df = cleaned_df.reset_index(drop=True)
        valid_df = pd.DataFrame(columns=cleaned_df.columns)

    if train_df.empty:
        raise ValueError('训练集为空，无法训练模型。')

    model = HybridKNNChurnModel(
        config=config,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    model.fit(train_df)
    bundle_path = model.save(artifact_path / 'model_bundle.joblib')

    metrics_summary: dict[str, Any] = {
        'train_rows': int(len(train_df)),
        'valid_rows': int(len(valid_df)),
        'model_bundle_path': str(bundle_path),
        'created_at_utc': model.created_at_utc,
        'config': asdict(model.config),
    }

    if not valid_df.empty and 'churn_label' in valid_df.columns:
        predictions_df = model.predict_dataframe(valid_df)
        predictions_df['true_label'] = valid_df['churn_label'].astype(int).tolist()
        predictions_df.to_csv(artifact_path / 'validation_predictions.csv', index=False, encoding='utf-8-sig')

        y_true = predictions_df['true_label'].astype(int).to_numpy()
        y_pred = predictions_df['pred_label'].astype(int).to_numpy()
        y_score = predictions_df['churn_score'].astype(float).to_numpy()
        metrics_summary['validation'] = compute_validation_metrics(y_true, y_score, y_pred)
    else:
        metrics_summary['validation'] = None

    with (artifact_path / 'metrics.json').open('w', encoding='utf-8') as fp:
        json.dump(metrics_summary, fp, ensure_ascii=False, indent=2)

    cleaned_df.to_csv(artifact_path / 'cleaned_data_snapshot.csv', index=False, encoding='utf-8-sig')
    return model, metrics_summary
