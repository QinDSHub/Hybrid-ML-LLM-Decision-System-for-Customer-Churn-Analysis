from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services.feature_engineering import build_single_customer_feature_frame, required_feature_columns
from app.services.model import ModelConfig, train_and_evaluate
from scripts.bootstrap_demo_model import make_demo_cleaned_data


def test_build_single_customer_feature_frame() -> None:
    payload = {
        'vin': 'VIN0001',
        'owner_type': '个人',
        'car_mode': 'Model-Z',
        'car_level': 'family_2',
        'member_level': '银卡',
        'purchase_date': '2022-01-01',
        'visits': [
            {'date': '2022-03-01', 'mile': 3000, 'repair_type': '首保'},
            {'date': '2022-09-01', 'mile': 12000, 'repair_type': '普通保养'},
            {'date': '2023-04-01', 'mile': 23000, 'repair_type': '普修'},
        ],
    }
    df = build_single_customer_feature_frame(payload, reference_date='2023-10-01')
    assert not df.empty
    assert df.iloc[0]['VIN'] == 'VIN0001'
    assert df.iloc[0]['all_times'] >= 2


def test_train_and_predict_with_hash_embedding(tmp_path: Path) -> None:
    cleaned_df = make_demo_cleaned_data(rows=40)
    model, metrics = train_and_evaluate(cleaned_df, tmp_path, config=ModelConfig(embedding_provider='hash'))
    assert model.is_fitted
    assert (tmp_path / 'model_bundle.joblib').exists()
    assert 'validation' in metrics

    sample = cleaned_df.iloc[[0]][required_feature_columns()]
    predictions = model.predict_records(sample)
    assert len(predictions) == 1
    assert predictions[0]['pred_label'] in (0, 1)


def test_fastapi_predict_features_endpoint(tmp_path: Path, monkeypatch) -> None:
    cleaned_df = make_demo_cleaned_data(rows=40)
    train_and_evaluate(cleaned_df, tmp_path, config=ModelConfig(embedding_provider='hash'))

    monkeypatch.setenv('MODEL_BUNDLE_PATH', str(tmp_path / 'model_bundle.joblib'))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    get_settings.cache_clear()

    from app.api.main import create_app

    app = create_app()
    sample = cleaned_df.iloc[0][required_feature_columns()].to_dict()

    with TestClient(app) as client:
        response = client.post('/v1/predict/features', json=sample)

    assert response.status_code == 200
    body = response.json()
    assert body['vin'] == sample['VIN']
    assert body['pred_label'] in (0, 1)
