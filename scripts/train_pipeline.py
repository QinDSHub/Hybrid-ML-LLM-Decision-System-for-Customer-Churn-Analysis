from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import json

from dotenv import load_dotenv

from app.core.config import get_settings
from app.services.feature_engineering import build_customer_master, build_service_history, build_training_dataset
from app.services.model import ModelConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train churn prediction model from raw CSV files.')
    parser.add_argument('--raw-dir', type=str, required=True, help='目录内需要包含 vehicle3.csv / member_info.csv / repare_maintain_info1.csv')
    parser.add_argument('--artifact-dir', type=str, default='artifacts/model', help='模型与指标输出目录')
    parser.add_argument('--processed-path', type=str, default='data/processed/cleaned_data.csv', help='清洗后数据导出路径')
    parser.add_argument('--split-date', type=str, default=None, help='可选，指定观测截断日期，例如 2025-12-31')
    parser.add_argument('--embedding-provider', type=str, choices=['openai', 'hash'], default='openai')
    parser.add_argument('--openai-model', type=str, default='text-embedding-3-small')
    parser.add_argument('--knn-k', type=int, default=10)
    parser.add_argument('--positive-threshold', type=float, default=0.4)
    parser.add_argument('--text-weight', type=float, default=0.3)
    parser.add_argument('--numeric-weight', type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = get_settings()

    raw_dir = Path(args.raw_dir)
    artifact_dir = Path(args.artifact_dir)
    processed_path = Path(args.processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    customer_df = build_customer_master(raw_dir)
    history_df = build_service_history(raw_dir, customer_df)
    cleaned_df = build_training_dataset(customer_df, history_df, split_date=args.split_date)
    cleaned_df.to_csv(processed_path, index=False, encoding='utf-8-sig')

    config = ModelConfig(
        knn_k=args.knn_k,
        positive_threshold=args.positive_threshold,
        text_weight=args.text_weight,
        numeric_weight=args.numeric_weight,
        embedding_provider=args.embedding_provider,
        openai_model=args.openai_model,
    )

    _, metrics = train_and_evaluate(
        cleaned_df,
        artifact_dir,
        config=config,
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
    )

    summary_path = artifact_dir / 'training_summary.json'
    with summary_path.open('w', encoding='utf-8') as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    print('Training completed.')
    print(f'Cleaned data saved to: {processed_path}')
    print(f'Artifacts saved to: {artifact_dir}')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
