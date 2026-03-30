from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import numpy as np
import pandas as pd

from app.services.model import ModelConfig, train_and_evaluate


def make_demo_cleaned_data(rows: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for idx in range(rows):
        label = int(idx % 2 == 0)
        is_train = idx < int(rows * 0.85)

        if label == 1:
            last_till_now_days = rng.integers(170, 360)
            day_diff_median = rng.uniform(120, 240)
            all_times = rng.integers(2, 5)
            member_level = '无'
            repair_type = '普通维修'
            all_repair_types = '普通维修;首次保养'
            owner_type = '个人'
        else:
            last_till_now_days = rng.integers(10, 80)
            day_diff_median = rng.uniform(25, 90)
            all_times = rng.integers(4, 10)
            member_level = '金卡'
            repair_type = '首次保养'
            all_repair_types = '首次保养;机油保养'
            owner_type = '企业' if idx % 3 == 0 else '个人'

        records.append(
            {
                'VIN': f'DEMO{idx:05d}',
                'last_mile': float(rng.uniform(8_000, 120_000)),
                'last_till_now_days': int(last_till_now_days),
                'first_to_purchase_day_diff': float(rng.uniform(20, 200)),
                'first_to_purchase_mile_diff': float(rng.uniform(1_000, 15_000)),
                'second_to_first_day_diff': float(rng.uniform(20, 220)),
                'second_to_first_mile_diff': float(rng.uniform(1_000, 18_000)),
                'day_diff_median': float(day_diff_median),
                'mile_diff_median': float(rng.uniform(2_000, 20_000)),
                'day_speed_median': float(rng.uniform(15, 120)),
                'day_cv': float(rng.uniform(0.05, 0.9)),
                'mile_cv': float(rng.uniform(0.05, 1.2)),
                'day_speed_cv': float(rng.uniform(0.05, 0.8)),
                'all_times': int(all_times),
                'car_age': int(rng.integers(1, 8)),
                'last_repair_type': repair_type,
                'all_repair_types': all_repair_types,
                'owner_type': owner_type,
                'car_mode': 'Model-X' if label == 1 else 'Model-Y',
                'car_level': 'family_2' if idx % 4 else 'family_1',
                'member_level': member_level,
                'churn_label': label,
                'dataset': 'train' if is_train else 'valid',
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / 'artifacts' / 'model'
    processed_dir = repo_root / 'data' / 'processed'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df = make_demo_cleaned_data()
    cleaned_df.to_csv(processed_dir / 'demo_cleaned_data.csv', index=False, encoding='utf-8-sig')

    config = ModelConfig(embedding_provider='hash', knn_k=10, positive_threshold=0.4)
    _, metrics = train_and_evaluate(cleaned_df, artifact_dir, config=config)

    print('Demo model created successfully.')
    print(metrics)


if __name__ == '__main__':
    main()
