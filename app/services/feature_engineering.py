from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

NUMERIC_COLUMNS = [
    'last_mile',
    'last_till_now_days',
    'first_to_purchase_day_diff',
    'first_to_purchase_mile_diff',
    'second_to_first_day_diff',
    'second_to_first_mile_diff',
    'day_diff_median',
    'mile_diff_median',
    'day_speed_median',
    'day_cv',
    'mile_cv',
    'day_speed_cv',
    'all_times',
    'car_age',
]

TEXT_COLUMNS = [
    'last_repair_type',
    'all_repair_types',
    'owner_type',
    'car_mode',
    'car_level',
    'member_level',
]

INTERNAL_REPAIR_PATTERN = re.compile(r'内部|二手')
NON_ACTIVE_REPAIR_PATTERN = re.compile(r'事故|三包|质量担保|索赔|PDI|返工|免费|售前|代验车|召回|受控')


def normalize_repair_type_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '未知'

    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.split(';') if part.strip()]
    else:
        raw_parts = [str(value).strip()]

    normalized_parts: list[str] = []
    for part in raw_parts:
        if '首' in part:
            normalized_parts.append('首次保养')
        elif '普修' in part:
            normalized_parts.append('普通维修')
        else:
            normalized_parts.append(part)

    deduped = sorted(set(normalized_parts))
    return ';'.join(deduped) if deduped else '未知'


def normalize_car_level_label(value: Any) -> str:
    mapping = {
        'family_1': '高档车',
        'family_2': '中档车',
        'family_3': '低档车',
        '高档车': '高档车',
        '中档车': '中档车',
        '低档车': '低档车',
    }
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '低档车'
    return mapping.get(str(value).strip(), str(value).strip())


def required_feature_columns() -> list[str]:
    return ['VIN', *NUMERIC_COLUMNS, *TEXT_COLUMNS]


def _ensure_raw_file(path: Path, filename: str) -> Path:
    file_path = path / filename
    if not file_path.exists():
        raise FileNotFoundError(f'找不到原始文件: {file_path}')
    return file_path


def build_customer_master(raw_dir: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    vehicle_path = _ensure_raw_file(raw_path, 'vehicle3.csv')
    member_path = _ensure_raw_file(raw_path, 'member_info.csv')
    repair_path = _ensure_raw_file(raw_path, 'repare_maintain_info1.csv')

    vehicle_df = pd.read_csv(vehicle_path)
    vehicle_df = vehicle_df[['VIN', '车主性质', '车型', 'family_name']].copy()
    vehicle_df.columns = ['VIN', 'owner_type', 'car_mode', 'car_level']

    member_df = pd.read_csv(member_path)
    member_df = member_df[['VIN', '会员等级']].drop_duplicates().copy()
    member_df = member_df[~member_df['会员等级'].isna()].reset_index(drop=True)
    member_df.columns = ['VIN', 'member_level']

    repair_df = pd.read_csv(repair_path)
    purchase_df = repair_df[['VIN', 'purchase_date']].drop_duplicates().dropna().copy()
    purchase_df['purchase_date'] = pd.to_datetime(purchase_df['purchase_date'], errors='coerce')
    purchase_df = purchase_df.dropna(subset=['purchase_date'])

    customer_df = vehicle_df.merge(member_df, on='VIN', how='outer').drop_duplicates().reset_index(drop=True)
    customer_df['member_level'] = customer_df['member_level'].fillna('无')
    customer_df['owner_type'] = customer_df['owner_type'].fillna('个人')
    customer_df = customer_df.dropna(subset=['VIN', 'car_mode', 'car_level']).reset_index(drop=True)

    customer_df = customer_df.merge(purchase_df, on='VIN', how='inner')
    customer_df['purchase_date'] = pd.to_datetime(customer_df['purchase_date'], errors='coerce')
    customer_df = customer_df.dropna(subset=['purchase_date']).drop_duplicates(subset=['VIN']).reset_index(drop=True)
    return customer_df


def build_service_history(raw_dir: str | Path, customer_df: pd.DataFrame) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    repair_path = _ensure_raw_file(raw_path, 'repare_maintain_info1.csv')

    repair_df = pd.read_csv(repair_path)
    repair_df = repair_df[['VIN', '修理日期', '公里数', '修理类型']].copy()
    repair_df.columns = ['VIN', 'date', 'mile', 'repair_type']
    repair_df = repair_df[repair_df['VIN'].isin(customer_df['VIN'])].copy()
    repair_df = repair_df.dropna(subset=['VIN', 'date', 'mile', 'repair_type']).reset_index(drop=True)
    repair_df['date'] = pd.to_datetime(repair_df['date'], errors='coerce')
    repair_df['mile'] = pd.to_numeric(repair_df['mile'], errors='coerce')
    repair_df = repair_df.dropna(subset=['date', 'mile']).reset_index(drop=True)

    internal_vins = repair_df.loc[
        repair_df['repair_type'].astype(str).str.contains(INTERNAL_REPAIR_PATTERN, regex=True, na=False),
        'VIN',
    ].drop_duplicates()
    repair_df = repair_df[~repair_df['VIN'].isin(internal_vins)].copy()
    repair_df = repair_df[
        ~repair_df['repair_type'].astype(str).str.contains(NON_ACTIVE_REPAIR_PATTERN, regex=True, na=False)
    ].copy()

    grouped_mile = repair_df.groupby(['VIN', 'date'], as_index=False)['mile'].mean()
    grouped_type = (
        repair_df[['VIN', 'date', 'repair_type']]
        .drop_duplicates()
        .groupby(['VIN', 'date'])['repair_type']
        .agg(lambda values: normalize_repair_type_string(';'.join(str(v) for v in values)))
        .reset_index()
    )

    history_df = grouped_mile.merge(grouped_type, on=['VIN', 'date'], how='inner')
    history_df = history_df.sort_values(['VIN', 'date']).reset_index(drop=True)
    return history_df


def _safe_cv(std_value: float, mean_value: float) -> float:
    if mean_value in (0, 0.0) or pd.isna(mean_value):
        return 0.0
    ratio = std_value / mean_value
    if pd.isna(ratio) or np.isinf(ratio):
        return 0.0
    return float(ratio)


def _clean_single_customer_history(visits_df: pd.DataFrame) -> pd.DataFrame:
    if visits_df.empty:
        return visits_df

    visits_df = visits_df.copy()
    visits_df['date'] = pd.to_datetime(visits_df['date'], errors='coerce')
    visits_df['mile'] = pd.to_numeric(visits_df['mile'], errors='coerce')
    visits_df['repair_type'] = visits_df['repair_type'].astype(str)
    visits_df = visits_df.dropna(subset=['VIN', 'date', 'mile', 'repair_type']).reset_index(drop=True)

    if visits_df['repair_type'].str.contains(INTERNAL_REPAIR_PATTERN, regex=True, na=False).any():
        raise ValueError('该 VIN 的维修记录包含“内部/二手”类型，原始逻辑会将其排除，不建议直接预测。')

    visits_df = visits_df[
        ~visits_df['repair_type'].str.contains(NON_ACTIVE_REPAIR_PATTERN, regex=True, na=False)
    ].copy()

    if visits_df.empty:
        raise ValueError('过滤事故/索赔/召回等非主动进店记录后，没有可用于预测的维修记录。')

    grouped_mile = visits_df.groupby(['VIN', 'date'], as_index=False)['mile'].mean()
    grouped_type = (
        visits_df[['VIN', 'date', 'repair_type']]
        .drop_duplicates()
        .groupby(['VIN', 'date'])['repair_type']
        .agg(lambda values: normalize_repair_type_string(';'.join(str(v) for v in values)))
        .reset_index()
    )
    return grouped_mile.merge(grouped_type, on=['VIN', 'date'], how='inner').sort_values('date').reset_index(drop=True)


def build_feature_row(profile: Mapping[str, Any], visits_df: pd.DataFrame, reference_date: str | pd.Timestamp | None = None) -> dict[str, Any] | None:
    if visits_df.empty:
        return None

    reference_ts = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp.today().normalize()
    purchase_date = pd.Timestamp(profile['purchase_date'])

    working_df = visits_df.sort_values('date').copy().reset_index(drop=True)
    if len(working_df) < 2:
        return None

    working_df['relative_last_date'] = working_df['date'].shift(1)
    working_df['relative_last_mile'] = working_df['mile'].shift(1)
    working_df.loc[working_df.index[0], 'relative_last_date'] = purchase_date
    working_df.loc[working_df.index[0], 'relative_last_mile'] = 0.0

    working_df['day_diff'] = (working_df['date'] - pd.to_datetime(working_df['relative_last_date'])).dt.days.astype(float)
    working_df['mile_diff'] = working_df['mile'] - working_df['relative_last_mile']
    working_df['day_speed'] = np.where(
        working_df['day_diff'] > 0,
        working_df['mile_diff'] / working_df['day_diff'],
        np.nan,
    )
    working_df = working_df[working_df['day_diff'] > 0].copy().reset_index(drop=True)
    if len(working_df) < 2:
        return None

    valid_speeds = working_df.loc[working_df['mile_diff'] >= 0, 'day_speed'].replace([np.inf, -np.inf], np.nan).dropna()
    median_day_speed = float(valid_speeds.median()) if not valid_speeds.empty else 0.0

    negative_mask = working_df['mile_diff'] < 0
    working_df.loc[negative_mask, 'day_speed'] = median_day_speed
    working_df.loc[negative_mask, 'mile_diff'] = working_df.loc[negative_mask, 'day_diff'] * median_day_speed

    working_df['mile_diff'] = working_df['mile_diff'].clip(lower=0).fillna(0.0)
    working_df['day_speed'] = working_df['day_speed'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    desc_df = working_df.sort_values('date', ascending=False).reset_index(drop=True)
    latest_row = desc_df.iloc[0]
    second_row = desc_df.iloc[1]

    day_diff_mean = float(working_df['day_diff'].mean())
    day_diff_std = float(working_df['day_diff'].std()) if len(working_df) > 1 else 0.0
    mile_diff_mean = float(working_df['mile_diff'].mean())
    mile_diff_std = float(working_df['mile_diff'].std()) if len(working_df) > 1 else 0.0
    day_speed_mean = float(working_df['day_speed'].mean())
    day_speed_std = float(working_df['day_speed'].std()) if len(working_df) > 1 else 0.0

    all_repair_types = normalize_repair_type_string(';'.join(working_df['repair_type'].astype(str).tolist()))
    car_age_days = max((reference_ts - purchase_date).days, 0)

    row = {
        'VIN': str(profile['VIN']),
        'purchase_date': purchase_date.date().isoformat(),
        'last_date': pd.Timestamp(latest_row['date']).date().isoformat(),
        'last_mile': float(latest_row['mile']),
        'last_till_now_days': int(max((reference_ts - pd.Timestamp(latest_row['date'])).days, 0)),
        'last_repair_type': normalize_repair_type_string(latest_row['repair_type']),
        'first_to_purchase_day_diff': float(latest_row['day_diff']),
        'first_to_purchase_mile_diff': float(latest_row['mile_diff']),
        'second_to_first_day_diff': float(second_row['day_diff']),
        'second_to_first_mile_diff': float(second_row['mile_diff']),
        'day_diff_median': float(working_df['day_diff'].median()),
        'day_diff_std': float(0.0 if pd.isna(day_diff_std) else day_diff_std),
        'day_diff_mean': day_diff_mean,
        'mile_diff_median': float(working_df['mile_diff'].median()),
        'mile_diff_std': float(0.0 if pd.isna(mile_diff_std) else mile_diff_std),
        'mile_diff_mean': mile_diff_mean,
        'day_speed_median': float(working_df['day_speed'].median()),
        'day_speed_std': float(0.0 if pd.isna(day_speed_std) else day_speed_std),
        'day_speed_mean': day_speed_mean,
        'day_cv': _safe_cv(0.0 if pd.isna(day_diff_std) else day_diff_std, day_diff_mean),
        'mile_cv': _safe_cv(0.0 if pd.isna(mile_diff_std) else mile_diff_std, mile_diff_mean),
        'day_speed_cv': _safe_cv(0.0 if pd.isna(day_speed_std) else day_speed_std, day_speed_mean),
        'all_times': int(len(working_df)),
        'all_repair_types': all_repair_types,
        'owner_type': str(profile.get('owner_type', '个人') or '个人'),
        'car_mode': str(profile.get('car_mode', '未知车型') or '未知车型'),
        'car_level': str(profile.get('car_level', '低档车') or '低档车'),
        'member_level': str(profile.get('member_level', '无') or '无'),
        'car_age': int(max(math.ceil(car_age_days / 365), 0)),
        'observation_date': reference_ts.date().isoformat(),
    }

    for col in NUMERIC_COLUMNS:
        row[col] = float(row[col]) if col != 'all_times' and col != 'car_age' and col != 'last_till_now_days' else int(row[col])

    return row


def _assign_dataset_split(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.copy().reset_index(drop=True)
    feature_df['dataset'] = 'train'

    if feature_df.empty or 'churn_label' not in feature_df:
        return feature_df

    class_counts = feature_df['churn_label'].value_counts()
    if feature_df['churn_label'].nunique() < 2 or len(feature_df) < 10 or class_counts.min() < 2:
        return feature_df

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, valid_idx = next(splitter.split(feature_df, feature_df['churn_label']))
    feature_df.loc[valid_idx, 'dataset'] = 'valid'
    return feature_df


def build_training_dataset(
    customer_df: pd.DataFrame,
    history_df: pd.DataFrame,
    split_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if history_df.empty:
        raise ValueError('维修历史为空，无法构建训练数据。')

    work_history = history_df.copy()
    work_history['date'] = pd.to_datetime(work_history['date'])
    observation_ts = pd.Timestamp(split_date) if split_date is not None else work_history['date'].max() - pd.DateOffset(months=3)
    work_history = work_history[work_history['date'] <= observation_ts].copy().reset_index(drop=True)

    stale_cutoff = observation_ts - pd.DateOffset(years=3)
    active_vins = (
        work_history.groupby('VIN', as_index=False)['date']
        .max()
        .rename(columns={'date': 'max_date'})
        .query('max_date >= @stale_cutoff')['VIN']
        .tolist()
    )

    work_history = work_history[work_history['VIN'].isin(active_vins)].copy()
    filtered_customers = customer_df[customer_df['VIN'].isin(active_vins)].copy().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    profile_index = filtered_customers.set_index('VIN').to_dict(orient='index')
    for vin, visits in work_history.groupby('VIN'):
        profile = profile_index.get(vin)
        if not profile:
            continue
        row = build_feature_row({'VIN': vin, **profile}, visits, reference_date=observation_ts)
        if row is not None:
            rows.append(row)

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError('特征构建结果为空，请检查原始数据质量和过滤条件。')

    feature_df['relative_next_instore_date'] = (
        pd.to_datetime(feature_df['last_date']) + pd.to_timedelta(feature_df['day_diff_median'], unit='D')
    )
    feature_df['max_relative_next_instore_date'] = (
        pd.to_datetime(feature_df['relative_next_instore_date']) + pd.DateOffset(months=3)
    )
    feature_df['churn_label'] = (
        feature_df['max_relative_next_instore_date'] <= observation_ts
    ).astype(int)

    feature_df = _assign_dataset_split(feature_df)
    return feature_df.reset_index(drop=True)


def build_single_customer_feature_frame(
    payload: Mapping[str, Any],
    reference_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    vin = str(payload['vin'])
    profile = {
        'VIN': vin,
        'owner_type': payload.get('owner_type', '个人'),
        'car_mode': payload.get('car_mode', '未知车型'),
        'car_level': payload.get('car_level', '低档车'),
        'member_level': payload.get('member_level', '无'),
        'purchase_date': payload['purchase_date'],
    }
    visits = payload.get('visits', [])
    if len(visits) < 2:
        raise ValueError('至少需要 2 条有效进店记录才可以构建预测特征。')

    raw_visit_rows = [
        {
            'VIN': vin,
            'date': visit['date'],
            'mile': visit['mile'],
            'repair_type': visit['repair_type'],
        }
        for visit in visits
    ]
    visits_df = _clean_single_customer_history(pd.DataFrame(raw_visit_rows))
    row = build_feature_row(profile, visits_df, reference_date=reference_date)
    if row is None:
        raise ValueError('有效维修记录不足，无法构建用于预测的特征。')
    return pd.DataFrame([row])


def coerce_feature_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    row = {key: payload.get(key) for key in required_feature_columns()}
    row['VIN'] = str(payload['VIN'])
    for col in NUMERIC_COLUMNS:
        row[col] = float(payload[col])
    for col in TEXT_COLUMNS:
        row[col] = str(payload[col])
    return pd.DataFrame([row])
