"""
ParkWatch AI — Data Preprocessing & ML Pipeline
Run this script once to generate zone_priority.csv and df_processed.csv
from the raw violation dataset.

Usage: python preprocess.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ParkWatch AI — Preprocessing Pipeline")
print("=" * 60)

# ─── 1. LOAD ─────────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv(
    'jan_to_may_police_violation_anonymized791b166.csv',
    usecols=['id', 'latitude', 'longitude', 'vehicle_type',
             'police_station', 'junction_name', 'created_datetime']
)
print(f"      Loaded {len(df):,} records")

# ─── 2. CLEAN ────────────────────────────────────────────────────────────────
print("\n[2/5] Cleaning & feature engineering...")
df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce')
df = df.dropna(subset=['latitude', 'longitude'])
df['hour'] = df['created_datetime'].dt.hour
df['day_of_week'] = df['created_datetime'].dt.dayofweek
df['month'] = df['created_datetime'].dt.month
df['date'] = df['created_datetime'].dt.date

# Vehicle congestion weights
vehicle_weight_map = {
    'LGV': 8, 'HGV': 8, 'TANKER': 8, 'LORRY': 8,
    'PRIVATE BUS': 6, 'BUS': 6, 'BUS (BMTC/KSRTC)': 6,
    'MAXI-CAB': 4, 'VAN': 4,
    'PASSENGER AUTO': 2, 'GOODS AUTO': 2, 'AUTO': 2, 'TEMPO': 2,
    'CAR': 1.5, 'JEEP': 1.5,
    'SCOOTER': 1, 'MOTOR CYCLE': 1, 'MOPED': 1,
}
df['vehicle_weight'] = df['vehicle_type'].map(vehicle_weight_map).fillna(1.0)
print(f"      Features added. Valid records: {len(df):,}")

# ─── 3. SPATIAL GRID CLUSTERING ──────────────────────────────────────────────
print("\n[3/5] Spatial grid clustering (300m cells)...")
GRID = 0.003  # ~300 metres
df['grid_lat'] = (df['latitude'] / GRID).round(0) * GRID
df['grid_lon'] = (df['longitude'] / GRID).round(0) * GRID
df['grid_id'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)

# ─── 4. AGGREGATE PER ZONE ───────────────────────────────────────────────────
print("\n[4/5] Computing congestion impact scores...")
agg = df.groupby('grid_id').agg(
    violation_count=('id', 'count'),
    weighted_violations=('vehicle_weight', 'sum'),
    center_lat=('grid_lat', 'first'),
    center_lon=('grid_lon', 'first'),
    police_station=('police_station', lambda x: x.mode()[0]),
    junction_name=('junction_name',
                   lambda x: x[x != 'No Junction'].mode()[0]
                   if (x != 'No Junction').any() else 'No Junction'),
).reset_index()

# Minimum 30 violations to qualify as a hotspot
agg = agg[agg['violation_count'] >= 30].copy()

# Night ratio (midnight–6am)
night_counts = df[df['hour'].between(0, 5)].groupby('grid_id').size().rename('night_count')
agg = agg.merge(night_counts, on='grid_id', how='left')
agg['night_count'] = agg['night_count'].fillna(0)
agg['night_ratio'] = (agg['night_count'] / agg['violation_count'] * 100).round(1)

# Zone criticality (named junctions score higher)
agg['zone_criticality'] = agg['junction_name'].apply(lambda x: 3 if x != 'No Junction' else 2)

# Recurrence rate
agg['recurrence_rate'] = (agg['violation_count'] / agg['violation_count'].mean()).round(3)

# CONGESTION IMPACT SCORE
agg['impact_score'] = (
    agg['weighted_violations'] *
    agg['zone_criticality'] *
    agg['recurrence_rate']
).round(2)

# Normalize 0–100
mn, mx = agg['impact_score'].min(), agg['impact_score'].max()
agg['impact_score_norm'] = ((agg['impact_score'] - mn) / (mx - mn) * 100).round(1)

# Sort and rank
agg = agg.sort_values('impact_score_norm', ascending=False).reset_index(drop=True)
agg['rank'] = agg.index + 1

# Priority tier
agg['priority_tier'] = pd.cut(
    agg['impact_score_norm'],
    bins=[-0.1, 33, 66, 100.1],
    labels=['Low', 'Medium', 'High']
)

# ─── 5. SAVE ─────────────────────────────────────────────────────────────────
print("\n[5/5] Saving outputs...")
agg.to_csv('zone_priority.csv', index=False)
df.to_csv('df_processed.csv', index=False)

print(f"\n✅ Done!")
print(f"   zone_priority.csv — {len(agg)} hotspot zones")
print(f"   df_processed.csv  — {len(df):,} processed records")
print(f"\n   Top 5 zones by impact score:")
top5 = agg.head(5)
for _, row in top5.iterrows():
    name = row['junction_name'].split('-')[-1].strip()[:40] if row['junction_name'] != 'No Junction' else row['police_station']
    print(f"   #{int(row['rank'])}. {name} | Score: {row['impact_score_norm']:.1f} | Violations: {int(row['violation_count']):,}")
print()

# ─── 6. DEPLOYMENT RECOMMENDATION ENGINE ─────────────────────────────────────
print("[6/7] Building deployment recommendation engine...")
df['merge_key'] = df.apply(
    lambda r: r['junction_name'] if r['junction_name'] != 'No Junction'
              else f"{r['police_station']}_{r['grid_id']}", axis=1
)
top_zones = agg.head(50).copy()
recommendations = []
for _, zone in top_zones.iterrows():
    zone_data = df[df['merge_key'] == zone['merge_key']]
    if len(zone_data) == 0:
        continue
    hourly_counts = zone_data.groupby('hour').size().reindex(range(24), fill_value=0)

    best_start, best_count = 0, -1
    for h in range(24):
        window_count = sum(hourly_counts.get((h + i) % 24, 0) for i in range(2))
        if window_count > best_count:
            best_count = window_count
            best_start = h
    window_label = f"{best_start:02d}:00\u2013{(best_start+2)%24:02d}:00"
    window_pct = best_count / zone['violation_count'] * 100 if zone['violation_count'] > 0 else 0

    if zone['impact_score_norm'] >= 80:
        officers = 3
    elif zone['impact_score_norm'] >= 60:
        officers = 2
    else:
        officers = 1

    recommendations.append({
        'rank': zone['rank'], 'junction_name': zone['junction_name'],
        'police_station': zone['police_station'], 'impact_score_norm': zone['impact_score_norm'],
        'priority_tier': zone['priority_tier'], 'recommended_officers': officers,
        'peak_window': window_label, 'peak_window_pct': round(window_pct, 1),
        'violation_count': zone['violation_count'], 'night_ratio': zone['night_ratio'],
        'weighted_violations': zone['weighted_violations'],
    })

rec_df = pd.DataFrame(recommendations)
MINUTES_PER_WEIGHTED_VIOLATION = 1.5
rec_df['est_daily_minutes_recovered'] = (
    rec_df['weighted_violations'] * (rec_df['peak_window_pct'] / 100) * MINUTES_PER_WEIGHTED_VIOLATION / 152
).round(0)
rec_df.to_csv('deployment_recommendations.csv', index=False)
print(f"   Saved deployment_recommendations.csv ({len(rec_df)} zones)")
print(f"   Est. total daily minutes recoverable (top 50): {rec_df['est_daily_minutes_recovered'].sum():,.0f}")

# ─── 7. PROPHET FORECAST ─────────────────────────────────────────────────────
print("\n[7/7] Training Prophet forecasting model...")
try:
    import logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    from prophet import Prophet

    daily = df.groupby('date').size().reset_index(name='y')
    daily.columns = ['ds', 'y']
    daily['ds'] = pd.to_datetime(daily['ds'])

    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False,
                changepoint_prior_scale=0.1, interval_width=0.8)
    m.fit(daily)
    future = m.make_future_dataframe(periods=14)
    forecast = m.predict(future)
    forecast.to_csv('prophet_forecast.csv', index=False)

    merged = daily.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    mape = ((merged['y'] - merged['yhat']).abs() / merged['y']).mean() * 100
    print(f"   Saved prophet_forecast.csv (MAPE: {mape:.1f}%)")
except ImportError:
    print("   Prophet not installed \u2014 skipping. Run: pip install prophet")