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
