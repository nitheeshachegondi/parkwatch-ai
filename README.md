# 🚦 ParkWatch AI — Bengaluru Traffic Intelligence Platform

## Overview
ParkWatch AI is an end-to-end machine learning system that detects illegal parking hotspots
across Bengaluru, quantifies their congestion impact, and generates prioritized enforcement
recommendations for traffic police.

**Theme:** Poor Visibility on Parking-Induced Congestion  
**Dataset:** Jan–May Police Violation Data (298,450 records)  
**Hackathon:** Bengaluru Traffic AI Challenge — HackerEarth

---

## 🏆 Key Innovations

1. **Congestion Impact Score** — Novel weighted formula combining vehicle type, zone criticality, 
   and recurrence rate. A truck parked at a junction scores 8× higher than a scooter.

2. **Night Enforcement Gap Discovery** — 46% of violations occur between midnight and 6am, 
   the window with least patrol coverage. First insight of this kind from this dataset.

3. **872 Spatial Hotspots** — Grid-based spatial clustering (300m cells) across all 298k records, 
   each with a ranked impact score and enforcement priority tier.

4. **14-Day Forecasting** — Daily violation demand prediction using rolling trend analysis 
   to enable proactive resource planning.

---

## 📊 Solution Architecture

```
Raw Violation Data (298,450 records)
        │
        ▼
  Data Preprocessing
  ├── Datetime parsing
  ├── Vehicle weight mapping (LGV=8x, Car=1.5x, Scooter=1x)
  └── Spatial grid binning (300m cells)
        │
        ▼
  Spatial Clustering (Grid-based, 872 zones)
        │
        ▼
  Congestion Impact Score
  = weighted_violations × zone_criticality × recurrence_rate
        │
        ▼
  Priority Ranking + Temporal Analysis
        │
        ▼
  Streamlit Dashboard (5 pages)
  ├── Executive Dashboard
  ├── Hotspot Map
  ├── Temporal Analysis
  ├── Enforcement Priority List
  └── Zone Forecaster
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd parkwatch_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place the file: jan_to_may_police_violation_anonymized791b166.csv
# Run the preprocessing script to generate zone_priority.csv and df_processed.csv:
python preprocess.py

# 4. Launch the app
streamlit run app.py
```

The app will open at http://localhost:8501

---

## 📁 File Structure

```
parkwatch_app/
├── app.py                  # Main Streamlit application
├── preprocess.py           # Data preprocessing & ML pipeline
├── requirements.txt        # Python dependencies
├── zone_priority.csv       # Pre-computed hotspot zones (872 zones)
├── df_processed.csv        # Processed violations dataset
└── README.md               # This file
```

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Total violations analyzed | 298,450 |
| Hotspot zones identified | 872 |
| High-priority zones | ~5% |
| Night window violation share | 46% |
| Top zone (Safina Plaza) violations | 8,377 |
| Top zone impact score | 100.0 |

---

## 🔬 Methodology

### Congestion Impact Score Formula
```
Impact Score = weighted_violations × zone_criticality × recurrence_rate

where:
  weighted_violations = Σ(vehicle_weight × violation)
  vehicle_weight: LGV/HGV=8, Bus=6, Maxi-cab=4, Auto=2, Car=1.5, Scooter=1
  zone_criticality: named junction=3, unnamed zone=2
  recurrence_rate = zone_violations / mean_violations_per_zone
```

### Why vehicle weighting matters
A single LGV illegally parked on a main road blocks equivalent carriageway space
as 8 scooters. Traditional violation counts treat them equally — ParkWatch AI does not.

---

## 👩‍💻 Built for
HackerEarth Bengaluru Traffic AI Hackathon — Round 2 Prototype Phase  
Deadline: June 21, 2026
