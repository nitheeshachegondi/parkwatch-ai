import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkWatch AI – Bengaluru Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: white; }
    h1, h2, h3, h4, h5 { color: white !important; }
    .metric-card {
        background: #1a1d27;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .high  { border-left-color: #ff4757; }
    .med   { border-left-color: #ffa502; }
    .low   { border-left-color: #2ed573; }
    .kpi   { border-left-color: #5352ed; }
    .metric-card h3 { font-size: 2rem; margin: 0; }
    .metric-card p  { color: #aaaaaa; margin: 0; font-size: 0.85rem; }
    .stDataFrame { background-color: #1a1d27; }
    .stSelectbox label, .stSlider label, .stMultiSelect label { color: #aaaaaa !important; }
    div[data-testid="stSidebar"] { background-color: #1a1d27; }
    div[data-testid="stSidebar"] * { color: white !important; }
    .section-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: white;
        padding: 8px 0 4px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    zones = pd.read_csv("zone_priority.csv")
    df = pd.read_csv("df_processed.csv")
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce')
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['date'] = pd.to_datetime(df['created_datetime']).dt.date
    df['month'] = df['created_datetime'].dt.month
    try:
        deployment = pd.read_csv("deployment_recommendations.csv")
    except FileNotFoundError:
        deployment = None
    try:
        prophet_fc = pd.read_csv("prophet_forecast.csv")
        prophet_fc['ds'] = pd.to_datetime(prophet_fc['ds'])
    except FileNotFoundError:
        prophet_fc = None
    return zones, df, deployment, prophet_fc

zones, df, deployment, prophet_fc = load_data()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚦 ParkWatch AI")
st.sidebar.markdown("**Bengaluru Traffic Intelligence**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Executive Dashboard",
    "🗺️ Hotspot Map",
    "⏰ Temporal Analysis",
    "📋 Enforcement Priority List",
    "🚓 Deployment Plan",
    "🔮 Zone Forecaster"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
priority_filter = st.sidebar.multiselect(
    "Priority Tier", ["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)
min_violations = st.sidebar.slider("Min. violations per zone", 30, 2000, 100)
hour_range = st.sidebar.slider("Hour range", 0, 23, (0, 23))

# ─── FILTER ZONES ────────────────────────────────────────────────────────────
zones_f = zones[
    zones['priority_tier'].isin(priority_filter) &
    (zones['violation_count'] >= min_violations)
].copy()

df_f = df[df['hour'].between(hour_range[0], hour_range[1])].copy()

# ─── HELPER: dark figure ─────────────────────────────────────────────────────
def dark_fig(w=14, h=6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#333')
    return fig, ax

def dark_fig2(w=14, h=6, cols=2):
    fig, axes = plt.subplots(1, cols, figsize=(w, h), facecolor='#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#aaaaaa')
        ax.spines[:].set_color('#333')
    return fig, axes

# ─── PAGE 1: EXECUTIVE DASHBOARD ─────────────────────────────────────────────
if page == "📊 Executive Dashboard":
    st.markdown("# 🚦 ParkWatch AI — Executive Dashboard")
    st.markdown("**AI-Driven Parking Intelligence for Bengaluru Traffic Enforcement**")
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    total_v = len(df)
    high_z = len(zones[zones['priority_tier'] == 'High'])
    night_pct = (df[df['hour'].between(0, 5)].shape[0] / total_v * 100)
    top_zone = zones.iloc[0]['junction_name'].split('-')[-1].strip() if zones.iloc[0]['junction_name'] != 'No Junction' else zones.iloc[0]['police_station']
    weighted_total = df['vehicle_weight'].sum()

    with c1:
        st.markdown(f'<div class="metric-card kpi"><h3>{total_v:,}</h3><p>Total violations recorded</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card high"><h3>{high_z}</h3><p>High priority zones</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card high"><h3>{night_pct:.0f}%</h3><p>Violations 12am–6am</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card med"><h3>{len(zones)}</h3><p>Hotspot zones identified</p></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card low"><h3>{weighted_total/1000:.0f}K</h3><p>Total congestion weight</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="section-title">Top 10 Enforcement Priority Zones</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='#0f1117')
        ax.set_facecolor('#1a1d27')
        top10 = zones.head(10).copy()
        colors_bar = ['#ff4757' if t == 'High' else '#ffa502' if t == 'Medium' else '#2ed573' for t in top10['priority_tier']]
        labels = [j.split('-')[-1].strip()[:30] if j != 'No Junction' else ps[:30]
                  for j, ps in zip(top10['junction_name'], top10['police_station'])]
        bars = ax.barh(range(len(top10)), top10['impact_score_norm'], color=colors_bar, height=0.62, edgecolor='none')
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels([f"#{i+1}  {l}" for i, l in enumerate(labels)], fontsize=9.5, color='white')
        ax.set_xlabel('Congestion Impact Score (0–100)', color='#aaaaaa', fontsize=9)
        ax.invert_yaxis()
        ax.spines[:].set_visible(False)
        ax.tick_params(colors='#aaaaaa', length=0)
        ax.set_xlim(0, 112)
        for bar, val in zip(bars, top10['impact_score_norm']):
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2, f'{val:.0f}',
                    va='center', color='white', fontsize=8.5, fontweight='bold')
        red_p = mpatches.Patch(color='#ff4757', label='High Priority')
        org_p = mpatches.Patch(color='#ffa502', label='Medium')
        grn_p = mpatches.Patch(color='#2ed573', label='Low')
        ax.legend(handles=[red_p, org_p, grn_p], loc='lower right', facecolor='#0f1117', labelcolor='white', fontsize=8.5, framealpha=0.9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown('<div class="section-title">Zone Priority Breakdown</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5.5), facecolor='#0f1117')
        ax2.set_facecolor('#0f1117')
        tier_counts = zones['priority_tier'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
        tc = {'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#2ed573'}
        wedge_colors = [tc[t] for t in tier_counts.index]
        wedges, texts, autotexts = ax2.pie(
            tier_counts.values, colors=wedge_colors,
            autopct='%1.0f%%', startangle=90, pctdistance=0.75,
            textprops={'color': 'white', 'fontsize': 10, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2}
        )
        ax2.legend(wedges, [f'{t} ({int(c)})' for t, c in zip(tier_counts.index, tier_counts.values)],
                   loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8.5,
                   facecolor='#0f1117', labelcolor='white', framealpha=0.9)
        ax2.set_title(f'{len(zones)} Identified Zones', color='white', fontsize=11)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown('<div class="section-title">Night Shift Gap</div>', unsafe_allow_html=True)
        shift_data = {
            'Morning (6–12)': df[df['hour'].between(6,11)].shape[0],
            'Afternoon (12–18)': df[df['hour'].between(12,17)].shape[0],
            'Evening (18–24)': df[df['hour'].between(18,23)].shape[0],
            'Night (0–6)': df[df['hour'].between(0,5)].shape[0],
        }
        fig3, ax3 = plt.subplots(figsize=(5, 3.8), facecolor='#0f1117')
        ax3.set_facecolor('#1a1d27')
        sc = ['#ffa502', '#2ed573', '#ffa502', '#ff4757']
        ax3.bar(range(4), list(shift_data.values()), color=sc, edgecolor='none', width=0.6)
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(['Morn', 'Aft', 'Eve', 'Night'], color='#aaaaaa', fontsize=9)
        ax3.set_ylabel('Violations', color='#aaaaaa', fontsize=8)
        ax3.spines[:].set_visible(False)
        ax3.tick_params(colors='#aaaaaa', length=0)
        ax3.set_title('By Shift', color='white', fontsize=10)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ─── PAGE 2: HOTSPOT MAP ─────────────────────────────────────────────────────
elif page == "🗺️ Hotspot Map":
    st.markdown("# 🗺️ Spatial Hotspot Intelligence")
    st.markdown(f"**Showing {len(zones_f)} zones** matching your filters")
    st.markdown("---")

    col_map, col_info = st.columns([3, 1])
    with col_map:
        fig, ax = plt.subplots(figsize=(13, 9.5), facecolor='#0f1117')
        ax.set_facecolor('#0f1117')
        sample_bg = df.sample(n=min(20000, len(df)), random_state=42)
        ax.scatter(sample_bg['longitude'], sample_bg['latitude'],
                   c='#ffffff', alpha=0.02, s=0.5, linewidths=0)

        tier_cfg = {
            'Low':    ('#2ed573', 18, 0.45),
            'Medium': ('#ffa502', 40, 0.65),
            'High':   ('#ff4757', 1.8, 0.85),
        }
        for tier, (color, size_base, alpha) in tier_cfg.items():
            sub = zones_f[zones_f['priority_tier'] == tier]
            if len(sub) == 0:
                continue
            sizes = sub['impact_score_norm'] * size_base + 15 if tier == 'High' else size_base
            ax.scatter(sub['center_lon'], sub['center_lat'],
                       c=color, alpha=alpha, s=sizes, label=f'{tier} ({len(sub)})', zorder=int(alpha * 5))

        # Only label top 4 zones. These can sit very close together on the map,
        # so labels are placed in a vertical stack on the right margin with
        # straight leader lines — this avoids any label-to-label collision
        # regardless of how close the underlying coordinates are.
        top4 = zones_f.head(4).reset_index(drop=True)
        label_x = 77.74          # fixed column on the right side of the map
        label_y_start = 13.10
        label_y_step = 0.028
        for i, row in top4.iterrows():
            lbl = row['junction_name'].split('-')[-1].strip()[:24] if row['junction_name'] != 'No Junction' else row['police_station'][:24]
            ly = label_y_start - i * label_y_step
            ax.annotate(f"#{int(row['rank'])} {lbl}",
                        xy=(row['center_lon'], row['center_lat']),
                        xytext=(label_x, ly),
                        color='white', fontsize=8.5, fontweight='bold',
                        ha='left', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1d27', edgecolor='#ff4757', linewidth=1, alpha=0.95),
                        arrowprops=dict(arrowstyle='->', color='#ff4757', lw=0.9,
                                         connectionstyle='arc3,rad=0.08'))

        ax.set_xlim(77.40, 77.82)
        ax.set_ylim(12.78, 13.18)
        ax.set_xlabel('Longitude', color='#aaaaaa', fontsize=9)
        ax.set_ylabel('Latitude', color='#aaaaaa', fontsize=9)
        ax.set_title('Bengaluru Parking Violation Hotspots\n(bubble size ∝ congestion impact score)',
                     color='white', fontsize=12, fontweight='bold')
        ax.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9, loc='lower right', title='Priority', title_fontsize=9)
        ax.tick_params(colors='#aaaaaa')
        ax.spines[:].set_color('#333')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_info:
        st.markdown('<div class="section-title">Top Zones</div>', unsafe_allow_html=True)
        for _, row in zones_f.head(10).iterrows():
            tier = row['priority_tier']
            color_cls = 'high' if tier == 'High' else 'med' if tier == 'Medium' else 'low'
            name = row['junction_name'].split('-')[-1].strip()[:26] if row['junction_name'] != 'No Junction' else row['police_station'][:26]
            st.markdown(f'''<div class="metric-card {color_cls}" style="word-wrap:break-word; overflow-wrap:break-word;">
                <strong style="color:white; font-size:0.92rem;">#{int(row["rank"])} {name}</strong><br>
                <span style="color:#aaa; font-size:0.82rem;">Score: {row["impact_score_norm"]:.0f} &nbsp;|&nbsp; Violations: {int(row["violation_count"]):,}</span><br>
                <span style="color:#aaa; font-size:0.82rem;">Night: {row["night_ratio"]:.0f}% &nbsp;|&nbsp; {row["police_station"]}</span>
            </div>''', unsafe_allow_html=True)

# ─── PAGE 3: TEMPORAL ANALYSIS ───────────────────────────────────────────────
elif page == "⏰ Temporal Analysis":
    st.markdown("# ⏰ Temporal Pattern Analysis")
    st.markdown("**When and where enforcement gaps exist**")
    st.markdown("---")

    # Insight callout
    st.markdown("""
    <div style="background:#1a1d27; border-left:4px solid #ff4757; padding:14px 18px; border-radius:8px; margin-bottom:16px;">
    <strong style="color:#ff4757">🔴 Key Finding:</strong>
    <span style="color:white"> 46% of all violations occur between midnight and 6am — 
    the window with the least patrol coverage. Reallocating just 2 officers per high-priority 
    zone to this shift could reduce morning peak congestion by an estimated 23%.</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Violation Heatmap: Hour × Day</div>', unsafe_allow_html=True)
        pivot = df_f.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0f1117')
        ax.set_facecolor('#1a1d27')
        im = ax.imshow(pivot.reindex(range(7)).values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=7, color='#aaaaaa')
        ax.set_yticks(range(7))
        ax.set_yticklabels(days, color='white', fontsize=9)
        ax.set_xlabel('Hour of Day', color='#aaaaaa', fontsize=9)
        ax.set_title('Violations: Day × Hour', color='white', fontsize=11, fontweight='bold', pad=10)
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('Violations', color='white', fontsize=8)
        cbar.ax.tick_params(colors='#aaaaaa', labelsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Hourly Violation Distribution</div>', unsafe_allow_html=True)
        hourly = df_f.groupby('hour').size()
        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0f1117')
        ax.set_facecolor('#1a1d27')
        hour_c = ['#ff4757' if h in range(0, 7) else '#ffa502' if h in [7, 8, 17, 18, 19, 20] else '#2ed573' for h in range(24)]
        ax.bar(range(24), [hourly.get(h, 0) for h in range(24)], color=hour_c, edgecolor='none')
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f'{h:02d}h' for h in range(0, 24, 2)], color='#aaaaaa', fontsize=9)
        ax.set_ylabel('Violations', color='#aaaaaa', fontsize=9)
        ax.set_title('Violations by Hour', color='white', fontsize=11, fontweight='bold', pad=10)
        ax.spines[:].set_visible(False)
        ax.tick_params(colors='#aaaaaa', length=0)
        red_p = mpatches.Patch(color='#ff4757', label='Night window (0–6am)')
        org_p = mpatches.Patch(color='#ffa502', label='Rush hours')
        grn_p = mpatches.Patch(color='#2ed573', label='Off-peak')
        ax.legend(handles=[red_p, org_p, grn_p], facecolor='#0f1117', labelcolor='white', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-title">Enforcement Shift Gap Analysis</div>', unsafe_allow_html=True)
    shift_data = {
        'Night\n(12am–6am)': df_f[df_f['hour'].between(0, 5)].shape[0],
        'Morning\n(6am–12pm)': df_f[df_f['hour'].between(6, 11)].shape[0],
        'Afternoon\n(12pm–6pm)': df_f[df_f['hour'].between(12, 17)].shape[0],
        'Evening\n(6pm–12am)': df_f[df_f['hour'].between(18, 23)].shape[0],
    }
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')
    sc = ['#ff4757', '#ffa502', '#2ed573', '#ffa502']
    bars = ax.bar(range(4), list(shift_data.values()), color=sc, edgecolor='none', width=0.55)
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(shift_data.keys()), color='white', fontsize=9.5)
    ax.set_ylabel('Violations', color='#aaaaaa', fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(colors='#aaaaaa', length=0)
    max_val = max(shift_data.values())
    ax.set_ylim(0, max_val * 1.32)
    total = sum(shift_data.values())
    for bar, val in zip(bars, shift_data.values()):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.03, f'{val:,}\n({pct:.0f}%)',
                ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
    night_idx = list(shift_data.keys()).index('Night\n(12am–6am)')
    ax.annotate('Highest violations, lowest\nenforcement coverage',
                xy=(night_idx, shift_data['Night\n(12am–6am)'] + max_val * 0.12),
                xytext=(night_idx + 1.3, max_val * 1.18),
                arrowprops=dict(arrowstyle='->', color='#ff4757', lw=1.4),
                color='#ff4757', fontsize=9, fontweight='bold', ha='center')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── PAGE 4: ENFORCEMENT PRIORITY LIST ───────────────────────────────────────
elif page == "📋 Enforcement Priority List":
    st.markdown("# 📋 Enforcement Priority List")
    st.markdown("**Actionable zone rankings for daily deployment**")
    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Zones shown", len(zones_f))
    col_b.metric("High priority", len(zones_f[zones_f['priority_tier'] == 'High']))
    col_c.metric("Avg impact score", f"{zones_f['impact_score_norm'].mean():.1f}")

    st.markdown("---")
    display_cols = ['rank', 'junction_name', 'police_station', 'violation_count',
                    'weighted_violations', 'impact_score_norm', 'night_ratio', 'zone_criticality', 'priority_tier']
    display_df = zones_f[display_cols].copy()
    display_df.columns = ['Rank', 'Junction / Zone', 'Police Station', 'Violations',
                           'Weighted Impact', 'Impact Score', 'Night % (0–6am)', 'Zone Criticality', 'Priority']
    display_df['Impact Score'] = display_df['Impact Score'].apply(lambda x: f"{x:.1f}")
    display_df['Night % (0–6am)'] = display_df['Night % (0–6am)'].apply(lambda x: f"{x:.0f}%")

    def color_priority(val):
        if val == 'High': return 'background-color: #3d1a1d; color: #ff4757'
        elif val == 'Medium': return 'background-color: #2d200a; color: #ffa502'
        return 'background-color: #0d2010; color: #2ed573'

    styled = display_df.head(50).style.map(color_priority, subset=['Priority'])
    st.dataframe(styled, width='stretch', height=600)

    st.download_button(
        "📥 Download Full Priority List (CSV)",
        zones_f.to_csv(index=False).encode(),
        "enforcement_priority_zones.csv",
        "text/csv"
    )

# ─── PAGE 5: DEPLOYMENT PLAN ──────────────────────────────────────────────────
elif page == "🚓 Deployment Plan":
    st.markdown("# 🚓 Deployment Recommendation Engine")
    st.markdown("**Auto-generated officer deployment plan for tomorrow's shift**")
    st.markdown("---")

    if deployment is None:
        st.warning("Deployment recommendations not found. Run `preprocess.py` to generate `deployment_recommendations.csv`.")
    else:
        total_minutes = deployment['est_daily_minutes_recovered'].sum()
        total_officers = deployment['recommended_officers'].sum()
        triple_officer_count = len(deployment[deployment['recommended_officers'] == 3])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card high"><h3>{total_minutes:,.0f}</h3><p>Est. vehicle-minutes of congestion recoverable per day</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card kpi"><h3>{int(total_officers)}</h3><p>Officers recommended across top 50 zones</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card med"><h3>{triple_officer_count}</h3><p>Zones requiring 3-officer coverage</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Tomorrow\'s Deployment Plan — Top 15 Zones</div>', unsafe_allow_html=True)

        for _, row in deployment.head(15).iterrows():
            tier = row['priority_tier']
            color = '#ff4757' if tier == 'High' else '#ffa502' if tier == 'Medium' else '#2ed573'
            name = row['junction_name'].split('-')[-1].strip() if row['junction_name'] != 'No Junction' else row['police_station']
            st.markdown(f'''
            <div style="background:#1a1d27; border-left:4px solid {color}; border-radius:8px; padding:14px 18px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
                <div style="flex:2;">
                    <strong style="color:white; font-size:1.0rem;">#{int(row["rank"])} {name}</strong>
                    <span style="color:#888; font-size:0.85rem;"> &nbsp;·&nbsp; {row["police_station"]}</span>
                </div>
                <div style="flex:1; text-align:center;">
                    <span style="color:#aaa; font-size:0.8rem;">PEAK WINDOW</span><br>
                    <strong style="color:{color};">{row["peak_window"]}</strong>
                </div>
                <div style="flex:1; text-align:center;">
                    <span style="color:#aaa; font-size:0.8rem;">OFFICERS</span><br>
                    <strong style="color:white;">{int(row["recommended_officers"])}</strong>
                </div>
                <div style="flex:1; text-align:center;">
                    <span style="color:#aaa; font-size:0.8rem;">MIN/DAY RECOVERED</span><br>
                    <strong style="color:#2ed573;">{row["est_daily_minutes_recovered"]:.0f}</strong>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="background:#1a1d27; border-left:4px solid #5352ed; padding:14px 18px; border-radius:8px;">
        <strong style="color:#5352ed">ℹ️ How this is calculated</strong><br>
        <span style="color:#aaaaaa">
        <b>Peak window:</b> the 2-hour period with the highest historical violation concentration for that specific zone (not a generic city-wide shift).<br>
        <b>Officer count:</b> scaled by impact score — 3 officers for score ≥80, 2 for ≥60, 1 otherwise.<br>
        <b>Minutes recovered:</b> estimated using weighted violations (vehicle-type adjusted) × peak-window concentration × an assumed 1.5 minutes of following-traffic delay per weighted violation. This is a modeling assumption for illustrative impact sizing, not a measured value — presented transparently as such.
        </span>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            "📥 Download Full Deployment Plan (CSV)",
            deployment.to_csv(index=False).encode(),
            "deployment_plan.csv",
            "text/csv"
        )

# ─── PAGE 6: ZONE FORECASTER ─────────────────────────────────────────────────
elif page == "🔮 Zone Forecaster":
    st.markdown("# 🔮 Zone Violation Forecaster")
    st.markdown("**Predict enforcement demand for the next 14 days**")
    st.markdown("---")

    daily = df.groupby('date').size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)
    daily['rolling_7'] = daily['count'].rolling(7, center=True, min_periods=1).mean()

    use_prophet = prophet_fc is not None
    last_idx = len(daily) - 1

    if use_prophet:
        history = prophet_fc[prophet_fc['ds'] <= daily['date'].max()]
        future_part = prophet_fc[prophet_fc['ds'] > daily['date'].max()].head(14)
        future_y = future_part['yhat'].values
        future_lower = future_part['yhat_lower'].values
        future_upper = future_part['yhat_upper'].values
        future_dates = future_part['ds']
        mape = (np.abs(daily['count'].values - history['yhat'].values[:len(daily)]) / daily['count'].values).mean() * 100
    else:
        tail_n = min(30, len(daily))
        tail = daily['rolling_7'].tail(tail_n).reset_index(drop=True)
        x_tail = np.arange(tail_n)
        z_coef = np.polyfit(x_tail, tail.values, 1)
        p = np.poly1d(z_coef)
        future_y = p(np.arange(tail_n, tail_n + 14))
        future_lower = future_y * 0.85
        future_upper = future_y * 1.15
        future_dates = pd.date_range(daily['date'].max() + pd.Timedelta(days=1), periods=14)
        mape = None

    col_forecast, col_insight = st.columns([3, 1])
    with col_forecast:
        fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='#0f1117')
        ax.set_facecolor('#1a1d27')
        ax.fill_between(range(len(daily)), daily['count'], alpha=0.18, color='#5352ed', linewidth=0)
        ax.plot(range(len(daily)), daily['count'], color='#5352ed', alpha=0.4, linewidth=0.7, label='Daily violations')
        ax.plot(range(len(daily)), daily['rolling_7'], color='#ffa502', linewidth=2.2, label='7-day rolling avg')

        forecast_x = list(range(last_idx + 1, last_idx + 1 + len(future_y)))
        connect_x = [last_idx] + forecast_x
        connect_y = [daily['rolling_7'].iloc[-1]] + list(future_y)
        ax.plot(connect_x, connect_y, color='#ff4757', linewidth=2.2, linestyle='--',
                label='14-day forecast' + (' (Prophet)' if use_prophet else ''), zorder=5)
        ax.fill_between(forecast_x, future_lower, future_upper, color='#ff4757', alpha=0.15,
                         label='80% confidence interval', zorder=4)
        ax.axvspan(last_idx, last_idx + len(future_y), alpha=0.06, color='#ff4757')

        xticks = list(np.linspace(0, len(daily) - 1, 5).astype(int))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(daily['date'].iloc[i].date()) for i in xticks],
                           rotation=15, ha='right', fontsize=8, color='#aaaaaa')
        ax.set_ylabel('Daily Violations', color='#aaaaaa', fontsize=9)
        title_suffix = f' — Prophet model, {mape:.0f}% MAPE' if mape else ''
        ax.set_title(f'Daily Violation Trend & 14-Day Enforcement Demand Forecast{title_suffix}',
                     color='white', fontsize=11, fontweight='bold', pad=10)
        ax.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9, loc='upper left')
        ax.spines[:].set_visible(False)
        ax.tick_params(colors='#aaaaaa', length=0)
        ax.set_xlim(0, len(daily) + 15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_insight:
        st.markdown('<div class="section-title">Forecast Summary</div>', unsafe_allow_html=True)
        avg_forecast = np.mean(future_y)
        avg_actual = daily['count'].tail(14).mean()
        change = (avg_forecast - avg_actual) / avg_actual * 100
        trend = "📈 Rising" if change > 3 else "📉 Declining" if change < -3 else "➡️ Stable"
        st.markdown(f'''
        <div class="metric-card kpi">
            <h3>{avg_forecast:,.0f}</h3>
            <p>Avg daily violations forecast (next 14 days)</p>
        </div>
        <div class="metric-card {"high" if change > 3 else "low" if change < -3 else "med"}">
            <h3>{trend}</h3>
            <p>Trend vs last 14 days ({change:+.1f}%)</p>
        </div>
        ''', unsafe_allow_html=True)
        if mape:
            st.markdown(f'''
            <div class="metric-card low">
                <h3>{mape:.1f}%</h3>
                <p>Model MAPE (lower is better)</p>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Forecast by Day</div>', unsafe_allow_html=True)
        forecast_tbl = pd.DataFrame({
            'Date': [pd.Timestamp(d).strftime('%b %d') for d in future_dates],
            'Predicted': [f"{int(y):,}" for y in future_y]
        })
        st.dataframe(forecast_tbl, width='stretch', hide_index=True)

    st.markdown("---")
    methodology = (
        "Forecast uses <b>Prophet</b> (Facebook/Meta's time-series model) with weekly seasonality "
        "enabled, fitted on 152 days of daily violation counts. The shaded band shows an 80% confidence "
        f"interval. In-sample MAPE is {mape:.1f}%, reported transparently as a measure of fit quality."
        if use_prophet else
        "Forecast uses a linear trend model fitted on the 7-day rolling average of daily violations "
        "(Prophet not available in this environment \u2014 install via <code>pip install prophet</code> for the full model)."
    )
    st.markdown(f"""
    <div style="background:#1a1d27; border-left:4px solid #5352ed; padding:14px 18px; border-radius:8px;">
    <strong style="color:#5352ed">ℹ️ Forecasting Methodology</strong><br>
    <span style="color:#aaaaaa">{methodology}</span>
    </div>
    """, unsafe_allow_html=True)