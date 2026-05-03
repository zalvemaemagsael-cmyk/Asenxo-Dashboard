import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASENXO Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stMetric { background: white; border-radius: 10px; padding: 16px; border: 1px solid #E5E7EB; }
    .block-container { padding-top: 1.5rem; }
    .section-header {
        font-size: 18px; font-weight: 600; color: #1E3A5F;
        border-left: 4px solid #2E75B6; padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    .kpi-positive { color: #27500A; font-weight: bold; }
    .kpi-negative { color: #A32D2D; font-weight: bold; }
    .kpi-neutral  { color: #633806; font-weight: bold; }
    .program-badge {
        background: #2E75B6; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 13px; font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIMULATE DATA ───────────────────────────────────────────────────────────
np.random.seed(42)
N = 120

PROVINCES   = ["Iloilo", "Aklan", "Capiz", "Antique", "Guimaras", "Negros Occidental"]
SECTORS     = ["Food Processing", "Metal Works", "Furniture", "Handicrafts", "Agri-Processing", "Garments"]
MSME_TYPES  = ["Micro", "Small", "Medium"]
ORG_TYPES   = ["Sole Proprietorship", "Partnership", "Corporation", "Cooperative"]

msme_names = [f"MSME-{str(i+1).zfill(3)}" for i in range(N)]

province   = np.random.choice(PROVINCES,  N)
sector     = np.random.choice(SECTORS,    N)
msme_type  = np.random.choice(MSME_TYPES, N, p=[0.60, 0.30, 0.10])
org_type   = np.random.choice(ORG_TYPES,  N)

# Pre-funding PIS values
pre_sales      = np.random.uniform(200_000,  5_000_000, N)
pre_capital    = pre_sales * np.random.uniform(0.15, 0.40, N)
pre_employment = np.random.randint(2, 50, N).astype(float)
pre_production = np.random.uniform(500, 20_000, N)
pre_assets     = pre_sales * np.random.uniform(0.20, 0.60, N)

# Post-funding PIS values (growth biased positive with noise)
growth_factor = np.random.normal(1.12, 0.25, N)   # avg 12% growth
post_sales      = np.clip(pre_sales      * growth_factor,                   50_000,  20_000_000)
post_capital    = np.clip(pre_capital    * np.random.normal(1.10, 0.20, N), 10_000,  8_000_000)
post_employment = np.clip(pre_employment * np.random.normal(1.08, 0.15, N), 1,       200)
post_production = np.clip(pre_production * np.random.normal(1.15, 0.30, N), 100,     80_000)
post_assets     = np.clip(pre_assets     * np.random.normal(1.10, 0.20, N), 10_000,  15_000_000)

# % changes
pct_sales      = (post_sales      - pre_sales)      / pre_sales      * 100
pct_capital    = (post_capital    - pre_capital)     / pre_capital    * 100
pct_employment = (post_employment - pre_employment)  / pre_employment * 100
pct_production = (post_production - pre_production)  / pre_production * 100
pct_assets     = (post_assets     - pre_assets)      / pre_assets     * 100

# DOST threshold (configurable in sidebar)
THRESHOLD = 10.0  # %

def impact_verdict(pct, threshold):
    if pct >= threshold:
        return "Positive"
    elif pct <= -threshold:
        return "Negative"
    else:
        return "Neutral"

verdict_sales      = [impact_verdict(p, THRESHOLD) for p in pct_sales]
verdict_capital    = [impact_verdict(p, THRESHOLD) for p in pct_capital]
verdict_employment = [impact_verdict(p, THRESHOLD) for p in pct_employment]
verdict_production = [impact_verdict(p, THRESHOLD) for p in pct_production]

# Overall verdict = majority across indicators
def overall_verdict(row):
    verdicts = [row["verdict_sales"], row["verdict_capital"],
                row["verdict_employment"], row["verdict_production"]]
    pos = verdicts.count("Positive")
    neg = verdicts.count("Negative")
    if pos >= 3:   return "Positive"
    elif neg >= 3: return "Negative"
    else:          return "Neutral"

# ─── REPAYMENT / PREDICTION DATA ────────────────────────────────────────────
total_months      = np.random.randint(6, 36, N)
on_time_payments  = (total_months * np.random.uniform(0.50, 1.0, N)).astype(int)
missed_payments   = total_months - on_time_payments
payment_rate      = on_time_payments / total_months * 100
consecutive_miss  = np.random.randint(0, 8, N)
months_since_last = np.random.randint(0, 12, N)

# Delinquency label: 1 = delinquent, 0 = on time
delinquent = ((consecutive_miss >= 3) | (payment_rate < 60)).astype(int)
status_label = ["Delinquent" if d else "On Time" for d in delinquent]

# Build main dataframe
df = pd.DataFrame({
    "MSME":              msme_names,
    "Province":          province,
    "Sector":            sector,
    "MSME Type":         msme_type,
    "Org Type":          org_type,
    # Pre PIS
    "pre_sales":         pre_sales,
    "pre_capital":       pre_capital,
    "pre_employment":    pre_employment,
    "pre_production":    pre_production,
    "pre_assets":        pre_assets,
    # Post PIS
    "post_sales":        post_sales,
    "post_capital":      post_capital,
    "post_employment":   post_employment,
    "post_production":   post_production,
    "post_assets":       post_assets,
    # % change
    "pct_sales":         pct_sales,
    "pct_capital":       pct_capital,
    "pct_employment":    pct_employment,
    "pct_production":    pct_production,
    "pct_assets":        pct_assets,
    # Verdicts
    "verdict_sales":     verdict_sales,
    "verdict_capital":   verdict_capital,
    "verdict_employment":verdict_employment,
    "verdict_production":verdict_production,
    # Repayment
    "total_months":      total_months,
    "on_time_payments":  on_time_payments,
    "missed_payments":   missed_payments,
    "payment_rate":      payment_rate,
    "consecutive_miss":  consecutive_miss,
    "months_since_last": months_since_last,
    "Status":            status_label,
    "delinquent":        delinquent,
})
df["overall_verdict"] = df.apply(overall_verdict, axis=1)

# ─── TRAIN CLASSIFICATION MODEL ──────────────────────────────────────────────
features = ["payment_rate", "consecutive_miss", "months_since_last",
            "pre_capital", "pre_sales", "missed_payments"]
X = df[features]
y = df["delinquent"]

clf = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
clf.fit(X, y)

df["predicted_status"]   = ["Delinquent" if p else "On Time" for p in clf.predict(X)]
df["delinquency_prob"]   = clf.predict_proba(X)[:, 1] * 100

feature_importance = pd.DataFrame({
    "Feature":    ["Payment Rate", "Consec. Misses", "Months Since Last Pmt",
                   "Working Capital", "Gross Sales", "Missed Payments"],
    "Importance": clf.feature_importances_ * 100
}).sort_values("Importance", ascending=True)

# ─── COLORS ─────────────────────────────────────────────────────────────────
COLORS = {
    "Positive":   "#27923A",
    "Neutral":    "#BA7517",
    "Negative":   "#A32D2D",
    "On Time":    "#2E75B6",
    "Delinquent": "#C94040",
    "primary":    "#2E75B6",
}

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/DOST_seal.png/240px-DOST_seal.png", width=60)
    st.markdown("### ASENXO")
    st.markdown('<span class="program-badge">DOST SETUP 4.0 iFund</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### Filters")
    sel_province = st.multiselect("Province", PROVINCES, default=PROVINCES)
    sel_sector   = st.multiselect("Sector",   SECTORS,   default=SECTORS)
    sel_type     = st.multiselect("MSME Type",MSME_TYPES,default=MSME_TYPES)

    st.markdown("---")
    st.markdown("#### DOST Impact Threshold")
    threshold = st.slider("% Change Required for Positive/Negative Impact",
                          min_value=5, max_value=30, value=10, step=1)
    st.caption(f"≥ +{threshold}% = Positive   |   ≤ -{threshold}% = Negative")

    st.markdown("---")
    page = st.radio("Navigate", ["Overview", "Impact Assessment", "Delinquency Prediction"])

# Apply filters
fdf = df[
    df["Province"].isin(sel_province) &
    df["Sector"].isin(sel_sector) &
    df["MSME Type"].isin(sel_type)
].copy()

# Recompute verdicts with sidebar threshold
fdf["verdict_sales"]      = fdf["pct_sales"].apply(lambda x: impact_verdict(x, threshold))
fdf["verdict_capital"]    = fdf["pct_capital"].apply(lambda x: impact_verdict(x, threshold))
fdf["verdict_employment"] = fdf["pct_employment"].apply(lambda x: impact_verdict(x, threshold))
fdf["verdict_production"] = fdf["pct_production"].apply(lambda x: impact_verdict(x, threshold))
fdf["overall_verdict"]    = fdf.apply(overall_verdict, axis=1)

total    = len(fdf)
pos_pct  = round(fdf["overall_verdict"].eq("Positive").sum() / total * 100, 1)
neu_pct  = round(fdf["overall_verdict"].eq("Neutral").sum()  / total * 100, 1)
neg_pct  = round(fdf["overall_verdict"].eq("Negative").sum() / total * 100, 1)
delinq_n = fdf["predicted_status"].eq("Delinquent").sum()
delinq_r = round(delinq_n / total * 100, 1)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## 📊 ASENXO Program Overview")
    st.markdown(f"**Region VI — DOST SETUP 4.0 iFund Program** &nbsp;|&nbsp; {total} beneficiaries in view")
    st.markdown("---")

    # KPI Row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Beneficiaries", total)
    c2.metric("Positive Impact", f"{pos_pct}%",  f"{int(fdf['overall_verdict'].eq('Positive').sum())} MSMEs")
    c3.metric("Neutral Impact",   f"{neu_pct}%",  f"{int(fdf['overall_verdict'].eq('Neutral').sum())} MSMEs")
    c4.metric("Negative Impact",  f"{neg_pct}%",  f"{int(fdf['overall_verdict'].eq('Negative').sum())} MSMEs")
    c5.metric("Delinquency Risk", f"{delinq_r}%", f"{delinq_n} flagged")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Overall Impact Distribution</div>', unsafe_allow_html=True)
        verdict_counts = fdf["overall_verdict"].value_counts().reset_index()
        verdict_counts.columns = ["Verdict", "Count"]
        fig = px.pie(verdict_counts, names="Verdict", values="Count",
                     color="Verdict",
                     color_discrete_map=COLORS,
                     hole=0.45)
        fig.update_layout(margin=dict(t=20, b=20), height=300,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Impact by Province</div>', unsafe_allow_html=True)
        prov_df = fdf.groupby(["Province", "overall_verdict"]).size().reset_index(name="Count")
        fig2 = px.bar(prov_df, x="Province", y="Count", color="overall_verdict",
                      color_discrete_map=COLORS, barmode="stack")
        fig2.update_layout(margin=dict(t=20, b=20), height=300,
                           xaxis_title="", yaxis_title="",
                           legend_title="Impact")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Average % Change per Indicator</div>', unsafe_allow_html=True)
        avg_changes = pd.DataFrame({
            "Indicator": ["Gross Sales", "Working Capital", "Employment", "Production Volume", "Assets"],
            "Avg % Change": [
                fdf["pct_sales"].mean(),
                fdf["pct_capital"].mean(),
                fdf["pct_employment"].mean(),
                fdf["pct_production"].mean(),
                fdf["pct_assets"].mean(),
            ]
        })
        colors = ["#27923A" if v >= 0 else "#A32D2D" for v in avg_changes["Avg % Change"]]
        fig3 = px.bar(avg_changes, x="Avg % Change", y="Indicator",
                      orientation="h", color="Indicator",
                      color_discrete_sequence=colors)
        fig3.add_vline(x=threshold,  line_dash="dash", line_color="#27923A",
                       annotation_text=f"+{threshold}%", annotation_position="top right")
        fig3.add_vline(x=-threshold, line_dash="dash", line_color="#A32D2D",
                       annotation_text=f"-{threshold}%", annotation_position="top left")
        fig3.update_layout(margin=dict(t=20, b=20), height=300,
                           showlegend=False, xaxis_title="% Change")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Impact by Sector</div>', unsafe_allow_html=True)
        sec_df = fdf.groupby(["Sector", "overall_verdict"]).size().reset_index(name="Count")
        fig4 = px.bar(sec_df, x="Count", y="Sector", color="overall_verdict",
                      color_discrete_map=COLORS, orientation="h", barmode="stack")
        fig4.update_layout(margin=dict(t=20, b=20), height=300,
                           xaxis_title="", yaxis_title="",
                           legend_title="Impact")
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: IMPACT ASSESSMENT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Impact Assessment":
    st.markdown("## 📈 Impact Assessment")
    st.markdown(f"Comparing pre-funding baseline vs. latest post-funding PIS report | Threshold: **±{threshold}%**")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessed MSMEs", total)
    c2.metric("Positive Impact", f"{int(fdf['overall_verdict'].eq('Positive').sum())}",
              f"{pos_pct}% of total")
    c3.metric("Neutral",         f"{int(fdf['overall_verdict'].eq('Neutral').sum())}",
              f"{neu_pct}% of total")
    c4.metric("Negative Impact", f"{int(fdf['overall_verdict'].eq('Negative').sum())}",
              f"{neg_pct}% of total")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Per-Indicator Verdict Distribution</div>', unsafe_allow_html=True)
        indicators = {
            "Gross Sales":      "verdict_sales",
            "Working Capital":  "verdict_capital",
            "Employment":       "verdict_employment",
            "Production Volume":"verdict_production",
        }
        rows = []
        for ind_name, col_name in indicators.items():
            for v in ["Positive", "Neutral", "Negative"]:
                rows.append({"Indicator": ind_name, "Verdict": v,
                              "Count": fdf[col_name].eq(v).sum()})
        ind_df = pd.DataFrame(rows)
        fig5 = px.bar(ind_df, x="Indicator", y="Count", color="Verdict",
                      color_discrete_map=COLORS, barmode="group")
        fig5.update_layout(margin=dict(t=20, b=20), height=320,
                           xaxis_title="", legend_title="Verdict")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sales Growth Distribution</div>', unsafe_allow_html=True)
        fig6 = px.histogram(fdf, x="pct_sales", nbins=30,
                            color_discrete_sequence=[COLORS["primary"]])
        fig6.add_vline(x=threshold,  line_dash="dash", line_color="#27923A",
                       annotation_text=f"+{threshold}%")
        fig6.add_vline(x=-threshold, line_dash="dash", line_color="#A32D2D",
                       annotation_text=f"-{threshold}%")
        fig6.add_vline(x=0, line_dash="dot", line_color="gray")
        fig6.update_layout(margin=dict(t=20, b=20), height=320,
                           xaxis_title="% Change in Gross Sales", yaxis_title="# of MSMEs")
        st.plotly_chart(fig6, use_container_width=True)

    # Scatter: pre vs post sales
    st.markdown('<div class="section-header">Pre vs. Post Gross Sales per MSME</div>', unsafe_allow_html=True)
    fig7 = px.scatter(fdf, x="pre_sales", y="post_sales",
                      color="overall_verdict", color_discrete_map=COLORS,
                      hover_data=["MSME", "Province", "Sector", "pct_sales"],
                      labels={"pre_sales": "Pre-Funding Sales (₱)",
                              "post_sales": "Post-Funding Sales (₱)",
                              "overall_verdict": "Impact"})
    max_val = max(fdf["pre_sales"].max(), fdf["post_sales"].max())
    fig7.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                   line=dict(dash="dot", color="gray"))
    fig7.update_layout(margin=dict(t=20, b=20), height=350)
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("Points above the diagonal line improved. Points below declined.")

    # Per-MSME table
    st.markdown('<div class="section-header">Per-MSME Impact Breakdown</div>', unsafe_allow_html=True)
    verdict_filter = st.selectbox("Filter by Verdict", ["All", "Positive", "Neutral", "Negative"])
    table_df = fdf.copy()
    if verdict_filter != "All":
        table_df = table_df[table_df["overall_verdict"] == verdict_filter]

    display_cols = {
        "MSME": "MSME",
        "Province": "Province",
        "Sector": "Sector",
        "pct_sales": "Sales Δ%",
        "pct_capital": "Capital Δ%",
        "pct_employment": "Employment Δ%",
        "pct_production": "Production Δ%",
        "overall_verdict": "Overall Impact"
    }
    tbl = table_df[list(display_cols.keys())].rename(columns=display_cols)
    for col in ["Sales Δ%", "Capital Δ%", "Employment Δ%", "Production Δ%"]:
        tbl[col] = tbl[col].apply(lambda x: f"{x:+.1f}%")
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=300)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: DELINQUENCY PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Delinquency Prediction":
    st.markdown("## ⚠️ Payment Delinquency Prediction")
    st.markdown("Classification model trained on repayment history to flag MSMEs at risk of non-payment.")
    st.markdown("---")

    # KPI row
    on_time_n = fdf["predicted_status"].eq("On Time").sum()
    high_risk  = fdf[fdf["delinquency_prob"] >= 75]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total MSMEs",        total)
    c2.metric("Predicted On Time",  on_time_n,  f"{round(on_time_n/total*100,1)}%")
    c3.metric("Predicted Delinquent", delinq_n, f"{delinq_r}% of total")
    c4.metric("High Risk (≥75%)",   len(high_risk), "immediate attention")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Predicted Status Distribution</div>', unsafe_allow_html=True)
        status_counts = fdf["predicted_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig8 = px.pie(status_counts, names="Status", values="Count",
                      color="Status",
                      color_discrete_map={"On Time": COLORS["On Time"],
                                         "Delinquent": COLORS["Delinquent"]},
                      hole=0.45)
        fig8.update_layout(margin=dict(t=20, b=20), height=300,
                           legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fig9 = px.bar(feature_importance, x="Importance", y="Feature",
                      orientation="h",
                      color_discrete_sequence=[COLORS["primary"]])
        fig9.update_layout(margin=dict(t=20, b=20), height=300,
                           xaxis_title="Importance (%)", yaxis_title="")
        st.plotly_chart(fig9, use_container_width=True)

    # Delinquency probability distribution
    st.markdown('<div class="section-header">Delinquency Risk Score Distribution</div>', unsafe_allow_html=True)
    fig10 = px.histogram(fdf, x="delinquency_prob", nbins=20,
                         color="predicted_status",
                         color_discrete_map={"On Time": COLORS["On Time"],
                                            "Delinquent": COLORS["Delinquent"]})
    fig10.add_vline(x=75, line_dash="dash", line_color="#A32D2D",
                    annotation_text="High Risk (75%)")
    fig10.add_vline(x=50, line_dash="dot", line_color="#BA7517",
                    annotation_text="Threshold (50%)")
    fig10.update_layout(margin=dict(t=20, b=20), height=300,
                        xaxis_title="Delinquency Probability (%)",
                        yaxis_title="# of MSMEs", legend_title="Predicted Status")
    st.plotly_chart(fig10, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Delinquency Risk by Province</div>', unsafe_allow_html=True)
        prov_risk = fdf.groupby("Province")["delinquency_prob"].mean().reset_index()
        prov_risk.columns = ["Province", "Avg Risk %"]
        prov_risk = prov_risk.sort_values("Avg Risk %", ascending=True)
        fig11 = px.bar(prov_risk, x="Avg Risk %", y="Province",
                       orientation="h",
                       color="Avg Risk %",
                       color_continuous_scale=["#2E75B6", "#FAC775", "#C94040"])
        fig11.update_layout(margin=dict(t=20, b=20), height=300,
                            xaxis_title="Avg Delinquency Risk (%)",
                            coloraxis_showscale=False)
        st.plotly_chart(fig11, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Risk by MSME Type</div>', unsafe_allow_html=True)
        type_risk = fdf.groupby("MSME Type")["delinquency_prob"].mean().reset_index()
        type_risk.columns = ["MSME Type", "Avg Risk %"]
        fig12 = px.bar(type_risk, x="MSME Type", y="Avg Risk %",
                       color="MSME Type",
                       color_discrete_sequence=["#2E75B6", "#BA7517", "#27923A"])
        fig12.update_layout(margin=dict(t=20, b=20), height=300,
                            xaxis_title="", yaxis_title="Avg Risk (%)",
                            showlegend=False)
        st.plotly_chart(fig12, use_container_width=True)

    # High risk table
    st.markdown('<div class="section-header">⚠️ High Risk MSMEs — Flagged for Immediate Attention</div>',
                unsafe_allow_html=True)
    risk_thresh = st.slider("Show MSMEs with delinquency risk above:", 50, 90, 75, step=5)
    flagged = fdf[fdf["delinquency_prob"] >= risk_thresh].copy()
    flagged = flagged.sort_values("delinquency_prob", ascending=False)

    st.caption(f"{len(flagged)} MSMEs flagged at ≥{risk_thresh}% delinquency risk")

    flag_display = flagged[[
        "MSME", "Province", "Sector", "MSME Type",
        "payment_rate", "consecutive_miss", "delinquency_prob", "predicted_status"
    ]].rename(columns={
        "payment_rate":      "Payment Rate %",
        "consecutive_miss":  "Consec. Misses",
        "delinquency_prob":  "Risk Score %",
        "predicted_status":  "Predicted Status"
    })
    flag_display["Payment Rate %"] = flag_display["Payment Rate %"].apply(lambda x: f"{x:.1f}%")
    flag_display["Risk Score %"]   = flag_display["Risk Score %"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(flag_display.reset_index(drop=True), use_container_width=True, height=300)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("ASENXO · DOST SETUP 4.0 iFund Program · Region VI · Simulated Data for Development Purposes Only")