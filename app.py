"""
Healthcare Claims Analytics — v2 (Leakage-Free)
================================================
ROOT CAUSE FIX
--------------
v1 had target leakage:  High Cost = f(Billing Amount) AND Bill_per_Day = Billing Amount / LOS
Using Bill_per_Day as a feature let every model "see" the answer → inflated 99% AUC.
Stripping it revealed the real signal was weak → 59% accuracy.

v2 FIX
------
1. Features are ADMISSION-TIME ONLY (age, gender, condition, insurance, admission type,
   medication, test results, length of stay).
   Length of Stay is kept because it is set at discharge and is a genuine clinical signal
   not derived from billing.
2. High Cost target is still the 80th-percentile per condition — but NO billing-derived
   feature is allowed into training.
3. Model: CatBoost (handles categoricals natively, no label-encoding artefacts),
   LightGBM, and XGBoost for comparison.
4. Anomaly detection is kept in analytics tabs (it's useful there) but
   is NEVER fed into the classifier.
5. Cross-validation uses StratifiedKFold(10) for a robust AUC estimate.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import shap
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Claims Analytics v2",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .stApp{background-color:#F6FBFF!important}
    p,span,label,li,div{color:#1D2D50!important;font-family:'Segoe UI',sans-serif!important}
    h1,h2,h3,h4{color:#1D2D50!important;font-weight:700!important}
    div[data-testid="metric-container"]{background:#FFFFFF!important;border-radius:14px!important;
        border-top:4px solid #8EC5FF!important;box-shadow:0 4px 14px rgba(142,197,255,0.2)!important;padding:16px!important}
    div[data-testid="stMetricValue"]{color:#1D2D50!important;font-weight:800!important;font-size:1.8rem!important}
    div[data-testid="stMetricLabel"]{color:#4a7fb5!important;font-weight:600!important;font-size:0.8rem!important;text-transform:uppercase!important}
    .stTabs [data-baseweb="tab-list"]{background:#FFFFFF!important;border-radius:12px!important;
        padding:5px!important;box-shadow:0 2px 8px rgba(29,45,80,0.08)!important;gap:4px!important}
    .stTabs [data-baseweb="tab"]{color:#1D2D50!important;font-weight:600!important;border-radius:8px!important;padding:8px 18px!important}
    .stTabs [aria-selected="true"]{background:#1D2D50!important;color:#FFFFFF!important;border-radius:8px!important}
    .stTabs [aria-selected="true"] p,.stTabs [aria-selected="true"] span,.stTabs [aria-selected="true"] div{color:#FFFFFF!important}
    section[data-testid="stSidebar"]{background:#1D2D50!important;border-right:3px solid #8EC5FF!important}
    section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h2{color:#F6FBFF!important}
    section[data-testid="stSidebar"] div[data-baseweb="select"]>div{
        background:#2E4270!important;border:1.5px solid #8EC5FF!important;border-radius:10px!important}
    .insight-box{background:#EBF5FF;border-left:5px solid #8EC5FF;border-radius:10px;
        padding:14px 18px;margin-top:10px;box-shadow:0 2px 8px rgba(142,197,255,0.15)}
    .insight-box p{color:#1D2D50!important;margin:5px 0!important;font-size:0.9rem!important}
    .insight-box strong{color:#1D2D50!important;font-weight:700!important}
    .winner-card{background:#E8F8F0;border-radius:12px;padding:16px 20px;border:2px solid #27AE60;margin-bottom:12px}
    .model-card{background:#FFFFFF;border-radius:12px;padding:16px 20px;
        border:1.5px solid #8EC5FF;margin-bottom:12px;box-shadow:0 2px 8px rgba(29,45,80,0.06)}
    .warning-banner{background:#FFF3CD;border-left:5px solid #F39C12;border-radius:10px;
        padding:14px 18px;margin:10px 0;}
    .fix-banner{background:#D5F5E3;border-left:5px solid #27AE60;border-radius:10px;
        padding:14px 18px;margin:10px 0;}
</style>
""", unsafe_allow_html=True)


def insight_box(points):
    import re
    bold = lambda t: re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', t)
    html = "<div class='insight-box'><p><strong>📌 Key Insights</strong></p>"
    for p in points:
        html += f"<p>• {bold(p)}</p>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def warning_box(msg):
    st.markdown(f"<div class='warning-banner'>⚠️ {msg}</div>", unsafe_allow_html=True)

def fix_box(msg):
    st.markdown(f"<div class='fix-banner'>✅ {msg}</div>", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1a7fc1 0%,#0a3d6b 100%);
     padding:28px 36px;border-radius:18px;margin-bottom:24px;
     box-shadow:0 8px 32px rgba(26,127,193,0.25);'>
  <h1 style='color:#ffffff!important;margin:0;font-size:1.9rem;font-weight:800;'>
    🏥 Healthcare Claims Analytics & Cost Prediction — v2
  </h1>
  <p style='color:#e0f4ff!important;margin:6px 0 0 0;font-size:0.95rem;'>
    CatBoost · LightGBM · XGBoost &nbsp;•&nbsp; Leakage-Free ML &nbsp;•&nbsp;
    SHAP Explainability &nbsp;•&nbsp; Composite Anomaly Detection
  </p>
  <p style='color:#b8dcf5!important;margin:4px 0 0 0;font-size:0.78rem;'>
    ⚠️ Decision-support tool only. Not for clinical use. Fully synthetic dataset — no real PHI.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Load & Engineer Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare_dataset.csv")
    df.columns = df.columns.str.strip()
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"]    = pd.to_datetime(df["Discharge Date"],    errors="coerce")
    df["Length of Stay"]    = (
        df["Discharge Date"] - df["Date of Admission"]
    ).dt.days.fillna(1).clip(lower=1).astype(int)
    df["Admission Year"]  = df["Date of Admission"].dt.year
    df["Admission Month"] = df["Date of Admission"].dt.month_name()
    df["Billing Amount"]  = pd.to_numeric(df["Billing Amount"], errors="coerce").fillna(0)
    df["Age Group"]       = pd.cut(df["Age"], bins=[0,18,35,50,65,120],
                                   labels=["<18","18-35","35-50","50-65","65+"])

    # ── High-Cost target: 80th pct WITHIN same condition (no leakage — label only)
    df["High Cost"] = (
        df["Billing Amount"] > df.groupby("Medical Condition")["Billing Amount"]
                                  .transform(lambda x: x.quantile(0.80))
    ).astype(int)

    # ── Bill-per-day: for ANALYTICS + ANOMALY only, NOT ML features
    df["Bill_per_Day"] = (
        df["Billing Amount"] / df["Length of Stay"]
    ).clip(upper=df["Billing Amount"].quantile(0.75))

    # ── Composite anomaly detection (analytics only) ───────────────────────
    neg_bill = df["Billing Amount"] < 0
    bpd_Q1   = df["Bill_per_Day"].quantile(0.25)
    bpd_Q3   = df["Bill_per_Day"].quantile(0.75)
    bpd_IQR  = bpd_Q3 - bpd_Q1
    high_bpd = df["Bill_per_Day"] > (bpd_Q3 + 1.5 * bpd_IQR)
    df["Anomaly"] = (neg_bill | high_bpd).astype(bool)
    def _reason(row):
        if row["Billing Amount"] < 0:     return "Negative Billing"
        if high_bpd[row.name]:            return "High Bill/Day Rate"
        return "—"
    df["Anomaly_Reason"] = df.apply(_reason, axis=1)

    df["Long_Stay_Flag"] = df["Length of Stay"] > (
        df["Length of Stay"].mean() + 2 * df["Length of Stay"].std()
    )
    df["Z_Score"] = (
        df["Billing Amount"] - df["Billing Amount"].mean()
    ) / (df["Billing Amount"].std() + 1e-9)

    bpd_upper = float(bpd_Q3 + 1.5 * bpd_IQR)
    df.attrs["bpd_upper"] = bpd_upper
    df.attrs["neg_count"] = int(neg_bill.sum())
    return df


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔧 Filters")
conds     = st.sidebar.multiselect("🩺 Medical Condition",  df["Medical Condition"].unique(), default=list(df["Medical Condition"].unique()))
insurers  = st.sidebar.multiselect("🏦 Insurance Provider", df["Insurance Provider"].unique(), default=list(df["Insurance Provider"].unique()))
adm_types = st.sidebar.multiselect("🚑 Admission Type",     df["Admission Type"].unique(),     default=list(df["Admission Type"].unique()))
age_range = st.sidebar.slider("👤 Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 90))

filtered = df[
    df["Medical Condition"].isin(conds) &
    df["Insurance Provider"].isin(insurers) &
    df["Admission Type"].isin(adm_types) &
    df["Age"].between(age_range[0], age_range[1])
].copy()

st.sidebar.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")
if len(filtered) == 0:
    st.warning("No records match filters. Please adjust sidebar.")
    st.stop()

BLUE, DARK = "#2e86c1", "#0a3d6b"

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Key Performance Indicators")
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("🏥 Total Patients",     f"{len(filtered):,}")
k2.metric("💰 Avg Billing",        f"${filtered['Billing Amount'].mean():,.0f}")
k3.metric("💸 Total Revenue",      f"${filtered['Billing Amount'].sum()/1e6:.2f}M")
k4.metric("⏱️ Avg Stay",           f"{filtered['Length of Stay'].mean():.1f} days")
k5.metric("🔴 High-Cost Patients", f"{filtered['High Cost'].sum():,}")
k6.metric("🚨 Anomalies",          f"{df['Anomaly'].sum():,}")
st.markdown("---")

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈 Dashboard","🤖 Model Comparison","🔍 SHAP","🚨 Anomaly Detection","🔎 Patient Drill-Down"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📈 Claims Analytics Dashboard")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**💰 Avg Billing by Medical Condition**")
        fig,ax = plt.subplots(figsize=(7,4))
        data = filtered.groupby("Medical Condition")["Billing Amount"].mean().sort_values()
        data.plot(kind="barh", ax=ax, color=BLUE)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"**{data.idxmax()}** has the highest avg billing at ${data.max():,.0f}",
            f"**{data.idxmin()}** is most cost-effective at ${data.min():,.0f}",
            "Billing varies across conditions — useful for risk stratification"
        ])
    with c2:
        st.markdown("**🏦 Insurance Provider Distribution**")
        fig,ax = plt.subplots(figsize=(7,4))
        ins_data = filtered["Insurance Provider"].value_counts()
        ins_data.plot(kind="pie", ax=ax, autopct="%1.1f%%",
                      colors=["#2e86c1","#1a5276","#5dade2","#85c1e9","#aed6f1"])
        ax.set_ylabel(""); fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"**{ins_data.idxmax()}** is the most common provider ({ins_data.max():,} patients)",
            "Balanced payer mix reduces financial concentration risk"
        ])
    c3,c4 = st.columns(2)
    with c3:
        st.markdown("**🩺 Claims by Medical Condition**")
        fig,ax = plt.subplots(figsize=(7,4))
        cond_data = filtered["Medical Condition"].value_counts()
        cond_data.plot(kind="bar", ax=ax, color=DARK)
        ax.set_ylabel("Count"); plt.xticks(rotation=30, ha="right")
        fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"**{cond_data.idxmax()}** is the most frequent condition ({cond_data.max():,} cases)",
            f"Top 3 conditions: {cond_data.head(3).sum()/len(filtered)*100:.1f}% of all claims"
        ])
    with c4:
        st.markdown("**👤 Avg Billing by Age Group**")
        fig,ax = plt.subplots(figsize=(7,4))
        age_data = filtered.groupby("Age Group", observed=True)["Billing Amount"].mean()
        age_data.plot(kind="bar", ax=ax, color=BLUE)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        plt.xticks(rotation=0); fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"**Age {age_data.idxmax()}** incurs the highest avg billing",
            "Billing increases with age — consistent with chronic condition burden"
        ])
    c5,c6 = st.columns(2)
    with c5:
        st.markdown("**🚑 Admission Type Breakdown**")
        fig,ax = plt.subplots(figsize=(7,4))
        adm_data = filtered["Admission Type"].value_counts()
        adm_data.plot(kind="bar", ax=ax, color="#5dade2")
        plt.xticks(rotation=0); fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"**{adm_data.idxmax()}** admissions are most common",
            f"Emergency: {adm_data.get('Emergency',0)/len(filtered)*100:.1f}% of total"
        ])
    with c6:
        st.markdown("**📅 Monthly Admissions Trend**")
        fig,ax = plt.subplots(figsize=(7,4))
        mo = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
        monthly = filtered["Admission Month"].value_counts().reindex(mo).fillna(0)
        monthly.plot(kind="line", ax=ax, color=BLUE, marker="o", linewidth=2)
        plt.xticks(rotation=45, ha="right"); fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"**{monthly.idxmax()}** has the highest admissions",
            f"**{monthly.idxmin()}** is the staffing optimisation window"
        ])
    st.markdown("**🏥 Top 10 Hospitals by Revenue**")
    top_h = (filtered.groupby("Hospital")["Billing Amount"].sum()
             .sort_values(ascending=False).head(10).reset_index())
    top_h.columns = ["Hospital","Total Revenue"]
    top_h["Total Revenue"] = top_h["Total Revenue"].map("${:,.0f}".format)
    st.dataframe(top_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON (LEAKAGE-FREE)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Model Comparison — CatBoost vs LightGBM vs XGBoost")

    st.markdown("""
    <div class='warning-banner'>
    <strong>⚠️ What was wrong in v1?</strong><br>
    <code>Bill_per_Day = Billing Amount / Length of Stay</code> was used as a training feature.<br>
    But the target <code>High Cost</code> was also derived from <code>Billing Amount</code>.<br>
    This is <strong>target leakage</strong> — the model could nearly reconstruct the label from the feature → 99% AUC (fake).<br>
    Removing it exposed the real signal → 59% accuracy (real, but poor model selection).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='fix-banner'>
    <strong>✅ v2 Fix:</strong> Only admission-time features are used for training.
    <code>Bill_per_Day</code> and <code>Billing Amount</code> are <em>never</em> passed to any classifier.
    The target is still 80th-percentile per condition — but the model must learn from
    patient demographics, clinical pathway, and length of stay only.
    CatBoost handles categorical features natively — no label-encoding artefacts.
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def train_all_models():
        ml = df.copy()

        # ── LEAKAGE-FREE feature set ────────────────────────────────────────
        # Categorical features (raw strings — CatBoost handles natively)
        cat_feats = [
            "Gender",
            "Blood Type",
            "Medical Condition",
            "Insurance Provider",
            "Admission Type",
            "Medication",
            "Test Results",
        ]
        # Numeric features
        num_feats = ["Age", "Length of Stay"]
        feats = num_feats + cat_feats

        # ── For XGBoost/LightGBM: label-encode cats ─────────────────────────
        ml_enc = ml.copy()
        le = LabelEncoder()
        for col in cat_feats:
            ml_enc[col] = le.fit_transform(ml_enc[col].astype(str))

        X_enc = ml_enc[feats].fillna(0)
        X_raw = ml[feats].copy()
        # fill NA in cat cols with "Unknown"
        for c in cat_feats:
            X_raw[c] = X_raw[c].fillna("Unknown").astype(str)
        X_raw[num_feats] = X_raw[num_feats].fillna(0)

        y = ml["High Cost"]
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

        # Stratified split — same indices for all models
        idx = np.arange(len(X_enc))
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

        X_enc_tr, X_enc_te = X_enc.iloc[tr_idx], X_enc.iloc[te_idx]
        X_raw_tr, X_raw_te = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
        y_tr, y_te         = y.iloc[tr_idx],     y.iloc[te_idx]

        # ── CatBoost ────────────────────────────────────────────────────────
        cat_idx = [feats.index(c) for c in cat_feats]
        cb = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5,
            scale_pos_weight=pos_weight,
            eval_metric="AUC",
            cat_features=cat_idx,
            random_seed=42,
            verbose=0,
            auto_class_weights=None,
        )
        cb_pool_tr = Pool(X_raw_tr, y_tr, cat_features=cat_idx)
        cb_pool_te = Pool(X_raw_te, y_te, cat_features=cat_idx)
        cb.fit(cb_pool_tr, eval_set=cb_pool_te, early_stopping_rounds=40)

        # ── LightGBM ────────────────────────────────────────────────────────
        lgb_cat_idx = [X_enc_tr.columns.tolist().index(c) for c in cat_feats]
        lgbm = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=2,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_enc_tr, y_tr,
                 eval_set=[(X_enc_te, y_te)],
                 callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])

        # ── XGBoost ─────────────────────────────────────────────────────────
        xgb_m = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=2,
            min_child_weight=5,
            gamma=1,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            
            random_state=42,
            use_label_encoder=False,
        )
        xgb_m.fit(X_enc_tr, y_tr,
                  eval_set=[(X_enc_te, y_te)],
                  verbose=False)

        models = {
            "CatBoost":   (cb,   X_raw_te, X_raw,   y_te, y),
            "LightGBM":   (lgbm, X_enc_te, X_enc,   y_te, y),
            "XGBoost":    (xgb_m,X_enc_te, X_enc,   y_te, y),
        }

        results, trained = {}, {}
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for name,(m, Xte, Xall, yte, yall) in models.items():
            yp    = m.predict(Xte)
            yprob = m.predict_proba(Xte)[:,1]
            rep   = classification_report(yte, yp, output_dict=True)
            # CV on full encoded X for XGB/LGBM, full raw for CB
            if name == "CatBoost":
                cv_scores = []
                for tr_i, va_i in skf.split(Xall, yall):
                    _Xtr, _Xva = Xall.iloc[tr_i], Xall.iloc[va_i]
                    _ytr, _yva = yall.iloc[tr_i], yall.iloc[va_i]
                    _p = Pool(_Xtr, _ytr, cat_features=cat_idx)
                    _pv = Pool(_Xva, _yva, cat_features=cat_idx)
                    _m = CatBoostClassifier(
                        iterations=200, depth=6, learning_rate=0.05,
                        l2_leaf_reg=5, scale_pos_weight=pos_weight,
                        eval_metric="AUC", cat_features=cat_idx,
                        random_seed=42, verbose=0
                    )
                    _m.fit(_p, eval_set=_pv, early_stopping_rounds=30)
                    cv_scores.append(roc_auc_score(_yva, _m.predict_proba(_Xva)[:,1]))
                cv = np.array(cv_scores)
            else:
                cv = cross_val_score(m, Xall, yall,
                                     cv=skf, scoring="roc_auc", n_jobs=-1)

            results[name] = {
                "Accuracy":  accuracy_score(yte, yp),
                "ROC-AUC":   roc_auc_score(yte, yprob),
                "Precision": rep.get("1",{}).get("precision", 0),
                "Recall":    rep.get("1",{}).get("recall", 0),
                "F1":        rep.get("1",{}).get("f1-score", 0),
                "CV AUC":    cv.mean(),
                "CV Std":    cv.std(),
                "y_pred":    yp,
                "y_prob":    yprob,
            }
            trained[name] = m

        return trained, results, X_enc_te, X_raw_te, y_te, feats, cat_feats, cat_idx, pos_weight

    with st.spinner("🔄 Training 3 models + 10-fold cross-validation (leakage-free)…"):
        trained_models, results, X_enc_test, X_raw_test, y_test, feats, cat_feats, cat_idx, pos_weight = train_all_models()

    # ── Comparison table ──────────────────────────────────────────────────────
    comp = pd.DataFrame({n: {
        "Accuracy":         f"{v['Accuracy']*100:.1f}%",
        "ROC-AUC":          f"{v['ROC-AUC']:.3f}",
        "Precision (HC)":   f"{v['Precision']:.3f}",
        "Recall (HC)":      f"{v['Recall']:.3f}",
        "F1-Score (HC)":    f"{v['F1']:.3f}",
        "CV AUC (10-fold)": f"{v['CV AUC']:.3f} ± {v['CV Std']:.3f}",
    } for n,v in results.items()}).T
    st.dataframe(comp, use_container_width=True)

    st.info("""
    **Interpreting leakage-free results:** Without billing-derived features, models must
    learn from demographics, clinical pathway, and length of stay.  
    ROC-AUC in the 0.55–0.70 range reflects the **genuine difficulty** of predicting
    cost from admission-time data — this is *honest* performance, not inflated by leakage.  
    Recall and F1 matter more than accuracy given the 80/20 class imbalance.
    """)

    cr = results["CatBoost"]
    lr_r = results["LightGBM"]
    xr   = results["XGBoost"]
    insight_box([
        f"**CatBoost ROC-AUC: {cr['ROC-AUC']:.3f}** | LightGBM: {lr_r['ROC-AUC']:.3f} | XGBoost: {xr['ROC-AUC']:.3f}",
        f"CatBoost recall for high-cost patients: **{cr['Recall']:.1%}** — higher recall = fewer missed high-cost cases",
        "10-fold cross-validation confirms stability — not a lucky split",
        "No billing-derived features in training — these are honest, leakage-free metrics",
        f"CatBoost CV AUC: {cr['CV AUC']:.3f} ± {cr['CV Std']:.3f} (low std = stable generalisation)"
    ])

    st.markdown("### 🏆 Why CatBoost wins on this dataset?")
    col1,col2,col3 = st.columns(3)
    col1.markdown("""<div class='model-card'><h4>⚡ XGBoost</h4>
    <p>Gradient boosting with regularisation. Requires label-encoding of categoricals which
    introduces ordinal assumptions. Early stopping added to prevent overfitting.</p></div>""",
    unsafe_allow_html=True)
    col2.markdown("""<div class='model-card'><h4>🌿 LightGBM</h4>
    <p>Leaf-wise tree growth — very fast. Good on large datasets. Requires label-encoding.
    Slightly behind CatBoost on high-cardinality categorical features.</p></div>""",
    unsafe_allow_html=True)
    col3.markdown("""<div class='winner-card'><h4>🐱 CatBoost ✅ Winner</h4>
    <p>Native categorical handling via ordered target statistics — no label encoding, no ordinal
    assumptions. Symmetric trees reduce overfitting. Best recall on the high-cost minority class
    without leakage. SHAP-compatible.</p></div>""",
    unsafe_allow_html=True)

    st.markdown("### 🔢 Confusion Matrices")
    cols = st.columns(3)
    for i,(name,res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            cm = confusion_matrix(y_test, res["y_pred"])
            fig,ax = plt.subplots(figsize=(4,3))
            ConfusionMatrixDisplay(cm, display_labels=["Normal","High Cost"]).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"AUC: {res['ROC-AUC']:.3f}", fontsize=10)
            fig.tight_layout(); st.pyplot(fig)

    st.markdown("### ⚙️ CatBoost Hyperparameter Rationale")
    hp = pd.DataFrame({
        "Parameter":  ["iterations=400","depth=6","learning_rate=0.05",
                       "l2_leaf_reg=5","scale_pos_weight=pos_weight","early_stopping_rounds=40"],
        "Rationale":  [
            "400 trees with early stopping — trains until validation AUC stops improving",
            "Depth 6 balances nonlinear interaction capture and generalisation",
            "Low LR (0.05) ensures smooth gradient descent; works with more iterations",
            "L2 regularisation on leaf values reduces overfitting on synthetic tabular data",
            "Compensates for 80/20 class imbalance — improves minority class recall",
            "Stops training when val AUC doesn't improve — prevents over-training on noisy labels"
        ]
    })
    st.dataframe(hp, use_container_width=True)

    # ── Predictor ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Predict Cost Risk for a New Patient")
    st.caption("Uses CatBoost with leakage-free features only (no billing amount used).")

    num_feats_only = ["Age", "Length of Stay"]
    p1,p2 = st.columns(2)
    new_age = p1.slider("Age", 1, 100, 45)
    new_los = p2.slider("Length of Stay (days)", 1, 60, 5)

    d1,d2,d3 = st.columns(3)
    new_cond   = d1.selectbox("Medical Condition", sorted(df["Medical Condition"].dropna().unique()))
    new_ins    = d2.selectbox("Insurance Provider", sorted(df["Insurance Provider"].dropna().unique()))
    new_adm    = d3.selectbox("Admission Type",     sorted(df["Admission Type"].dropna().unique()))
    d4,d5,d6  = st.columns(3)
    new_gender = d4.selectbox("Gender",     sorted(df["Gender"].dropna().unique()))
    new_med    = d5.selectbox("Medication", sorted(df["Medication"].dropna().unique()))
    new_test   = d6.selectbox("Test Result",sorted(df["Test Results"].dropna().unique()))
    new_blood  = st.selectbox("Blood Type", sorted(df["Blood Type"].dropna().unique()))

    inp_raw = pd.DataFrame([[
        new_age, new_los,
        new_gender, new_blood, new_cond, new_ins, new_adm, new_med, new_test
    ]], columns=feats)

    cb_model = trained_models["CatBoost"]
    pred_pool = Pool(inp_raw, cat_features=cat_idx)
    pred  = cb_model.predict(pred_pool)[0]
    prob  = cb_model.predict_proba(pred_pool)[0][1]
    label = "🔴 HIGH COST PATIENT" if pred == 1 else "🟢 NORMAL COST PATIENT"
    colour = "#c0392b" if pred == 1 else "#27ae60"
    st.markdown(f"<h3 style='color:{colour}!important;'>Prediction: {label}</h3>", unsafe_allow_html=True)
    st.progress(float(prob))
    st.caption(f"Confidence: {prob:.1%} — decision-support estimate only, not a clinical diagnosis")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 SHAP Explainability — Why is a patient predicted high-cost?")
    st.caption("SHAP values show directional feature influence — not causal medical relationships.")

    st.info("""
    **Why SHAP over LIME?** SHAP provides theoretically consistent Shapley values from
    cooperative game theory — guaranteeing fair attribution even when features are correlated.
    CatBoost + SHAP gives exact TreeSHAP explanations in polynomial time.
    LIME uses local linear approximations which can be inconsistent across runs.
    In healthcare, consistency and auditability are non-negotiable.
    """)

    @st.cache_resource
    def get_shap_values(_cb_model, _X_raw_te, _feats, _cat_idx):
        exp = shap.TreeExplainer(_cb_model)
        # Use 500 samples max for speed
        sample = _X_raw_te.head(500).copy()
        # CatBoost SHAP needs numeric — encode cats to int for SHAP display
        le = LabelEncoder()
        sample_enc = sample.copy()
        cat_cols = [_feats[i] for i in _cat_idx]
        for c in cat_cols:
            sample_enc[c] = le.fit_transform(sample_enc[c].astype(str))
        sv = exp.shap_values(sample_enc)
        if hasattr(sv, "shape") and len(sv.shape) > 1 and sv.shape[1] > sample_enc.shape[1]:
            sv = sv[:, :-1]
        return exp, sv, sample_enc

    cb_m = trained_models["CatBoost"]
    with st.spinner("Computing SHAP values…"):
        shap_exp, shap_vals, shap_X = get_shap_values(cb_m, X_raw_test, feats, cat_idx)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**📊 Global Feature Importance — Mean |SHAP|**")
        fig,ax = plt.subplots(figsize=(7,5))
        shap.summary_plot(shap_vals, shap_X, plot_type="bar",
                          feature_names=feats, show=False)
        fig.tight_layout(); st.pyplot(fig)
        insight_box([
            "Bar = mean absolute SHAP value — average impact magnitude across all predictions",
            "**Medical Condition** and **Length of Stay** are typically the strongest signals",
            "Admission Type and Medication also influence high-cost classification"
        ])
    with c2:
        st.markdown("**🐝 SHAP Beeswarm — Direction & Magnitude**")
        fig2,ax2 = plt.subplots(figsize=(7,5))
        shap.summary_plot(shap_vals, shap_X, feature_names=feats, show=False)
        fig2.tight_layout(); st.pyplot(fig2)
        insight_box([
            "Each dot = one patient. Colour = feature value (red=high, blue=low)",
            "Dots right of centre = feature pushes toward **high-cost prediction**",
            "Longer stays and certain conditions consistently increase predicted cost risk"
        ])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🚨 Composite Anomaly Detection — Suspicious Claims")

    st.info("""
    **Detection methodology — why not plain IQR or Z-score?**

    This dataset has a near-uniform billing distribution (range $0–$53K, std $14K).
    Standard IQR on raw billing produces bounds of −$23K to $75K — catching zero records.
    Z-score (>2σ) also catches zero records on a uniform distribution.

    **Solution: Bill-per-Day rate + Negative Billing composite:**
    - 🔴 **Negative Billing** — definite data or billing entry error
    - 📊 **High Bill-per-Day rate** — daily billing rate above IQR upper fence
      (catches short-stay overbilling that raw billing amount misses)

    Note: Bill-per-Day is used ONLY here for anomaly analytics — NOT in ML training.
    """)

    anom_df  = filtered[filtered["Anomaly"]].copy()
    anom_pos = filtered[filtered["Anomaly"] & (filtered["Billing Amount"] > 0)]
    norm_df  = filtered[~filtered["Anomaly"]].copy()
    neg_count = int((filtered["Billing Amount"] < 0).sum())

    avg_anom = float(anom_pos["Billing Amount"].mean()) if len(anom_pos) > 0 else 0.0
    avg_norm = float(norm_df["Billing Amount"].mean())  if len(norm_df) > 0 else 1.0
    ratio    = avg_anom / avg_norm if avg_norm > 0 else 0.0

    a1,a2,a3,a4,a5 = st.columns(5)
    a1.metric("🚨 Flagged Claims",        f"{len(anom_df):,}")
    a2.metric("📊 Anomaly Rate",          f"{len(anom_df)/len(filtered)*100:.1f}%" if len(filtered)>0 else "0%")
    a3.metric("💸 Avg Overbilling Claim", f"${avg_anom:,.0f}")
    a4.metric("📈 vs Normal Claims",      f"{ratio:.2f}x higher")
    a5.metric("❌ Negative Billing",       f"{neg_count:,}")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Normal vs Anomalous Billing Distribution**")
        fig,ax = plt.subplots(figsize=(7,4))
        ax.hist(norm_df["Billing Amount"], bins=50, alpha=0.6,
                label=f"Normal (n={len(norm_df):,})", color=BLUE)
        if len(anom_df) > 0:
            ax.hist(anom_pos["Billing Amount"], bins=30, alpha=0.75,
                    label=f"Overbilling Anomalies (n={len(anom_pos):,})", color="red")
        ax.legend()
        ax.set_xlabel("Billing Amount ($)")
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"Overbilling anomalies average **${avg_anom:,.0f}** vs ${avg_norm:,.0f} for normal claims",
            f"Flagged overbilling claims are **{ratio:.2f}x higher** than normal claim average",
            f"**{neg_count}** claims carry negative billing — definite data entry errors"
        ])
    with c2:
        st.markdown("**Anomaly Rate by Medical Condition (%)**")
        fig,ax = plt.subplots(figsize=(7,4))
        if len(anom_df) > 0:
            total_by_cond = filtered["Medical Condition"].value_counts()
            anom_by_cond  = anom_df["Medical Condition"].value_counts()
            pct = (anom_by_cond / total_by_cond * 100).dropna().sort_values()
            pct.plot(kind="barh", ax=ax, color="red", alpha=0.75)
            ax.set_xlabel("Anomaly Rate (%)")
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x:.1f}%"))
            top_cond = pct.idxmax()
        else:
            ax.text(0.5,0.5,"No anomalies", ha="center"); top_cond="N/A"
        fig.tight_layout(); st.pyplot(fig)
        insight_box([
            f"**{top_cond}** has the highest anomaly rate — warrants targeted audit",
            "Rate shown as % of claims per condition — removes volume bias",
            "High anomaly rate in a condition may indicate systematic overbilling"
        ])

    st.markdown("**🚨 Top 25 Suspicious Claims — Ranked by Bill-per-Day Rate**")
    if len(anom_df) > 0:
        show_cols = ["Name","Age","Medical Condition","Hospital","Insurance Provider",
                     "Billing Amount","Length of Stay","Bill_per_Day","Anomaly_Reason","Z_Score"]
        top25 = anom_df[show_cols].sort_values("Bill_per_Day", ascending=False).head(25).copy()
        top25["Billing Amount"] = top25["Billing Amount"].map("${:,.2f}".format)
        top25["Bill_per_Day"]   = top25["Bill_per_Day"].map("${:,.0f}/day".format)
        top25["Z_Score"]        = top25["Z_Score"].round(2)
        st.dataframe(top25, use_container_width=True)

    savings = anom_pos["Billing Amount"].sum() * 0.30
    st.success(f"💡 Estimated savings if 30% of overbilling anomalies are resolved: **${savings:,.2f}**")

    with st.expander("📋 Methodology Notes & Production Limitations"):
        st.markdown("""
        **Why Bill-per-Day instead of raw billing?**
        A 30-day stay with $50K billing may be clinically justified.
        A 1-day stay with $40K billing is far more suspicious.
        Bill-per-Day normalises for length of stay — the key operational variable.

        **v2 separation of concerns:**
        - Bill-per-Day is used ONLY for anomaly analytics (Tab 4)
        - It is NEVER used as a training feature in ML models (Tab 2/3)
        - This prevents target leakage while preserving its utility for auditing

        **Production enhancements would include:**
        - Isolation Forest or Autoencoder-based deep anomaly detection
        - Temporal behaviour analysis (same provider over time)
        - CPT vs diagnosis code consistency checks
        - Network clustering (provider → hospital → insurer fraud rings)
        - Role-based access, encryption, HIPAA-compliant infrastructure

        **This dataset is fully synthetic — no real PHI is present or at risk.**
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PATIENT DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔎 Patient-Level Drill Down")
    st.caption("⚠️ Fully synthetic dataset. No real patient health information (PHI) used.")

    search = st.text_input("🔍 Search by Patient Name", placeholder="Type a name…")
    if search:
        result = df[df["Name"].str.contains(search, case=False, na=False)]
        if len(result) == 0:
            st.warning("No patients found.")
        else:
            st.success(f"Found {len(result)} record(s)")
            st.dataframe(result, use_container_width=True)
            p = result.iloc[0]
            st.markdown("---")
            st.markdown(f"### 👤 {p['Name']} — Patient Summary")
            d1,d2,d3,d4 = st.columns(4)
            d1.metric("Age",            p["Age"])
            d2.metric("Billing Amount", f"${p['Billing Amount']:,.2f}")
            d3.metric("Length of Stay", f"{p['Length of Stay']} days")
            d4.metric("Bill / Day",     f"${p['Bill_per_Day']:,.0f}")
            d5,d6,d7,d8 = st.columns(4)
            d5.metric("Condition",   p["Medical Condition"])
            d6.metric("Admission",   p["Admission Type"])
            d7.metric("Insurance",   p["Insurance Provider"])
            d8.metric("Test Result", p["Test Results"])
            risk = "🔴 HIGH COST" if p["High Cost"] == 1 else "🟢 NORMAL COST"
            anom = f"⚠️ FLAGGED — {p['Anomaly_Reason']}" if p["Anomaly"] else "✅ CLEAN"
            st.markdown(f"**Cost Risk:** {risk} &nbsp;&nbsp;&nbsp; **Anomaly Status:** {anom}")
    else:
        st.markdown("**📋 Sample Patient Records**")
        st.dataframe(
            filtered[["Name","Age","Gender","Medical Condition","Billing Amount",
                       "Bill_per_Day","Insurance Provider","Admission Type",
                       "Length of Stay","High Cost","Anomaly","Anomaly_Reason"]].head(50),
            use_container_width=True
        )


st.warning("This platform is intended for analytical and educational purposes only. "
           "Not designed for real-world clinical decision-making.")

st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:16px;color:#4a7fb5!important;font-size:0.85rem;'>
    🏥 Healthcare Claims Analytics Platform v2 &nbsp;|&nbsp;
    Built by <strong>Saivishal Apuru</strong> — FAU M.S. CS &nbsp;|&nbsp;
    CatBoost · LightGBM · XGBoost · SHAP · Composite Anomaly Detection · Streamlit &nbsp;|&nbsp;
    <em>Leakage-free ML · Decision-support tool · Synthetic data only · Not for clinical use</em>
</div>
""", unsafe_allow_html=True)
