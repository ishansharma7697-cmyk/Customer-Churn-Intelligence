import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------- Page setup ----------
st.set_page_config(page_title="Churn Intelligence Dashboard", page_icon="📉", layout="wide")
BASE = Path(__file__).parent

# Works on older and newer Streamlit versions
_NEW_API = tuple(int(p) for p in st.__version__.split(".")[:2]) >= (1, 50)


def show(fig):
    if _NEW_API:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .hero {padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
           background: linear-gradient(135deg, #4f46e5, #7c3aed);}
    .hero h1 {margin: 0; font-size: 1.9rem; color: #ffffff;}
    .hero p {margin: .3rem 0 0; color: #e9e7ff;}
    </style>
    <div class="hero">
        <h1>📉 Customer Churn Intelligence</h1>
        <p>Predict churn risk for a customer, explore the dataset, and see what drives the model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Load pipeline, metrics and data ----------
@st.cache_resource
def load_pipeline():
    return joblib.load(BASE / "churn_pipeline.pkl")


@st.cache_data
def load_metrics():
    path = BASE / "model_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_data():
    path = BASE / "Telco-Customer-Churn.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("churn_pipeline.pkl not found. Run the whole notebook first and keep the file "
             "in the same folder as app.py.")
    st.stop()

metrics = load_metrics()
data = load_data()


def predict_many(rows):
    """The pipeline does scaling + encoding itself, so we pass raw customer values."""
    return pipeline.predict_proba(pd.DataFrame(rows))[:, 1]


# ---------- Sidebar: customer profile ----------
with st.sidebar:
    st.header("🧾 Customer profile")

    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    line_opts = ["No phone service"] if phone == "No" else ["No", "Yes"]
    addon_opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
    multiple_lines = st.selectbox("Multiple lines", line_opts)
    online_security = st.selectbox("Online security", addon_opts)
    online_backup = st.selectbox("Online backup", addon_opts)
    device_protection = st.selectbox("Device protection", addon_opts)
    tech_support = st.selectbox("Tech support", addon_opts)
    streaming_tv = st.selectbox("Streaming TV", addon_opts)
    streaming_movies = st.selectbox("Streaming movies", addon_opts)

    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])

    st.subheader("Billing")
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, 0.5)
    manual_total = st.checkbox("Enter total charges manually")
    if manual_total:
        total = st.number_input("Total charges", min_value=0.0, value=float(monthly * tenure), step=10.0)
    else:
        total = float(round(monthly * tenure, 2))
        st.caption(f"Total charges (monthly × tenure): **{total:,.2f}**")

    st.subheader("Settings")
    threshold = st.slider("Churn threshold", 0.10, 0.90, 0.50, 0.05,
                          help="A customer is flagged High risk at or above this probability.")

profile = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

prob = float(predict_many([profile])[0])

if prob >= threshold:
    level, color = "High risk", "#ef4444"
elif prob >= threshold * 0.6:
    level, color = "Medium risk", "#f59e0b"
else:
    level, color = "Low risk", "#22c55e"

tab_pred, tab_dash, tab_model = st.tabs(["🔮 Prediction", "📊 Data Dashboard", "🧠 Model Insights"])

# =====================================================================
# TAB 1: PREDICTION
# =====================================================================
with tab_pred:
    left, right = st.columns([1.1, 1])

    with left:
        with st.container(border=True):
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Churn probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, threshold * 60], "color": "rgba(34,197,94,0.25)"},
                        {"range": [threshold * 60, threshold * 100], "color": "rgba(245,158,11,0.25)"},
                        {"range": [threshold * 100, 100], "color": "rgba(239,68,68,0.25)"},
                    ],
                    "threshold": {"line": {"color": "gray", "width": 4}, "value": threshold * 100},
                },
            ))
            gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
            show(gauge)

    with right:
        with st.container(border=True):
            st.markdown(f"### Risk level: <span style='color:{color}'>{level}</span>",
                        unsafe_allow_html=True)
            if prob >= threshold:
                st.error("This customer is likely to churn.")
            else:
                st.success("This customer is likely to stay.")

            if data is not None:
                avg = data["ChurnFlag"].mean()
                st.metric("Versus average customer", f"{prob * 100:.1f}%",
                          delta=f"{(prob - avg) * 100:+.1f} pts vs {avg * 100:.1f}% average",
                          delta_color="inverse")
            else:
                st.metric("Churn probability", f"{prob * 100:.1f}%")

        k1, k2, k3 = st.columns(3)
        k1.metric("Tenure", f"{tenure} mo")
        k2.metric("Monthly", f"{monthly:,.0f}")
        k3.metric("Contract", contract.replace("Month-to-month", "Monthly"))

    # ----- Suggested actions (simple rules, not model output) -----
    st.subheader("💡 Suggested retention actions")
    st.caption("Rule-based tips from common churn patterns in the Telco data. They are not model output.")
    tips = []
    if contract == "Month-to-month":
        tips.append("Offer a discount for moving to a 1- or 2-year contract.")
    if internet == "Fiber optic" and tech_support == "No":
        tips.append("Bundle free or discounted tech support for fiber customers.")
    if internet != "No" and online_security == "No":
        tips.append("Promote the online security add-on.")
    if payment == "Electronic check":
        tips.append("Encourage automatic payment (bank transfer or card).")
    if tenure < 12:
        tips.append("Run an early-tenure onboarding or check-in campaign.")
    if monthly > 80:
        tips.append("Review the plan price. High monthly bills raise churn risk.")
    if tips:
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.write("No specific risk factors flagged for this profile.")

    # ----- What-if analysis -----
    st.subheader("🔬 What-if analysis")
    c1, c2 = st.columns(2)

    with c1:
        contracts = ["Month-to-month", "One year", "Two year"]
        probs_c = predict_many([{**profile, "Contract": c} for c in contracts]) * 100
        fig_c = px.bar(x=contracts, y=probs_c, text=[f"{p:.1f}%" for p in probs_c],
                       labels={"x": "Contract", "y": "Churn probability (%)"},
                       title="If this customer switched contract")
        fig_c.update_traces(marker_color=["#ef4444", "#f59e0b", "#22c55e"])
        fig_c.update_layout(height=340, yaxis_range=[0, 100])
        show(fig_c)

    with c2:
        months = list(range(0, 73, 3))
        probs_t = predict_many([{**profile, "tenure": t, "TotalCharges": monthly * t}
                                for t in months]) * 100
        fig_t = px.line(x=months, y=probs_t, markers=True,
                        labels={"x": "Tenure (months)", "y": "Churn probability (%)"},
                        title="Churn risk across tenure")
        fig_t.add_vline(x=tenure, line_dash="dash", line_color="gray")
        fig_t.update_layout(height=340, yaxis_range=[0, 100])
        show(fig_t)

# =====================================================================
# TAB 2: DATA DASHBOARD
# =====================================================================
with tab_dash:
    if data is None:
        st.warning("Put Telco-Customer-Churn.csv in the same folder as app.py to enable this tab.")
    else:
        f1, f2 = st.columns(2)
        sel_contract = f1.multiselect("Contract", sorted(data["Contract"].unique()),
                                      default=sorted(data["Contract"].unique()))
        sel_internet = f2.multiselect("Internet service", sorted(data["InternetService"].unique()),
                                      default=sorted(data["InternetService"].unique()))
        d = data[data["Contract"].isin(sel_contract) & data["InternetService"].isin(sel_internet)]

        if d.empty:
            st.info("No customers match these filters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers", f"{len(d):,}")
            m2.metric("Churn rate", f"{d['ChurnFlag'].mean() * 100:.1f}%")
            m3.metric("Avg monthly charge", f"{d['MonthlyCharges'].mean():.1f}")
            m4.metric("Avg tenure", f"{d['tenure'].mean():.1f} mo")

            def churn_bar(col, title, horizontal=False):
                g = d.groupby(col)["ChurnFlag"].mean().mul(100).reset_index()
                g.columns = [col, "Churn rate (%)"]
                if horizontal:
                    fig = px.bar(g, y=col, x="Churn rate (%)", orientation="h", text_auto=".1f", title=title)
                else:
                    fig = px.bar(g, x=col, y="Churn rate (%)", text_auto=".1f", title=title)
                fig.update_layout(height=340)
                return fig

            r1a, r1b = st.columns(2)
            with r1a:
                show(churn_bar("Contract", "Churn rate by contract"))
            with r1b:
                show(churn_bar("InternetService", "Churn rate by internet service"))

            r2a, r2b = st.columns(2)
            with r2a:
                show(churn_bar("PaymentMethod", "Churn rate by payment method", True))
            with r2b:
                pie = px.pie(d, names="Churn", hole=0.55, title="Churned vs retained",
                             color="Churn", color_discrete_map={"No": "#22c55e", "Yes": "#ef4444"})
                pie.update_layout(height=340)
                show(pie)

            r3a, r3b = st.columns(2)
            with r3a:
                hist = px.histogram(d, x="tenure", color="Churn", nbins=36, barmode="overlay", opacity=0.7,
                                    title="Tenure distribution",
                                    color_discrete_map={"No": "#22c55e", "Yes": "#ef4444"})
                hist.update_layout(height=340)
                show(hist)
            with r3b:
                box = px.box(d, x="Churn", y="MonthlyCharges", color="Churn", title="Monthly charges",
                             color_discrete_map={"No": "#22c55e", "Yes": "#ef4444"})
                box.update_layout(height=340, showlegend=False)
                show(box)

# =====================================================================
# TAB 3: MODEL INSIGHTS
# =====================================================================
with tab_model:
    fitted = pipeline.named_steps["model"]
    names = [n.split("__", 1)[1] for n in pipeline.named_steps["preprocess"].get_feature_names_out()]

    st.markdown(f"**Model:** `{type(fitted).__name__}`  |  **Input features after encoding:** {len(names)}")

    if metrics:
        st.caption("Test-set results saved by the notebook (threshold 0.5)")
        a, b, c, d_, e = st.columns(5)
        a.metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
        b.metric("Precision", f"{metrics['test_precision']:.3f}")
        c.metric("Recall", f"{metrics['test_recall']:.3f}")
        d_.metric("F1", f"{metrics['test_f1']:.3f}")
        e.metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")

    if hasattr(fitted, "feature_importances_"):
        values, label = np.asarray(fitted.feature_importances_), "Importance"
    elif hasattr(fitted, "coef_"):
        values, label = np.asarray(fitted.coef_)[0], "Coefficient"
    else:
        values, label = None, ""

    if values is None:
        st.info("This model type does not expose feature importances.")
    else:
        imp = pd.Series(values, index=names)
        top = imp.reindex(imp.abs().sort_values(ascending=False).head(15).index).iloc[::-1]
        fig_i = px.bar(x=top.values, y=top.index, orientation="h",
                       labels={"x": label, "y": "Feature"}, title="Top 15 features")
        fig_i.update_layout(height=520)
        show(fig_i)
        st.caption("Importance shows what the model relies on, not that a feature causes churn.")
