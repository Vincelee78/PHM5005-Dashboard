"""PHM5005: NHANES elevated HbA1c — presentation dashboard.

Uses the real CSV/PNG outputs exported by the accompanying Jupyter notebook,
and its already-fitted 2017–March 2020 scikit-learn pipelines.
It NEVER retrains on the 2021–2023 temporal-validation participants.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
import streamlit as st


# ---------------------------------------------------------------------------
# Page and project paths
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PHM5005 | Group H | HbA1c Temporal Validation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "phm5005_outputs"
FIGURE_DIR = RESULTS_DIR / "figures"
MODEL_FILE = RESULTS_DIR / "hba1c_logistic_models.joblib"

NAVY = "#0B1525"
TEAL = "#43D8C5"
BLUE = "#74A9FF"
TEXT = "#EFF6FF"
MUTED = "#A9BDD4"
PANEL = "#14263D"
GRID = "#2D4660"
AGE_MODEL = "Age only"
FULL_MODEL = "Age + gender + BMI"


# ---------------------------------------------------------------------------
# Presentation-friendly styling, using standard Streamlit controls
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      /* App shell: the same navy palette as .streamlit/config.toml. */
      .stApp {background: #0B1525; color: #EFF6FF;}
      .block-container {padding-top: 1.35rem; padding-bottom: 2.4rem; max-width: 1350px;}
      /* The hero is the only gradient; other content uses restrained dark surfaces. */
      .hero {
        border-radius: 22px; padding: 29px 34px;
        background: linear-gradient(112deg, #183454 0%, #184B67 65%, #126D73 100%);
        color: #FFFFFF; box-shadow: 0 14px 36px rgba(0,0,0,.23);
        margin-bottom: 23px;
      }
      .eyebrow {font-size: .76rem; font-weight: 750; letter-spacing: .11em;
        color: #7CF3E1; margin-bottom: .52rem;}
      .hero-title {font-size: clamp(1.68rem, 3vw, 2.42rem); line-height: 1.16;
        font-weight: 760; letter-spacing: -.025em; margin-bottom: 10px;}
      .hero-sub {font-size: 1rem; color: #DCEAF5; line-height: 1.55;}
      .pill {display:inline-block; border:1px solid rgba(255,255,255,.38);
        border-radius: 99px; padding: 6px 13px; color:#DCFCF6;
        font-size:.78rem; margin-top:15px;}
      [data-testid="stMetric"] {background: #14263D; border:1px solid #2B445D;
        border-radius: 15px; padding: 16px 19px; box-shadow:0 3px 14px rgba(0,0,0,.12);}
      [data-testid="stMetricLabel"] {color:#B9CCE0; font-size:.9rem;}
      [data-testid="stMetricValue"] {font-size: 1.72rem; color:#FFFFFF;}
      .section-title {font-size:1.16rem; font-weight:750; margin:21px 0 10px; color:#F0F7FF;}
      .muted {color:#B8C9DA; font-size:.92rem; line-height:1.5;}
      .note {border-left:4px solid #43D8C5; background:#153C48;
        padding:14px 17px; border-radius:0 12px 12px 0; margin:13px 0 17px;
        color:#DCF7F4; font-size:.91rem; line-height:1.55;}
      .disclaimer {background:#15283D; border:1px solid #29445E;
        border-radius:12px; padding:13px 16px; color:#BACDDD; font-size:.86rem; line-height:1.5;}
      div[data-baseweb="tab-list"] {gap: 12px;}
      button[data-baseweb="tab"] {font-size:1rem; font-weight:650;}
      .stPlotlyChart {background:#14263D; border:1px solid #2B445D;
        border-radius:15px; padding:7px;}
      [data-testid="stImage"] img {border-radius: 13px; border: 1px solid #34516B;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">PHM5005 · GROUP H · PRECISION HEALTH & MEDICINE</div>
      <div class="hero-title">Elevated HbA1c · Temporal ML Validation</div>
      <div class="hero-sub">
        Can simple models trained on NHANES 2017–March 2020 provide reliable
        predictions in the later August 2021–August 2023 population?
      </div>
      <div class="pill">Survey-weighted evaluation &nbsp;·&nbsp; Educational demonstration</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load *exported* analysis files. Missing optional files do not crash the app.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_trusted_models(path: str) -> dict:
    # Only load a joblib file you saved yourself from the PHM5005 notebook.
    # Joblib/pickle files may execute code when opened; do not use unknown files.
    return joblib.load(path)


def optional_table(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.is_file():
        return pd.DataFrame()
    try:
        return read_csv(str(path))
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


def pretty_metric(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}" if pd.notna(value) else "—"
    except (TypeError, ValueError):
        return "—"


def get_model_row(table: pd.DataFrame, cycle: str, model: str):
    if table.empty or not {"cycle", "model"}.issubset(table.columns):
        return None
    rows = table.loc[(table["cycle"] == cycle) & (table["model"] == model)]
    return None if rows.empty else rows.iloc[0]


def plot_bars(table: pd.DataFrame, metric: str, label: str, y_title: str):
    """Plot only actual metrics present in the exported CSV."""
    fig = go.Figure()
    for cycle, color in [("Development", BLUE), ("Temporal validation", TEAL)]:
        frame = table.loc[table["cycle"] == cycle]
        if frame.empty:
            continue
        frame = frame.set_index("model").reindex([AGE_MODEL, FULL_MODEL])
        values = pd.to_numeric(frame[metric], errors="coerce")
        fig.add_bar(
            name=("Development (apparent)" if cycle == "Development" else "2021–2023 validation"),
            x=["Age only", "Age + gender + BMI"],
            y=values.tolist(),
            text=[f"{v:.4f}" if pd.notna(v) else "" for v in values],
            textposition="outside", marker_color=color,
            hovertemplate="%{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(
        title=dict(text=label, font=dict(size=17, color=TEXT)),
        barmode="group", height=355, margin=dict(l=20, r=15, t=65, b=20),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="Arial", color=TEXT),
        legend=dict(orientation="h", y=1.19, x=0, font=dict(color=TEXT)),
        yaxis=dict(title=y_title, rangemode="tozero", gridcolor=GRID),
        xaxis=dict(title=""),
    )
    if metric == "auc":
        fig.update_yaxes(range=[0, 1.02])
    return fig


def chart_style(fig: go.Figure, title: str, height: int = 350):
    """Apply a consistent dark look to validation figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=16)),
        height=height, margin=dict(l=20, r=18, t=65, b=35),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        legend=dict(orientation="h", y=1.17, x=0),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


def make_validation_roc(predictions: pd.DataFrame):
    """Recreate survey-weighted ROC curves from actual saved model predictions."""
    needed = {"y", "survey_weight", "pred_age_only", "pred_age_gender_bmi"}
    if not needed.issubset(predictions.columns):
        return None
    fig = go.Figure()
    for label, column, color in [
        (AGE_MODEL, "pred_age_only", BLUE),
        (FULL_MODEL, "pred_age_gender_bmi", TEAL),
    ]:
        frame = predictions[["y", "survey_weight", column]].apply(pd.to_numeric, errors="coerce").dropna()
        frame = frame.loc[frame["survey_weight"] > 0]
        if frame["y"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(frame["y"], frame[column], sample_weight=frame["survey_weight"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=color, width=3), name=label))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color=MUTED, dash="dash", width=1.5), name="Chance"))
    chart_style(fig, "Survey-weighted ROC curves · 2021–2023")
    fig.update_xaxes(title_text="False positive rate", range=[0,1])
    fig.update_yaxes(title_text="True positive rate", range=[0,1])
    return fig


def weighted_calibration_deciles(probabilities, outcomes, weights):
    """Aggregate actual predictions into weighted quantile bands, as in the notebook."""
    arr = pd.DataFrame({"p": probabilities, "y": outcomes, "w": weights}).apply(pd.to_numeric, errors="coerce").dropna()
    arr = arr.loc[arr["w"] > 0].sort_values("p")
    if arr.empty or arr["p"].nunique() < 3:
        return []
    cum = arr["w"].cumsum().to_numpy() / arr["w"].sum()
    vals = arr["p"].to_numpy()
    edges = np.unique(np.interp(np.linspace(0,1,11), np.r_[0,cum], np.r_[vals[0], vals]))
    groups = np.digitize(vals, edges[1:-1], right=True)
    points=[]
    for group in np.unique(groups):
        mask = groups==group
        w = arr["w"].to_numpy()[mask]
        points.append((np.average(vals[mask],weights=w), np.average(arr["y"].to_numpy()[mask],weights=w)))
    return points


def make_validation_calibration(predictions: pd.DataFrame):
    needed = {"y", "survey_weight", "pred_age_only", "pred_age_gender_bmi"}
    if not needed.issubset(predictions.columns):
        return None
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect calibration", line=dict(color=MUTED, dash="dash",width=1.5)))
    for label, column, color in [
        (AGE_MODEL,"pred_age_only",BLUE),
        (FULL_MODEL,"pred_age_gender_bmi",TEAL),
    ]:
        points=weighted_calibration_deciles(predictions[column],predictions["y"],predictions["survey_weight"])
        if points:
            fig.add_trace(go.Scatter(x=[x for x,_ in points],y=[y for _,y in points], mode="lines+markers", name=label, line=dict(color=color,width=3), marker=dict(size=7)))
    chart_style(fig,"Survey-weighted calibration · 2021–2023")
    fig.update_xaxes(title_text="Mean predicted probability",range=[0,0.65],tickformat=".0%")
    fig.update_yaxes(title_text="Observed proportion",range=[0,0.65],tickformat=".0%")
    return fig


def show_validation_figure(filename: str, heading: str, fallback_builder, predictions: pd.DataFrame):
    """Prefer notebook PNGs; recreate interactive curves if PNGs aren't present."""
    st.markdown(f"**{heading}**")
    path = FIGURE_DIR / filename
    if path.is_file():
        st.image(str(path), use_container_width=True)
    elif not predictions.empty:
        fig = fallback_builder(predictions)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("The prediction CSV lacks columns required for this chart.")
    else:
        st.info("Copy `predictions_temporal_validation.csv` or the corresponding PNG into `phm5005_outputs`.")


metrics = optional_table("primary_weighted_metrics.csv")
validation_predictions = optional_table("predictions_temporal_validation.csv")
manifest_path = RESULTS_DIR / "analysis_manifest.json"
manifest = read_json(str(manifest_path)) if manifest_path.is_file() else {}

# Split the application into a results view and an individual-prediction demo.
validation_tab, prediction_tab = st.tabs(
    ["📊  Temporal validation & results", "🧪  Individual prediction demo"]
)


# ---------------------------------------------------------------------------
# TAB 1: Display actual, previously calculated NHANES evaluation results.
# ---------------------------------------------------------------------------
with validation_tab:
    st.markdown('<div class="section-title">Validation at a glance</div>', unsafe_allow_html=True)

    validation_full = get_model_row(metrics, "Temporal validation", FULL_MODEL)
    validation_age = get_model_row(metrics, "Temporal validation", AGE_MODEL)
    cohort_n = (int(validation_full["n"]) if validation_full is not None
                else manifest.get("validation_n"))

    cols = st.columns(4)
    cols[0].metric("Validation participants", f"{cohort_n:,}" if cohort_n is not None else "—")
    cols[1].metric("Expanded model · AUC",
                   pretty_metric(validation_full["auc"]) if validation_full is not None else "—")
    cols[2].metric("Expanded model · Brier",
                   pretty_metric(validation_full["brier"]) if validation_full is not None else "—")
    cols[3].metric("Expanded model · calibration slope",
                   pretty_metric(validation_full["calibration_slope"]) if validation_full is not None else "—")

    st.markdown(
        '<div class="note"><b>Main finding:</b> The 2021–2023 dataset was used '
        'to evaluate the frozen models, <b>not to retrain them</b>. '
        'AUC measures ranking, while calibration measures whether probabilities '
        'match observed frequencies.</div>',
        unsafe_allow_html=True,
    )

    if metrics.empty:
        st.warning("Missing `phm5005_outputs/primary_weighted_metrics.csv`. "
                   "Copy this output from your notebook to display real model performance.")
    elif not {"cycle", "model", "auc", "brier"}.issubset(metrics.columns):
        st.error("The primary metrics file does not match the expected notebook columns.")
    else:
        st.markdown('<div class="section-title">How well did the models generalise?</div>',
                    unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        with left:
            st.plotly_chart(plot_bars(metrics, "auc", "Discrimination · ROC-AUC", "AUC (higher is better)"),
                            use_container_width=True, config={"displayModeBar": False})
        with right:
            st.plotly_chart(plot_bars(metrics, "brier", "Probability error · Brier score", "Brier (lower is better)"),
                            use_container_width=True, config={"displayModeBar": False})

        if validation_full is not None and validation_age is not None:
            auc_gain = float(validation_full["auc"] - validation_age["auc"])
            st.caption(f"In temporal validation, adding gender and BMI changed AUC by "
                       f"{auc_gain:+.4f} relative to the age-only baseline. "
                       "Development estimates are apparent (in-sample) and may be optimistic.")

        st.markdown('<div class="section-title">Actual ROC & calibration curves</div>',
                    unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        with left:
            show_validation_figure("roc_temporal_validation.png", "Survey-weighted ROC · 2021–2023", make_validation_roc, validation_predictions)
        with right:
            show_validation_figure("calibration_temporal_validation.png", "Survey-weighted calibration · 2021–2023", make_validation_calibration, validation_predictions)

        with st.expander("See the exact weighted evaluation table"):
            columns = [c for c in ["cycle", "model", "n", "auc", "brier",
                                   "calibration_intercept", "calibration_slope",
                                   "mean_prediction_minus_observed"] if c in metrics.columns]
            st.dataframe(metrics[columns].round(4), hide_index=True, use_container_width=True)

    if not validation_predictions.empty and {"y", "survey_weight",
            "pred_age_only", "pred_age_gender_bmi"}.issubset(validation_predictions.columns):
        st.markdown('<div class="section-title">Explore predictions on real NHANES participants</div>',
                    unsafe_allow_html=True)
        choice = st.selectbox("Prediction distribution", [AGE_MODEL, FULL_MODEL], index=1)
        pred_col = "pred_age_only" if choice == AGE_MODEL else "pred_age_gender_bmi"
        probabilities = pd.to_numeric(validation_predictions[pred_col], errors="coerce").to_numpy()
        weights = pd.to_numeric(validation_predictions["survey_weight"], errors="coerce").to_numpy()
        mask = np.isfinite(probabilities) & np.isfinite(weights) & (weights > 0)
        if mask.any():
            # Each bar is a survey-weighted fraction, not a number of people.
            bin_edges = np.linspace(0, 1, 21)
            weighted_counts, _ = np.histogram(probabilities[mask], bins=bin_edges, weights=weights[mask])
            weighted_proportions = weighted_counts / weighted_counts.sum()
            hist = go.Figure(go.Bar(
                x=(bin_edges[:-1] + bin_edges[1:]) / 2,
                y=weighted_proportions * 100,
                marker_color=TEAL, width=.042,
                hovertemplate="Predicted probability: %{x:.1%}<br>Survey-weighted share: %{y:.1f}%<extra></extra>",
            ))
            hist.update_layout(height=310, margin=dict(l=20, r=15, t=10, b=45),
                               paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
                               xaxis=dict(title="Predicted probability", tickformat=".0%", range=[0, 1]),
                               yaxis=dict(title="Weighted share of participants (%)", gridcolor=GRID))
            st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})
            st.caption("This distribution is calculated from the actual stored 2021–2023 predictions. "
                       "These are model estimates, not confirmed diagnoses.")

    st.markdown('<div class="section-title">Robustness & comparisons</div>',
                unsafe_allow_html=True)
    with st.expander("HbA1c ≥5.8% threshold sensitivity analysis", expanded=False):
        tbl = optional_table("sensitivity_hba1c_5_8_weighted_metrics.csv")
        if not tbl.empty:
            st.dataframe(tbl.round(4), hide_index=True, use_container_width=True)
        else:
            st.info("Add `sensitivity_hba1c_5_8_weighted_metrics.csv` to show this analysis.")
    with st.expander("Complete-case BMI sensitivity analysis", expanded=False):
        tbl = optional_table("sensitivity_complete_case_bmi.csv")
        if not tbl.empty:
            st.dataframe(tbl.round(4), hide_index=True, use_container_width=True)
        else:
            st.info("Add `sensitivity_complete_case_bmi.csv` to show this analysis.")
    with st.expander("Exploratory Random Forest comparison", expanded=False):
        tbl = optional_table("optional_random_forest_results.csv")
        if not tbl.empty:
            st.dataframe(tbl.round(4), hide_index=True, use_container_width=True)
            st.caption("This is an untuned comparator and used as comparison to the Logistic Regression model.")
        else:
            st.info("No Random Forest comparison supplied. This is optional.")

    
    # ============================================================
    # CONCLUSION AND REFLECTION — PHM5005 PROJECT
    # Display the main scientific findings and lessons learned.
    # ============================================================

    st.markdown(
        '<div class="section-title">Conclusions & Key Findings</div>',
        unsafe_allow_html=True,
    )

    # Create two columns to display four conclusion cards.
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # Conclusion 1: Temporal generalisability.
        with st.container(border=True):
            st.markdown("#### 01 · Temporal Generalisability")
            st.metric("Expanded LR · Validation AUC", "0.7231")
            st.write(
                "The expanded Logistic Regression model retained moderate "
                "discrimination in the 2021–2023 NHANES cohort. "
                "Its apparent development AUC was 0.7379, compared with "
                "0.7231 in temporal validation. Development performance "
                "was measured in-sample and may be optimistic."
            )

        # Conclusion 3: Calibration.
        with st.container(border=True):
            st.markdown("#### 03 · Imperfect Calibration")
            st.metric("Validation calibration slope", "0.9044")
            st.write(
                "Although discrimination remained moderate, predicted "
                "probabilities were not perfectly calibrated. The "
                "calibration slope was below 1, and average predicted "
                "risk exceeded observed prevalence by approximately "
                "1.1 percentage points."
            )

    with col2:
        # Conclusion 2: Value of additional predictors.
        with st.container(border=True):
            st.markdown("#### 02 · Modest Predictive Improvement")
            st.metric("AUC gain over age-only model", "+0.0273")
            st.write(
                "Adding gender and BMI improved discrimination and "
                "the Brier score compared with using age alone. "
                "However, the improvement was modest, indicating "
                "limited additional predictive benefit."
            )

        # Conclusion 4: Model complexity.
        with st.container(border=True):
            st.markdown("#### 04 · Simpler Models Can Generalise Better")
            st.metric("Random Forest · Validation AUC", "0.7056")
            st.write(
                "The exploratory Random Forest achieved a higher "
                "apparent development AUC but a lower validation AUC "
                "than expanded Logistic Regression (0.7056 vs 0.7231). "
                "The untuned Random Forest did not demonstrate a "
                "temporal performance advantage."
            )

    # Display the overall scientific conclusion.
    st.markdown("### Overall Conclusion")

    st.success(
        "Simple, interpretable Logistic Regression models retained "
        "moderate discrimination across NHANES cycles. However, "
        "imperfect calibration highlights the importance of "
        "evaluating both discrimination and probability reliability "
        "before considering clinical screening applications."
    )

    # ============================================================
    # PROJECT REFLECTION
    # ============================================================

    # st.markdown(
    #     '<div class="section-title">Reflection & Future Directions</div>',
    #     unsafe_allow_html=True,
    # )

    # # Use an expander to keep the main dashboard uncluttered.
    # with st.expander("Key lessons learned", expanded=True):

    #     st.markdown(
    #         """
    #         **1. Model validation matters as much as training**

    #         A model can perform well on its development dataset but
    #         perform differently on newer participants. Temporal
    #         validation provides evidence about generalisability.

    #         **2. Discrimination and calibration are different**

    #         ROC-AUC measures how well a model ranks participants,
    #         whereas calibration assesses whether predicted
    #         probabilities correspond to observed outcomes.

    #         **3. Data preparation and study design are essential**

    #         Eligibility screening, appropriate NHANES survey
    #         weighting, missing-data handling and separation of
    #         development and validation data are important for
    #         credible machine-learning research.
    #         """
    #     )

    # # Display possible improvements to the research.
    # with st.expander("Future improvements", expanded=False):

    #     st.markdown(
    #         """
    #         - Explore additional clinically available non-laboratory
    #           predictors beyond age, gender and BMI.
    #         - Incorporate internal cross-validation and appropriate
    #           model tuning using the development cohort.
    #         - Investigate recalibration using separate data without
    #           compromising independent model evaluation.
    #         - Validate the models in other populations and assess
    #           their potential clinical usefulness.
    #         """
    #     )

    st.markdown(
        '<div class="disclaimer">Interpretation: NHANES is a cross-sectional population survey. '
        'The outcome is <b>measured HbA1c ≥5.7%</b>, not future diabetes or a confirmed diagnosis. '
        'Temporal performance changes cannot be attributed to a single cause. '
        'Survey-bootstrap intervals from the notebook are approximate.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# TAB 2: Predict using the original fitted, frozen logistic pipelines.
# ---------------------------------------------------------------------------
with prediction_tab:
    st.markdown('<div class="section-title">Try the models with example participant details</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Change the three inputs to see predictions from the '
        'two <b>previously trained</b> Logistic Regression models. '
        'No new model fitting occurs here.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    inputs_col, output_col = st.columns([0.87, 1.13], gap="large")
    with inputs_col:
        st.markdown("#### Participant inputs")
        age = st.slider("Age (years)", min_value=20, max_value=80, value=55, step=1)
        recorded_gender = st.radio("Recorded gender", ["Male", "Female"], horizontal=True)
        bmi = st.slider("Body mass index · BMI (kg/m²)", min_value=15.0,
                        max_value=50.0, value=29.0, step=0.5)
        st.caption("Assumes an adult without previously reported diabetes or prediabetes, "
                   "and not recorded as pregnant. NHANES age is top-coded at 80.")

    with output_col:
        st.markdown("#### Model predictions")
        if not MODEL_FILE.is_file():
            st.warning("The trained-model file is missing. Export `primary_models` "
                       "from Section 9 of your notebook as "
                       "`phm5005_outputs/hba1c_logistic_models.joblib`.")
            st.info("Your results tab can work without the model file, but live predictions require it.")
        else:
            try:
                models = load_trusted_models(str(MODEL_FILE))
                # Preserve the exact feature names and order used during fitting.
                participant = pd.DataFrame({"age": [age],
                                            "female": [1 if recorded_gender == "Female" else 0],
                                            "bmi": [float(bmi)]})
                results = {}
                for model_name in [AGE_MODEL, FULL_MODEL]:
                    details = models[model_name]
                    required_features = details["features"]
                    model = details["estimator"]
                    results[model_name] = float(model.predict_proba(
                        participant[required_features])[0, 1])

                first, second = st.columns(2)
                first.metric("Age-only Logistic Regression", f"{results[AGE_MODEL]:.1%}")
                second.metric("Expanded Logistic Regression", f"{results[FULL_MODEL]:.1%}")

                demo_fig = go.Figure()
                demo_fig.add_bar(x=[100 * results[AGE_MODEL]], y=["Age only"],
                                 orientation="h", marker_color=BLUE,
                                 text=[f"{results[AGE_MODEL]:.1%}"], textposition="outside",
                                 hovertemplate="%{x:.1f}%<extra>Age only</extra>")
                demo_fig.add_bar(x=[100 * results[FULL_MODEL]], y=["Age + gender + BMI"],
                                 orientation="h", marker_color=TEAL,
                                 text=[f"{results[FULL_MODEL]:.1%}"], textposition="outside",
                                 hovertemplate="%{x:.1f}%<extra>Expanded LR</extra>")
                demo_fig.update_layout(
                    title="Estimated probability of elevated HbA1c",
                    showlegend=False, height=260, margin=dict(l=10, r=38, t=58, b=25),
                    paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
                    xaxis=dict(title="Estimated probability (%)", range=[0, 105], gridcolor=GRID),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(demo_fig, use_container_width=True,
                                config={"displayModeBar": False})
                delta_pp = (results[FULL_MODEL] - results[AGE_MODEL]) * 100
                st.caption(f"Difference between expanded and age-only models: {delta_pp:+.1f} "
                           "percentage points. These are the *model's* estimates, not observed HbA1c values.")
            except Exception as exc:
                st.error("The saved pipelines could not be loaded or used. Confirm that "
                         "you exported `primary_models` and installed a compatible "
                         "scikit-learn version.")
                with st.expander("Technical error details"):
                    st.code(str(exc))

    
    st.markdown(
        '<div class="disclaimer"><b>Educational demonstration only.</b> '
        'The models estimate whether measured HbA1c is at least 5.7% within the '
        'NHANES study population. They do not diagnose diabetes, predict future '
        'disease, or provide an individual medical recommendation. '
        'Their temporal calibration was imperfect.</div>',
        unsafe_allow_html=True,
    )

st.caption("PHM5005 group project · CDC NHANES public-use data · Models trained on 2017–March 2020 · "
           "Validated on August 2021–August 2023")