# app.py
# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL AI FAILURE OBSERVATORY — Main Streamlit Application
# This is the ONLY file the user interacts with.
# It handles: file upload, demo loading, running all engines,
#             displaying results, and offering PDF download.
# Run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st          # Streamlit: turns Python into a web app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for the learning curve chart
import matplotlib                # for color maps
import io
import os

# Import our custom modules
from core.orchestrator import Orchestrator
from core.trust_score import TrustScoreEngine
from core.pdf_reporter import PDFReporter

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the FIRST Streamlit call in any app
# Sets the browser tab title, icon, and layout width
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical AI Failure Observatory",
    page_icon="🔬",
    layout="wide",              # use the full browser width
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — inject raw CSS to style the app beyond Streamlit's defaults
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }

    /* Score card — the big trust score display */
    .score-card {
        background: linear-gradient(135deg, #1a2240 0%, #0d47a1 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .score-number {
        font-size: 72px;
        font-weight: 900;
        line-height: 1;
    }
    .score-label {
        font-size: 16px;
        opacity: 0.8;
        margin-top: 8px;
    }

    /* Severity badges */
    .badge-high   { background:#ffcccc; color:#c62828; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:bold; }
    .badge-medium { background:#fff3cc; color:#f57f17; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:bold; }
    .badge-low    { background:#e8f5e9; color:#2e7d32; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:bold; }
    .badge-none   { background:#f0f0f0; color:#555555; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:bold; }

    /* Section divider */
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 30px 0;
    }

    /* Engine card */
    .engine-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)
# unsafe_allow_html=True is needed to inject raw HTML/CSS into Streamlit


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def severity_badge(severity: str) -> str:
    """
    Return an HTML badge for a given severity string.
    Used to display colored severity labels in tables and text.
    """
    css_class = f"badge-{severity.lower()}" if severity.lower() in \
                ["high", "medium", "low", "none"] else "badge-none"
    return f'<span class="{css_class}">{severity}</span>'


def load_demo_data() -> pd.DataFrame:
    """
    Load the demo dataset from the data/ folder.
    If the file doesn't exist, generate a synthetic demo dataset.
    This ensures the app always works even without the real CSV.
    """
    demo_path = "data/demo_sample.csv"

    if os.path.exists(demo_path):
        # Load real demo file
        return pd.read_csv(demo_path)
    else:
        # Generate synthetic data that mimics credit risk data structure
        # This is our fallback so the app never crashes on demo
        np.random.seed(42)
        n = 2000

        df = pd.DataFrame({
            # Numeric features — realistic credit risk variables
            'AMT_INCOME_TOTAL':    np.random.lognormal(11, 0.5, n),
            'AMT_CREDIT':          np.random.lognormal(12, 0.6, n),
            'AMT_ANNUITY':         np.random.lognormal(9, 0.4, n),
            'DAYS_BIRTH':          np.random.randint(-25000, -6000, n),
            'DAYS_EMPLOYED':       np.random.randint(-10000, 0, n),
            'CNT_CHILDREN':        np.random.choice([0,1,2,3], n, p=[0.5,0.3,0.15,0.05]),
            'EXT_SOURCE_1':        np.random.beta(2, 5, n),
            'EXT_SOURCE_2':        np.random.beta(3, 4, n),
            'EXT_SOURCE_3':        np.random.beta(2, 3, n),

            # Categorical features
            'NAME_CONTRACT_TYPE':  np.random.choice(['Cash loans', 'Revolving loans'], n),
            'CODE_GENDER':         np.random.choice(['M', 'F'], n, p=[0.4, 0.6]),
            'NAME_EDUCATION_TYPE': np.random.choice(
                ['Secondary', 'Higher education', 'Incomplete higher'], n
            ),

            # Missing values — simulate real-world messiness
            # np.nan inserted randomly by masking ~15% of rows
            'OCCUPATION_TYPE': np.where(
                np.random.random(n) < 0.15,   # 15% will be NaN
                np.nan,
                np.random.choice(['Laborers', 'Core staff', 'Managers', 'Drivers'], n)
            ),

            # Target — highly imbalanced (only 8% defaulters, realistic for credit)
            'TARGET': np.random.choice([0, 1], n, p=[0.92, 0.08])
        })

        # Inject some outliers so the noise auditor has something to find
        outlier_indices = np.random.choice(n, size=30, replace=False)
        df.loc[outlier_indices, 'AMT_INCOME_TOTAL'] = 99999999   # impossible income

        # Inject some duplicate rows
        dupe_rows = df.sample(20, random_state=42)
        df = pd.concat([df, dupe_rows], ignore_index=True)

        return df


def plot_learning_curve(curve_data: dict, diagnosis: str) -> plt.Figure:
    """
    Draw the learning curve chart from the curve autopsy results.
    Returns a matplotlib Figure object (Streamlit can display this directly).
    """
    # Extract the curve data points
    train_sizes = curve_data["train_sizes"]
    train_mean  = curve_data["train_scores_mean"]
    train_std   = curve_data["train_scores_std"]
    val_mean    = curve_data["val_scores_mean"]
    val_std     = curve_data["val_scores_std"]

    # Diagnosis → color for the chart title
    diag_colors = {
        "HEALTHY": "green", "DATA-STARVED": "red",
        "OVER-COMPLEX": "orange", "LEAKY": "purple"
    }
    title_color = diag_colors.get(diagnosis, "black")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#f8f9fa')   # match app background
    ax.set_facecolor('#ffffff')

    # Plot training score line
    ax.plot(train_sizes, train_mean,
            color='#0066cc', linewidth=2.5, marker='o', markersize=5,
            label='Training AUC')

    # Shaded area = ±1 standard deviation around the training curve
    # This shows stability — wider band = less stable
    ax.fill_between(train_sizes,
                    np.array(train_mean) - np.array(train_std),
                    np.array(train_mean) + np.array(train_std),
                    alpha=0.15, color='#0066cc')

    # Plot validation score line
    ax.plot(train_sizes, val_mean,
            color='#e65100', linewidth=2.5, marker='s', markersize=5,
            label='Validation AUC (CV=5)')

    ax.fill_between(train_sizes,
                    np.array(val_mean) - np.array(val_std),
                    np.array(val_mean) + np.array(val_std),
                    alpha=0.15, color='#e65100')

    # Labels and formatting
    ax.set_xlabel("Training Set Size (samples)", fontsize=11)
    ax.set_ylabel("AUC Score", fontsize=11)
    ax.set_title(f"Learning Curve — Diagnosis: {diagnosis}",
                 fontsize=13, fontweight='bold', color=title_color)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.05)          # AUC always between 0.5 and 1.0
    ax.grid(True, alpha=0.3)        # light grid for readability
    ax.spines['top'].set_visible(False)    # remove top border
    ax.spines['right'].set_visible(False)  # remove right border

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Upload and configuration controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/microscope.png", width=60)
    st.title("🔬 Observatory")
    st.caption("Clinical AI Failure Diagnostic Tool")
    st.markdown("---")

    # Data source selection
    st.subheader("1. Load Data")
    data_source = st.radio(
        "Choose data source:",
        [" Upload CSV", " Use Demo Dataset"],
        index=1 
    )

    df = None   

    if data_source == " Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"],
            help="CSV must have a clear target column (0/1 or binary)"
        )
        if uploaded_file:
           
            df = pd.read_csv(uploaded_file)
            st.success(f" Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    else:
        
        df = load_demo_data()
        st.success(f" Demo loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.info("Using synthetic Home Credit-style data with injected corruption patterns.")

    st.markdown("---")

    
    st.subheader("2. Configure")

    if df is not None:
        
        default_target = "TARGET" if "TARGET" in df.columns else df.columns[-1]
        target_col = st.selectbox(
            "Select target column:",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_target)
        )

        
        id_col_options = ["None"] + df.columns.tolist()
        id_col_selection = st.selectbox(
            "ID column (optional — for group overlap check):",
            options=id_col_options,
            index=0
        )
        id_col = None if id_col_selection == "None" else id_col_selection

    st.markdown("---")
    st.subheader("3. Run")

   
    run_button = st.button(
        " Run Full Diagnostic",
        use_container_width=True,    
        type="primary"               
    )

    st.markdown("---")
    st.caption("Built by enghamza-AI · Stage 1 Flagship")
    st.caption("Clinical AI Failure Observatory v1.0")



st.title("🔬 Clinical AI Failure Observatory")
st.markdown(
    "Upload any tabular medical dataset. Get a full diagnostic: "
    "**corruption audit · bias-variance analysis · leakage scan · "
    "learning curve autopsy** — and one Model Trust Score."
)
st.markdown("---")


if df is not None and not run_button:
    st.subheader(" Data Preview")

    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        st.metric("Numeric Columns", numeric_cols)

    
    st.dataframe(df.head(10), use_container_width=True)
    st.info(" Configure settings in the sidebar, then click **Run Full Diagnostic**")



if run_button and df is not None:

    st.markdown("---")

    
    progress_bar = st.progress(0)
    status_text = st.empty()   

    def update_progress(pct: int, message: str):
        """Callback passed to the orchestrator to update the UI progress bar."""
        progress_bar.progress(pct)
        status_text.text(f"⚙️ {message}...")

   
    status_text.text(" Starting full diagnostic run...")

    try:
       
        orch = Orchestrator(df=df, target_col=target_col, id_col=id_col)

       
        full_report = orch.run_all(progress_callback=update_progress)

        
        trust_engine = TrustScoreEngine(full_report)
        trust_result = trust_engine.compute()

       
        progress_bar.empty()
        status_text.empty()

        st.success(" Diagnostic complete!")

    except Exception as e:
        
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Diagnostic failed: {str(e)}")
        st.exception(e)    
        st.stop()          


  

   
    score  = trust_result["score"]
    grade  = trust_result["grade"]
    verdict = trust_result["verdict"]

    
    grade_colors_hex = {
        "A": "#2e7d32", "B": "#558b2f",
        "C": "#f57f17", "D": "#e65100", "F": "#c62828"
    }
    score_color = grade_colors_hex.get(grade, "#333333")

    
    st.markdown(f"""
    <div class="score-card">
        <div class="score-number" style="color:{score_color}">{score}</div>
        <div class="score-label">Model Trust Score / 100</div>
        <div style="font-size:32px; font-weight:bold; margin-top:10px; color:{score_color}">
            Grade: {grade}
        </div>
        <div style="font-size:14px; margin-top:16px; max-width:600px;
                    margin-left:auto; margin-right:auto; line-height:1.6;">
            {verdict}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    
    st.subheader(" Score Breakdown")
    st.caption("How points were deducted across the 4 diagnostic engines")

    breakdown = trust_result.get("breakdown", {})

    
    categories = {
        " Noise Audit":           ("noise_audit",    25),
        " Bias-Variance":         ("bias_variance",  25),
        " Leakage Scanner":      ("leakage_scan",   30),
        " Curve Autopsy":         ("curve_autopsy",  20),
    }

    cols = st.columns(4)
    for i, (display, (key, max_pts)) in enumerate(categories.items()):
        cat_data = breakdown.get(key, {})
        penalty = cat_data.get("final_penalty", 0)
        with cols[i]:
            st.metric(
                label=display,
                value=f"−{penalty} pts",
                delta=f"max −{max_pts}",
                delta_color="inverse"   
            )

    st.markdown("---")

    
    st.subheader(" Engine Diagnostics")

    tab1, tab2, tab3, tab4 = st.tabs([
        " Noise Audit",
        " Bias-Variance",
        " Leakage Scan",
        " Curve Autopsy"
    ])

   
    with tab1:
        st.markdown("### Data Quality Findings")
        st.caption("6 corruption archetypes checked across the dataset")

        noise = full_report.get("noise_audit", {})

        checks_display = {
            "Missing Values":      ("missing_values",   "n_affected",       "columns affected"),
            "Outliers":            ("outliers",          "n_affected",       "columns with outliers"),
            "Duplicate Rows":      ("duplicates",        "n_duplicate_rows", "exact duplicate rows"),
            "Class Imbalance":     ("class_imbalance",   "imbalance_ratio",  "× imbalance ratio"),
            "Low-Variance Cols":   ("low_variance",      "n_affected",       "near-constant columns"),
            "Dtype Mismatches":    ("dtype_mismatches",  "n_affected",       "wrong-type columns"),
        }

        
        col_pairs = list(checks_display.items())
        for i in range(0, len(col_pairs), 2):
            c1, c2 = st.columns(2)
            for col_widget, (display_name, (key, metric_key, metric_label)) \
                    in zip([c1, c2], col_pairs[i:i+2]):
                check_data = noise.get(key, {})
                severity = check_data.get("severity", "NONE")
                value = check_data.get(metric_key, 0)

                with col_widget:
                    
                    border_color = {
                        "HIGH": "#c62828", "MEDIUM": "#f57f17",
                        "LOW": "#2e7d32", "NONE": "#9e9e9e"
                    }.get(severity, "#9e9e9e")

                    st.markdown(f"""
                    <div style="border-left:4px solid {border_color};
                                padding:12px; background:white;
                                border-radius:8px; margin-bottom:10px;">
                        <div style="font-weight:bold; font-size:14px;">{display_name}</div>
                        <div style="font-size:24px; font-weight:900;
                                    color:{border_color};">{value}</div>
                        <div style="font-size:11px; color:#666;">{metric_label}</div>
                        <div style="margin-top:6px;">
                            {severity_badge(severity)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

  
    with tab2:
        st.markdown("### Model Complexity Analysis")
        st.caption("5 models trained from simple → complex. Train vs Test AUC at each level.")

        bv = full_report.get("bias_variance", {})
        models_data = bv.get("models", {})

        
        rows = []
        for model_name, model_info in models_data.items():
            rows.append({
                "Model": model_name,
                "Train AUC": f"{model_info['train_auc']:.4f}",
                "Test AUC":  f"{model_info['test_auc']:.4f}",
                "Gap":       f"{model_info['gap']:.4f}",
                "Diagnosis": model_info['diagnosis']
            })

        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        best_model = bv.get("best_model", "N/A")
        best_auc   = bv.get("best_test_auc", 0)
        st.success(f"🏆 Best model: **{best_model}** — Test AUC: **{best_auc:.4f}**")

  
    with tab3:
        st.markdown("### The 5 Sins of Data Leakage")
        st.caption("Detected leakage means reported performance is inflated and untrustworthy.")

        leakage = full_report.get("leakage_scan", {})
        summary = leakage.get("summary", {})

        sins_detected = summary.get("total_sins_detected", 0)
        overall_severity = summary.get("overall_severity", "CLEAN")

        
        if sins_detected == 0:
            st.success(f" No leakage detected — dataset appears CLEAN")
        else:
            st.error(f" {sins_detected} leakage sin(s) detected — Overall: {overall_severity}")

        st.markdown("---")

       
        sin_configs = [
            ("target_encoding_leak", "Sin 1", "Target Encoding Before Split",
             "Encoding using target before train/test split reveals future information."),
            ("feature_from_target",  "Sin 2", "Feature Derived from Target",
             "A feature that predicts the target almost perfectly alone — suspicious."),
            ("scaling_leak",         "Sin 3", "Timestamp-Contaminated Scaling",
             "StandardScaler fit on all data lets test statistics bleed into training."),
            ("group_overlap",        "Sin 4", "Group Overlap",
             "Same entity (patient/borrower) appears in both train and test."),
            ("duplicate_leakage",    "Sin 5", "Duplicate ID Leakage",
             "Exact duplicate rows appear in both train and test sets."),
        ]

        for key, sin_num, sin_name, sin_desc in sin_configs:
            sin_data = leakage.get(key, {})
            detected = sin_data.get("detected", False)
            severity = sin_data.get("severity", "NONE")

         
            border = "#c62828" if detected else "#2e7d32"
            icon = "⚠️" if detected else "✅"
            status_text_str = "DETECTED" if detected else "CLEAN"

           
            inflation = sin_data.get("auc_inflation", None)
            inflation_str = f"AUC inflation: +{inflation:.4f}" if inflation else ""

            st.markdown(f"""
            <div style="border-left:4px solid {border}; padding:14px;
                        background:white; border-radius:8px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between;
                            align-items:center;">
                    <div>
                        <span style="font-size:11px; color:#999;
                                     font-weight:bold;">{sin_num}</span>
                        <span style="font-size:15px; font-weight:bold;
                                     margin-left:8px;">{sin_name}</span>
                    </div>
                    <div style="color:{border}; font-weight:bold;">
                        {icon} {status_text_str}
                    </div>
                </div>
                <div style="font-size:12px; color:#666; margin-top:6px;">
                    {sin_desc}
                </div>
                {f'<div style="font-size:11px; color:#e65100; margin-top:4px; font-weight:bold;">{inflation_str}</div>' if inflation_str else ''}
            </div>
            """, unsafe_allow_html=True)

  
    with tab4:
        st.markdown("### Learning Curve Autopsy")
        st.caption("How does model performance change as training data increases?")

        curve = full_report.get("curve_autopsy", {})
        diagnosis   = curve.get("diagnosis", "N/A")
        explanation = curve.get("explanation", "")
        recommendation = curve.get("recommendation", "")
        curve_data  = curve.get("curve_data", {})

        
        diag_colors_display = {
            "HEALTHY": "success", "DATA-STARVED": "error",
            "OVER-COMPLEX": "warning", "LEAKY": "error"
        }

        if diagnosis == "HEALTHY":
            st.success(f" Diagnosis: **{diagnosis}**")
        elif diagnosis == "OVER-COMPLEX":
            st.warning(f" Diagnosis: **{diagnosis}**")
        else:
            st.error(f" Diagnosis: **{diagnosis}**")

       
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Train AUC", f"{curve.get('final_train_auc', 0):.4f}")
        with col2:
            st.metric("Final Val AUC", f"{curve.get('final_val_auc', 0):.4f}")
        with col3:
            st.metric("Gap (Train−Val)", f"{curve.get('gap', 0):.4f}")

        
        if curve_data:
            fig = plot_learning_curve(curve_data, diagnosis)
            st.pyplot(fig)      
            plt.close(fig)      

        st.markdown("---")
        st.markdown(f"**Explanation:** {explanation}")
        st.markdown(f"**Recommendation:** {recommendation}")


    
    st.markdown("---")
    st.subheader("📄 Download Report")

    try:
        
        reporter = PDFReporter(
            full_report=full_report,
            trust_score_result=trust_result
        )
        pdf_bytes = reporter.generate()

        
        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,                           
            file_name="clinical_ai_failure_report.pdf",
            mime="application/pdf",                 
            use_container_width=True,
            type="primary"
        )
        st.caption("The PDF contains all engine findings, tables, and recommendations.")

    except Exception as e:
        st.warning(f"PDF generation failed: {str(e)}. Raw results are shown above.")


elif run_button and df is None:
    
    st.error("⚠️ Please load a dataset first using the sidebar.")