import streamlit as st
import pandas as pd
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.ovr_logic import perform_ftest_ovr, perform_levene_ovr, get_sample_variance_ratio_ovr
from logic.twopop_logic import plot_confidence_interval

# --- Rule 4: Cache heavy DataFrame scans for UI elements ---
@st.cache_data(show_spinner=False)
def get_valid_ovr_categoricals(df, categorical_cols):
    """Caches the identification of columns with >2 categories."""
    return [col for col in categorical_cols if df[col].nunique() > 2]

@st.cache_data(show_spinner=False)
def get_unique_categories(df, cat_col):
    """Caches the unique categories extraction."""
    return df[cat_col].dropna().unique().tolist()

def render_ovr_variances_page():
    st.title("📊 One-vs-Rest: Variances Tests")
    
    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
        
    # Lightweight check using cached function
    valid_categorical_cols = get_valid_ovr_categoricals(df, all_categorical_cols)
                
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return

    # --- 2. Test Setup (UI Selection) ---
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="ovr_var_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable", valid_categorical_cols, key="ovr_var_cat")
    
    # Use cached function for lightning-fast dropdown
    available_categories = get_unique_categories(df, selected_cat_col)
    target_cat = st.selectbox(
        "Select Target Population ('One')", 
        available_categories, 
        key="ovr_var_target"
    )
    
    st.caption(f"Comparing variances of: **{target_cat}** vs **The Rest**")
    
    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox(
            "Alternative hypothesis", 
            ["two-sided", "less", "greater"], 
            key="ovr_var_alt"
        )
    with col4:
        confidence = st.slider(
            "Confidence level", 0.80, 0.99, 0.95, 0.01, key="ovr_var_conf"
        )

    # --- 3. Context ID and Cache Management ---
    # Construct an ID that uniquely identifies the current test configuration
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{target_cat}_{alternative}_{confidence}"

    # Reset page state if the context changed
    if ("ovr_var_state" not in st.session_state or 
        st.session_state.get("ovr_var_id") != current_context_id):
        
        st.session_state.ovr_var_state = {}  # Clean dictionary for results
        st.session_state.ovr_var_id = current_context_id

    # Reference to the isolated state dictionary
    state = st.session_state.ovr_var_state

    st.divider()

    # --- 4. Dynamic Test Selection ---
    
    # Tooltip explaining when to use each option
    help_tooltip = (
        "**Test Recommendations:**\n\n"
        "📊 **F-Test:** Use this when you are confident that the populations are strictly normally distributed.\n\n"
        "🛡️ **Levene's Test:** Use this when your data deviates from normality or you are unsure. It is more robust against non-normal data."
    )
    
    selected_test = st.selectbox(
        "Select Statistical Test",
        ["F-Test for Equality of Variances", "Levene's Test for Equality of Variances"],
        help=help_tooltip,
        key="ovr_test_selector"
    )

    # A single dynamic expander for the selected statistical test
    with st.expander(f"🧪 Test: {selected_test} (One-vs-Rest)", expanded=True):
        
        # --- F-TEST LOGIC ---
        if selected_test == "F-Test for Equality of Variances":
            if st.button("Run F-Test", key="btn_run_ovr_ftest"):
                with st.spinner("Computing F-test statistics..."):
                    f_stat, p_value, ci, code = perform_ftest_ovr(
                        df, selected_num_col, selected_cat_col, target_cat, alternative, confidence
                    )
                    state["ftest"] = {
                        "stat": f_stat,
                        "p_value": p_value,
                        "ci": ci,
                        "code": code,
                        "is_sampled": False
                    }

            if "ftest" in state:
                res_f = state["ftest"]
                
                f1, f2 = st.columns(2)
                f1.metric("F-statistic", f"{res_f['stat']:.4f}")
                f2.metric(f"P-value ({alternative})", f"{res_f['p_value']:.4f}")
                st.markdown(f"**CI for the Variance Ratio** ($\sigma^2_{{{target_cat}}} / \sigma^2_{{{"Rest"}}}$)")
                st.metric("Confidence Interval", f"({res_f['ci'][0]:.4f}, {res_f['ci'][1]:.4f})")
                show_code(res_f["code"])

        # --- LEVENE'S TEST LOGIC ---
        elif selected_test == "Levene's Test for Equality of Variances":
            if st.button("Run Levene's Test", key="btn_run_ovr_levene"):
                with st.spinner("Computing Levene statistics..."):
                    stat_levene, p_value_levene, ci_levene, code_levene, is_sampled = perform_levene_ovr(
                        df, selected_num_col, selected_cat_col, target_cat, confidence
                    )
                    state["levene"] = {
                        "stat": stat_levene,
                        "p_value": p_value_levene,
                        "ci": ci_levene,
                        "code": code_levene,
                        "is_sampled": is_sampled
                    }

            if "levene" in state:
                res_l = state["levene"]
                
                l1, l2 = st.columns(2)
                l1.metric("Levene Statistic", f"{res_l['stat']:.4f}")
                l2.metric("P-value", f"{res_l['p_value']:.4f}")
                st.markdown(rf"**CI for the Variance Ratio** ($\sigma^2_{{{target_cat}}} / \sigma^2_{{{"Rest"}}}$)")
                st.metric("Confidence Interval", f"({res_l['ci'][0]:.4f}, {res_l['ci'][1]:.4f})")
                
                show_code(res_l["code"])
                # Rule 3: Transparency regarding Undersampling for Bootstrap CI in Levene's
                if res_l.get("is_sampled"):
                    st.info(
                        "ℹ️ **Note:** Calculating Bootstrap confidence intervals on large datasets (>5000 rows) "
                        "causes memory overflow. The data was proportionately sampled before bootstrapping to "
                        "maintain accuracy and prevent crashes."
                    )

# --- 5. Visual Analysis (Independent of selector) ---
    with st.expander("📊 Visual Analysis (Variance Ratio Plot)", expanded=False):
        
        has_ftest = "ftest" in state
        has_levene = "levene" in state
        
        if not has_ftest and not has_levene:
            st.warning("⚠️ Please run at least one statistical test (F-Test or Levene's) first to obtain a confidence interval for the plot.")
            st.button("Generate Plot", key="btn_gen_ovr_var_plot", disabled=True)
        else:
            if st.button("Generate Plot", key="btn_gen_ovr_var_plot"):
                with st.spinner("Generating visualization..."):
                    
                    active_ci = None
                    ci_source = ""
                    
                    # Figure out which test data to plot based on dropdown selection
                    if selected_test == "F-Test for Equality of Variances" and has_ftest:
                        active_ci = state["ftest"]["ci"]
                        ci_source = "F-Test"
                    elif selected_test == "Levene's Test for Equality of Variances" and has_levene:
                        active_ci = state["levene"]["ci"]
                        ci_source = "Levene's Test"
                    else:
                        # Fallback if they changed the dropdown but didn't run the new test
                        if has_ftest:
                            active_ci = state["ftest"]["ci"]
                            ci_source = "F-Test"
                        else:
                            active_ci = state["levene"]["ci"]
                            ci_source = "Levene's Test"

                    # --- Lógica usando la función del backend ---
                    sample_variance_ratio, code_ratio = get_sample_variance_ratio_ovr(
                        df, selected_num_col, selected_cat_col, target_cat
                    )

                    fig, code_plot = plot_confidence_interval(
                        active_ci[0], active_ci[1], sample_variance_ratio, 
                        title=rf"CI for the Variance Ratio ($\sigma^2_{{{target_cat}}} / \sigma^2_{{{"Rest"}}}$)", 
                        x_label="Ratio", 
                        y_label="Variance Test", 
                        H0=1
                    )
                    
                    state["plot"] = {
                        "fig": fig, 
                        "code": code_plot,
                        "ratio": sample_variance_ratio,
                        "code_ratio": code_ratio,
                        "ci_source": ci_source
                    }

            if "plot" in state:
                res_p = state["plot"]
                st.metric("Sample Variance Ratio", f"{res_p['ratio']:.4f}")
                st.markdown("Sample Variance Ratio Logic:")
                show_code(res_p["code_ratio"])
                st.pyplot(res_p["fig"])
                show_code(res_p["code"])