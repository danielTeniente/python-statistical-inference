import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import (
    perform_ttest,
    plot_confidence_interval,
    get_sample_difference_in_means
)

@st.cache_data(show_spinner=False)
def cached_numeric_columns(df):
    """Return a list of numeric column names (cached per DataFrame)."""
    return get_numeric_columns(df)

@st.cache_data(show_spinner=False)
def cached_valid_categorical_cols(df):
    """
    Return categorical columns that have exactly two non‑null unique values.
    This is the lightweight filtering needed for two‑population tests.
    """
    cat_cols = get_categorical_columns(df)
    valid = []
    for col in cat_cols:
        # Use nunique(dropna=True) to ignore NaNs
        if df[col].nunique() == 2:
            valid.append(col)
    return valid

def render_twopop_means_page():
    st.title("Two Population Means Tests")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return

    df = st.session_state.df
    numeric_cols = cached_numeric_columns(df)
    valid_categorical_cols = cached_valid_categorical_cols(df)

    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with exactly two categories.")
        return

    # --- 2. Test Configuration ---
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox(
            "Select numerical variable", numeric_cols, key="tp_num"
        )
    with col2:
        selected_cat_col = st.selectbox(
            "Select grouping variable (2 populations)", valid_categorical_cols, key="tp_cat"
        )

    groups = df[selected_cat_col].dropna().unique()
    st.caption(f"Comparing groups: **{groups[0]}** vs **{groups[1]}**")

    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox(
            "Alternative hypothesis",
            ["two-sided", "less", "greater"],
            key="tp_alt"
        )
    with col4:
        confidence = st.slider(
            "Confidence level", 0.80, 0.99, 0.95, 0.01, key="tp_conf"
        )
    equal_var = st.checkbox("Assume equal variances", value=True, key="tp_eq_var")

    # --- 3. Context ID and Isolated State ---
    current_context_id = (
        f"{selected_num_col}_{selected_cat_col}_{alternative}_{confidence}_{equal_var}"
    )
    if (
        "twopop_means_state" not in st.session_state
        or st.session_state.get("twopop_means_id") != current_context_id
    ):
        st.session_state.twopop_means_state = {}
        st.session_state.twopop_means_id = current_context_id

    state = st.session_state.twopop_means_state
    st.divider()

    with st.expander("🧪 1. T-Test for Means Comparison", expanded=not state.get("ttest")):
        if st.button("Run T-Test", key="btn_run_ttest"):
            with st.spinner("Computing T-test statistics..."):
                t_stat, p_value, ci, code = perform_ttest(
                    df, selected_num_col, selected_cat_col,
                    alternative, confidence, equal_var
                )
                state["ttest"] = {
                    "stat": t_stat,
                    "p": p_value,
                    "ci": ci,
                    "code": code,
                    "is_sampled": False,
                }
            # If we also already have a plot, we can update its CI (avoids stale plot)
            if "plot" in state:
                state["plot"]["ci"] = ci

        if "ttest" in state:
            res = state["ttest"]
            show_code(res["code"])
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("T-statistic", f"{res['stat']:.4f}")
            col_m2.metric(f"P-value ({alternative})", f"{res['p']:.4f}")
            col_m3.metric(
                "Confidence Interval",
                f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})",
            )

    with st.expander("📊 2. Confidence Interval Plot", expanded=False):
        # Verify prerequisite: T‑test must be already computed
        if "ttest" not in state:
            st.warning("⚠️ Please run the T-Test first (expand the section above) to obtain the confidence interval.")
            st.button("Generate Plot", key="btn_gen_plot", disabled=True)
        else:
            if st.button("Generate Plot", key="btn_gen_plot"):
                with st.spinner("Generating visualization..."):
                    # Recover the CI from the already computed T‑test
                    ci = state["ttest"]["ci"]

                    # Calculate sample difference in means
                    sample_diff, code_diff = get_sample_difference_in_means(
                        df, selected_num_col, selected_cat_col
                    )
                    # Generate the plot
                    fig, code_plot = plot_confidence_interval(
                        ci[0], ci[1], sample_diff,
                        title="Confidence Interval for the Difference in Means",
                        x_label="Difference in Means",
                        y_label="Means Test",
                    )
                    state["plot"] = {
                        "fig": fig,
                        "diff": sample_diff,
                        "code_diff": code_diff,
                        "code_plot": code_plot,
                    }

        if "plot" in state:
            res_p = state["plot"]
            st.metric("Sample Difference in Means", f"{res_p['diff']:.4f}")
            st.markdown("**Sample Difference Logic:**")
            show_code(res_p["code_diff"])
            st.pyplot(res_p["fig"])
            st.markdown("**Plotting Logic:**")
            show_code(res_p["code_plot"])