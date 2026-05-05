import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import (
    perform_mannwhitney,
    plot_confidence_interval,
    get_sample_difference_in_medians
)

@st.cache_data(show_spinner=False)
def cached_numeric_columns(df):
    """Return a list of numeric column names (cached per DataFrame)."""
    return get_numeric_columns(df)

@st.cache_data(show_spinner=False)
def cached_valid_binary_cat_cols(df):
    """
    Return categorical columns with exactly two non-null unique values.
    Used for all two‑population tests.
    """
    cat_cols = get_categorical_columns(df)
    valid = []
    for col in cat_cols:
        if df[col].nunique() == 2:
            valid.append(col)
    return valid


def render_twopop_medians_page():
    st.title("Two Population Medians Tests")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return

    df = st.session_state.df
    numeric_cols = cached_numeric_columns(df)
    valid_categorical_cols = cached_valid_binary_cat_cols(df)

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
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="tpm_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (2 populations)", valid_categorical_cols, key="tpm_cat")

    groups = df[selected_cat_col].dropna().unique()
    st.caption(f"Comparing groups: **{groups[0]}** vs **{groups[1]}**")

    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="tpm_alt")
    with col4:
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="tpm_conf")

    # --- 3. Context ID and Isolated State ---
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{alternative}_{confidence}"
    if (
        "twopop_medians_state" not in st.session_state
        or st.session_state.get("twopop_medians_id") != current_context_id
    ):
        st.session_state.twopop_medians_state = {}
        st.session_state.twopop_medians_id = current_context_id

    state = st.session_state.twopop_medians_state
    st.divider()

    with st.expander("🧪 1. Mann-Whitney U Test (Medians Comparison)", expanded=not state.get("mann_whitney")):
        if st.button("Run Mann-Whitney Test", key="btn_run_mw"):
            with st.spinner("Computing statistics..."):
                u_stat, p_value, ci, code, is_sampled = perform_mannwhitney(
                    df, selected_num_col, selected_cat_col, alternative, confidence
                )
                state["mann_whitney"] = {
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "ci": ci,
                    "code": code,
                    "is_sampled": is_sampled,
                }

        if "mann_whitney" in state:
            res = state["mann_whitney"]
            show_code(res["code"])
            col1, col2, col3 = st.columns(3)
            col1.metric("U-statistic", f"{res['u_stat']:.4f}")
            col2.metric(f"P-value ({alternative})", f"{res['p_value']:.4f}")
            col3.metric(
                "Confidence Interval",
                f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})",
            )
            # Transparency: notify if the bootstrap CI was computed on a safety sample
            if res.get("is_sampled"):
                st.info(
                    "ℹ️ The bootstrap confidence interval was computed on a safety-sampled subset "
                    "of the data (proportional stratified sampling) to avoid memory overload."
                )

    with st.expander("📊 2. Visual Analysis (Confidence Interval Plot)", expanded=False):
        if "mann_whitney" not in state:
            st.warning("⚠️ Please run the Mann-Whitney test first (expand the section above) to obtain the confidence interval.")
            st.button("Generate Plot", key="btn_gen_mw_plot", disabled=True)
        else:
            if st.button("Generate Plot", key="btn_gen_mw_plot"):
                with st.spinner("Generating visualization..."):
                    ci = state["mann_whitney"]["ci"]
                    # Compute sample difference in medians
                    diff_medians, code_diff = get_sample_difference_in_medians(
                        df, selected_num_col, selected_cat_col
                    )
                    fig, code_plot = plot_confidence_interval(
                        ci[0], ci[1], diff_medians,
                        title="Confidence Interval for the Difference in Medians",
                        x_label="Difference in Medians",
                        y_label="Medians Test",
                    )
                    state["plot_data"] = {
                        "diff": diff_medians,
                        "code_diff": code_diff,
                        "fig": fig,
                        "code_plot": code_plot,
                    }

        if "plot_data" in state:
            p_res = state["plot_data"]
            st.metric("Sample Difference in Medians", f"{p_res['diff']:.4f}")
            st.markdown("**Sample Difference Logic:**")
            show_code(p_res["code_diff"])
            st.pyplot(p_res["fig"])
            st.markdown("**Plotting Logic:**")
            show_code(p_res["code_plot"])