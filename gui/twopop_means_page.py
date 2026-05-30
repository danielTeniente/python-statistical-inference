import streamlit as st
import pandas as pd
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
def cached_categorical_cols(df):
    """
    Return all categorical columns. We now allow columns with more than 
    2 categories, since the user will manually select which 2 to compare.
    """
    return get_categorical_columns(df)

@st.cache_data(show_spinner=False)
def filter_dataframe_by_categories(df, col, categories):
    """
    Filtrado eficiente del dataframe usando isin() y caché para no repetir
    la operación en cada re-render si los datos no han cambiado.
    """
    return df[df[col].isin(categories)].copy()

def render_twopop_means_page():
    st.title("Two Population Means Tests")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return

    df = st.session_state.df
    numeric_cols = cached_numeric_columns(df)
    categorical_cols = cached_categorical_cols(df)

    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    if not categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column.")
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
            "Select grouping variable", categorical_cols, key="tp_cat"
        )

    # Extraemos todas las categorías únicas válidas de la columna seleccionada
    unique_categories = df[selected_cat_col].dropna().unique().tolist()
    
    # Multiselect que restringe a un máximo de 2 opciones
    selected_categories = st.multiselect(
        "Select exactly 2 categories to compare",
        options=unique_categories,
        max_selections=2,
        key="tp_cats_select"
    )

    # Bloqueamos la vista de los tests hasta que existan exactamente 2 categorías
    if len(selected_categories) != 2:
        st.info("ℹ️ Please select exactly two categories to proceed with the tests.")
        return

    st.caption(f"Comparing groups: **{selected_categories[0]}** vs **{selected_categories[1]}**")

    # Creamos el df_filtrado de forma eficiente
    df_filtrado = filter_dataframe_by_categories(df, selected_cat_col, selected_categories)

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

    st.text("🔍 Data Filtering Code")
    filter_code = (
        f"# Assuming 'df' is your loaded pandas DataFrame\n"
        f"selected_categories = {selected_categories}\n"
        f"filtered_df = df[df['{selected_cat_col}'].isin(selected_categories)].copy()\n"
        f"# df = filtered_df  # Use this filtered dataframe for your analysis\n"
    )
    # If the column is categorical, add the line that removes unused categories
    if isinstance(df[selected_cat_col].dtype, pd.CategoricalDtype):
        filter_code += (
            f"# Remove unused categories (safety step for categorical columns)\n"
            f"filtered_df['{selected_cat_col}'] = filtered_df['{selected_cat_col}'].cat.remove_unused_categories()\n"
        )
    show_code(filter_code)

    cat_str = "_".join(str(c) for c in selected_categories)
    current_context_id = (
        f"{selected_num_col}_{selected_cat_col}_{cat_str}_{alternative}_{confidence}_{equal_var}"
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
                    df_filtrado, selected_num_col, selected_cat_col,
                    alternative, confidence, equal_var
                )
                state["ttest"] = {
                    "stat": t_stat,
                    "p": p_value,
                    "ci": ci,
                    "code": code,
                    "is_sampled": False,
                }
            if "plot" in state:
                state["plot"]["ci"] = ci

        if "ttest" in state:
            res = state["ttest"]
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("T-statistic", f"{res['stat']:.4f}")
            col_m2.metric(f"P-value ({alternative})", f"{res['p']:.4f}")
            col_m3.metric(
                "Confidence Interval",
                f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})",
            )
            show_code(res["code"])

    with st.expander("📊 2. Confidence Interval Plot", expanded=False):
        if "ttest" not in state:
            st.warning("⚠️ Please run the T-Test first (expand the section above) to obtain the confidence interval.")
            st.button("Generate Plot", key="btn_gen_plot", disabled=True)
        else:
            if st.button("Generate Plot", key="btn_gen_plot"):
                with st.spinner("Generating visualization..."):
                    ci = state["ttest"]["ci"]

                    # Utilizamos el df_filtrado aquí también
                    sample_diff, code_diff = get_sample_difference_in_means(
                        df_filtrado, selected_num_col, selected_cat_col
                    )
                    
                    # Asignamos dinámicamente los nombres al título usando las categorías seleccionadas
                    name1, name2 = selected_categories[0], selected_categories[1]
                    
                    fig, code_plot = plot_confidence_interval(
                        ci[0], ci[1], sample_diff,
                        title=rf"CI for the Difference in Means ($\mu_{{{name1}}} - \mu_{{{name2}}}$)",
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
            st.markdown("Sample Difference Logic:")
            show_code(res_p["code_diff"])
            st.pyplot(res_p["fig"])
            st.markdown("**Plotting Logic:**")
            show_code(res_p["code_plot"])