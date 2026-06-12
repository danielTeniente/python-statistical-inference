import streamlit as st
import pandas as pd
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.twopop_logic import perform_ftest, perform_levene, plot_confidence_interval, get_sample_variance_ratio

@st.cache_data(show_spinner=False)
def cached_numeric_columns(df):
    """Return a list of numeric column names (cached per DataFrame)."""
    return get_numeric_columns(df)

@st.cache_data(show_spinner=False)
def cached_categorical_cols(df):
    """
    Return all categorical columns. We allow columns with more than 
    2 categories, since the user will manually select which 2 to compare.
    """
    return get_categorical_columns(df)

@st.cache_data(show_spinner=False)
def filter_dataframe_by_categories(df, col, categories):
    """
    Filtrado eficiente del dataframe usando isin() y caché.
    """
    return df[df[col].isin(categories)].copy()


def render_twopop_variances_page():
    st.title("Two Population Variances Tests")

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
            "Select numerical variable", numeric_cols, key="tpv_num"
        )
    with col2:
        selected_cat_col = st.selectbox(
            "Select grouping variable", categorical_cols, key="tpv_cat"
        )

    # Extraemos todas las categorías únicas válidas de la columna seleccionada
    unique_categories = df[selected_cat_col].dropna().unique().tolist()
    
    # Multiselect que restringe a un máximo de 2 opciones
    selected_categories = st.multiselect(
        "Select exactly 2 categories to compare",
        options=unique_categories,
        max_selections=2,
        key="tpv_cats_select"
    )

    # Bloqueamos la vista de los tests hasta que existan exactamente 2 categorías
    if len(selected_categories) != 2:
        st.info("ℹ️ Please select exactly two categories to proceed with the tests.")
        return

    st.caption(f"Comparing variances of: **{selected_categories[0]}** vs **{selected_categories[1]}**")

    # Creamos el df_filtrado de forma eficiente
    df_filtrado = filter_dataframe_by_categories(df, selected_cat_col, selected_categories)

    col3, col4 = st.columns(2)
    with col3:
        alternative = st.selectbox(
            "Alternative hypothesis", 
            ["two-sided", "less", "greater"], 
            key="tpv_alt"
        )
    with col4:
        confidence = st.slider(
            "Confidence level", 0.80, 0.99, 0.95, 0.01, key="tpv_conf"
        )

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

    # --- 3. Context ID and Isolated State ---
    cat_str = "_".join(str(c) for c in selected_categories)
    current_context_id = f"{selected_num_col}_{selected_cat_col}_{cat_str}_{alternative}_{confidence}"
    
    if (
        "twopop_var_state" not in st.session_state
        or st.session_state.get("twopop_var_id") != current_context_id
    ):
        st.session_state.twopop_var_state = {}
        st.session_state.twopop_var_id = current_context_id

    state = st.session_state.twopop_var_state
    st.divider()

    # --- 4. Dynamic Test Selection ---
    
    # Texto de ayuda (Tooltip) explicando cuándo usar cada opción
    help_tooltip = (
        "**Test Recommendations:**\n\n"
        "📊 **F-Test:** Use this when you are confident that both populations are strictly normally distributed.\n\n"
        "🛡️ **Levene's Test:** Use this when your data deviates from normality or you are unsure. It is more robust against non-normal data."
    )
    
    selected_test = st.selectbox(
        "Select Statistical Test",
        ["F-Test for Equality of Variances", "Levene's Test for Equality of Variances"],
        help=help_tooltip,
        key="tpv_test_selector"
    )

    # Un único desplegable dinámico para el test estadístico seleccionado
    with st.expander(f"🧪 Test: {selected_test}", expanded=True):
        
        # --- F-TEST LOGIC ---
        if selected_test == "F-Test for Equality of Variances":
            if st.button("Run F‑Test", key="btn_run_ftest"):
                with st.spinner("Computing F‑test statistics..."):
                    f_stat, p_val, ci, code = perform_ftest(
                        df_filtrado, selected_num_col, selected_cat_col, selected_categories, alternative, confidence
                    )
                    state["ftest"] = {
                        "f_stat": f_stat,
                        "p_value": p_val,
                        "ci": ci,
                        "code": code,
                        "is_sampled": False,
                    }

            if "ftest" in state:
                res_f = state["ftest"]
                c1, c2 = st.columns(2)
                c1.metric("F-statistic", f"{res_f['f_stat']:.4f}")
                c2.metric(f"P-value ({alternative})", f"{res_f['p_value']:.4f}")
                st.metric(
                    "Confidence Interval",
                    f"({res_f['ci'][0]:.4f}, {res_f['ci'][1]:.4f})",
                )
                show_code(res_f["code"])

        # --- LEVENE'S TEST LOGIC ---
        elif selected_test == "Levene's Test for Equality of Variances":
            if st.button("Run Levene's Test", key="btn_run_levene"):
                with st.spinner("Computing Levene statistics..."):
                    stat, p_val_l, ci_l, code_l, is_sampled = perform_levene(
                        df_filtrado, selected_num_col, selected_cat_col, selected_categories, confidence
                    )
                    state["levene"] = {
                        "stat": stat,
                        "p_value": p_val_l,
                        "ci": ci_l,
                        "code": code_l,
                        "is_sampled": is_sampled,
                    }

            if "levene" in state:
                res_l = state["levene"]
                c1, c2 = st.columns(2)
                c1.metric("Levene Statistic", f"{res_l['stat']:.4f}")
                c2.metric("P-value", f"{res_l['p_value']:.4f}")
                st.metric(
                    "Confidence Interval",
                    f"({res_l['ci'][0]:.4f}, {res_l['ci'][1]:.4f})",
                )
                if res_l.get("is_sampled"):
                    st.info(
                        "ℹ️ The bootstrap confidence interval was computed on a safety-sampled subset "
                        "of the data (proportional stratified sampling) to avoid memory overload."
                    )
                show_code(res_l["code"])

    # --- 5. Visual Analysis (Independiente del selector) ---
    with st.expander("📊 Visual Analysis (Variance Ratio Plot)", expanded=False):
        
        has_ftest = "ftest" in state
        has_levene = "levene" in state
        
        if not has_ftest and not has_levene:
            st.warning("⚠️ Please run at least one statistical test (F-Test or Levene's) first to obtain a confidence interval for the plot.")
            st.button("Generate Plot", key="btn_gen_var_plot", disabled=True)
        else:
            if st.button("Generate Plot", key="btn_gen_var_plot"):
                with st.spinner("Generating visualization..."):
                    name1, name2 = selected_categories[0], selected_categories[1]
                    
                    active_ci = None
                    ci_source = ""
                    
                    if selected_test == "F-Test for Equality of Variances" and has_ftest:
                        active_ci = state["ftest"]["ci"]
                        ci_source = "F-Test"
                    elif selected_test == "Levene's Test for Equality of Variances" and has_levene:
                        active_ci = state["levene"]["ci"]
                        ci_source = "Levene's Test"
                    else:
                        # Fallback por si cambiaron el dropdown pero no ejecutaron el test
                        if has_ftest:
                            active_ci = state["ftest"]["ci"]
                            ci_source = "F-Test"
                        else:
                            active_ci = state["levene"]["ci"]
                            ci_source = "Levene's Test"
                    
                    sample_variance_ratio, code_ratio = get_sample_variance_ratio(
                        df_filtrado, selected_num_col, selected_cat_col, selected_categories
                    )

                    # 3. Graficamos
                    fig, code_plot = plot_confidence_interval(
                        active_ci[0], active_ci[1], sample_variance_ratio,
                        title=rf"CI for the Variance Ratio ($\sigma_{{{name1}}}^2 / \sigma_{{{name2}}}^2$)",
                        x_label="Ratio",
                        y_label="Variance Test",
                        H0=1,
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