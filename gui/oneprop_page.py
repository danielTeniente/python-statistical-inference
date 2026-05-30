import streamlit as st
import pandas as pd
from logic.proportions_logic import (
    perform_one_proportion_binomial_test, 
    perform_one_proportion_ztest, 
    get_one_proportion_interval
)
from gui.components import show_code

# --- HELPER FUNCTIONS CON CACHÉ ---
@st.cache_data(show_spinner=False)
def get_valid_binary_columns(df):
    """Escanea el DataFrame una sola vez para encontrar columnas binarias válidas."""
    valid_cols = []
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            valid_cols.append(col)
        elif df[col].nunique() == 2:
            valid_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                valid_cols.append(col)
    return valid_cols

@st.cache_data(show_spinner=False)
def get_column_metadata(df, col):
    """Extrae los valores únicos y tipo de dato de una columna específica cacheando el resultado."""
    unique_vals = df[col].dropna().unique()
    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    is_0_1_only = set(unique_vals).issubset({0, 1, 0.0, 1.0})
    return unique_vals, is_numeric, is_0_1_only


def render_oneprop_test_page():
    st.title("📊 One Proportion Test")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # --- 2. Lightweight Variable Filtering (Cached) ---
    valid_cols = get_valid_binary_columns(df)

    if not valid_cols:
        st.error("The dataset does not contain any valid binary columns (0/1 or exactly two categories).")
        return

    # --- 3. Configuration UI ---
    st.markdown("### Configuration")
    col_ui1, col_ui2 = st.columns(2)
    
    with col_ui1:
        selected_col = st.selectbox("Select variable", valid_cols, key="one_prop_col")
        p0 = st.number_input(
            label="Hypothesized Proportion (H₀)", 
            value=0.50, min_value=0.01, max_value=0.99, step=0.05,
            key="one_prop_p0"
        )
    
    # Logic for Success Term (Cached)
    unique_vals, is_numeric, is_0_1_only = get_column_metadata(df, selected_col)
    
    success_term = None
    with col_ui2:
        if not (is_numeric and is_0_1_only):
            success_term = st.selectbox(
                "Value for 'Success'", 
                options=unique_vals,
                key="one_prop_success"
            )
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"], key="one_prop_alt")
    
    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="one_prop_conf")

    # --- 4. Dynamic Methodology Selection ---
    st.markdown("### Methodology Setup")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        test_tooltip = (
            "**Test Recommendations:**\n\n"
            "🎯 **Binomial Test (Exact):** Use this for small sample sizes or when your hypothesized proportion is close to 0 or 1. It is always valid.\n\n"
            "📊 **Z-Test (Normal Approximation):** Use this for large samples where expected successes ($np_0$) and expected failures ($n(1-p_0)$) are at least 10."
        )
        selected_test = st.selectbox(
            "Select Statistical Test",
            ["Binomial Test (Exact)", "Z-Test (Normal Approximation)"],
            help=test_tooltip,
            key="oneprop_test_selector"
        )

    with col_m2:
        ci_tooltip = (
            "**CI Recommendations:**\n\n"
            "🎯 **Clopper-Pearson (Exact):** Highly conservative. Best for small sample sizes.\n\n"
            "📊 **Wilson Score:** Often preferred in practice, even for extreme proportions."
        )
        selected_ci = st.selectbox(
            "Select Confidence Interval Method",
            ["Clopper-Pearson (Exact)", "Wilson Score"],
            help=ci_tooltip,
            key="oneprop_ci_selector"
        )

    # --- 5. Context ID and State Management ---
    current_context_id = f"{selected_col}_{p0}_{alternative}_{confidence}_{success_term}_{selected_test}_{selected_ci}"
    
    if ("oneprop_state" not in st.session_state or 
        st.session_state.get("oneprop_context_id") != current_context_id):
        
        st.session_state.oneprop_state = {}  
        st.session_state.oneprop_context_id = current_context_id

    state = st.session_state.oneprop_state

    st.divider()

    # --- 6. Unified Execution Section ---
    with st.expander(f"🧪 Execution: {selected_test.split('(')[0].strip()} + {selected_ci.split('(')[0].strip()}", expanded=True):
        
        # Validación Didáctica de Suposiciones para el Z-Test
        if selected_test == "Z-Test (Normal Approximation)":
            n = len(df[selected_col].dropna())
            exp_successes = n * p0
            exp_failures = n * (1 - p0)
            
            if exp_successes < 10 or exp_failures < 10:
                st.warning(f"🎓 **Assumption Alert:** Expected successes ($np_0$ = {exp_successes:.1f}) or failures ($n(1-p_0)$ = {exp_failures:.1f}) are below 10. The normal approximation may be inaccurate. The **Binomial Test** is highly recommended here.")

        if st.button("Run Analysis", key="btn_run_oneprop"):
            with st.spinner("Computing statistics..."):
                
                # 1. Ejecutar el test seleccionado
                if selected_test == "Binomial Test (Exact)":
                    stat, p_val, code_test = perform_one_proportion_binomial_test(
                        df=df, selected_column=selected_col, p0=p0, 
                        alternative=alternative, success_term=success_term
                    )
                else:
                    stat, p_val, code_test = perform_one_proportion_ztest(
                        df=df, selected_column=selected_col, p0=p0, 
                        alternative=alternative, success_term=success_term
                    )
                
                # 2. Ejecutar el IC seleccionado
                method_str = 'beta' if "Clopper-Pearson" in selected_ci else 'wilson'
                (low, up), code_ci = get_one_proportion_interval(
                    df=df, selected_column=selected_col, confidence=confidence,
                    success_term=success_term, method=method_str
                )
                
                # Guardar en el estado
                state["results"] = {
                    "stat": stat,
                    "p_val": p_val,
                    "ci": (low, up),
                    "code_test": code_test,
                    "code_ci": code_ci
                }

        # Mostrar Resultados
        if "results" in state:
            res = state["results"]
            
            c1, c2, c3 = st.columns(3)
            if selected_test == "Binomial Test (Exact)":
                c1.metric("Sample Proportion", f"{res['stat']:.4f} ({res['stat']*100:.2f}%)")
            else:
                c1.metric("Z-Statistic", f"{res['stat']:.4f}")
                
            c2.metric(f"p-value ({alternative})", f"{res['p_val']:.4f}")
            c3.metric(f"Confidence Interval", f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})")
            
            st.markdown("#### Logic")
            st.markdown("**Test:**")
            show_code(res["code_test"])
            st.markdown(f"**Confidence Interval ({selected_ci.split('(')[0].strip()}):**")
            show_code(res["code_ci"])