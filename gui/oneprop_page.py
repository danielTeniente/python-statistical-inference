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
    st.title("One Proportion Test")

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

    # --- 4. Context ID and State Management ---
    current_context_id = f"{selected_col}_{p0}_{alternative}_{confidence}_{success_term}"
    
    if ("oneprop_state" not in st.session_state or 
        st.session_state.get("oneprop_context_id") != current_context_id):
        
        st.session_state.oneprop_state = {}  
        st.session_state.oneprop_context_id = current_context_id

    state = st.session_state.oneprop_state

    st.divider()

    # --- 5. Granular Execution Sections ---

    # SECTION: Binomial Test
    with st.expander("Binomial Test (Exact)", expanded=not state.get("binomial")):
        st.markdown("*Use this exact test for small sample sizes or extreme proportions.*")
        if st.button("Run Binomial Test", key="btn_run_binomial"):
            with st.spinner("Computing exact binomial probabilities..."):
                stat, p_val, code = perform_one_proportion_binomial_test(
                    df=df, selected_column=selected_col, p0=p0, 
                    alternative=alternative, success_term=success_term
                )
                state["binomial"] = {"stat": stat, "p": p_val, "code": code}
        
        if "binomial" in state:
            res_b = state["binomial"]
            show_code(res_b["code"])
            bm1, bm2 = st.columns(2)
            bm1.metric("Success Proportion", f"{res_b['stat']:.4f} ({res_b['stat']*100:.2f}%)")
            bm2.metric("p-value", f"{res_b['p']:.4f}")

    # SECTION: Z-Test
    with st.expander("Z-Test (Normal Approximation)", expanded=False):
        # Validación Didáctica de Suposiciones
        n = len(df[selected_col].dropna())
        exp_successes = n * p0
        exp_failures = n * (1 - p0)
        
        if exp_successes < 10 or exp_failures < 10:
            st.warning(f"🎓 **Assumption Alert:** Expected successes ({exp_successes:.1f}) or failures ({exp_failures:.1f}) are below 10. The normal approximation may be inaccurate. The **Binomial Test** is recommended.")
        else:
            st.success(f"✔️ **Assumption Met:** Expected counts ($np_0$ = {exp_successes:.1f}, $n(1-p_0)$ = {exp_failures:.1f}) are $\\ge 10$.")

        if st.button("Run Z-Test", key="btn_run_ztest"):
            with st.spinner("Computing Z-statistic..."):
                stat, p_val, code = perform_one_proportion_ztest(
                    df=df, selected_column=selected_col, p0=p0, 
                    alternative=alternative, success_term=success_term
                )
                state["ztest"] = {"stat": stat, "p": p_val, "code": code}
        
        if "ztest" in state:
            res_z = state["ztest"]
            show_code(res_z["code"])
            zm1, zm2 = st.columns(2)
            zm1.metric("Z-Statistic", f"{res_z['stat']:.4f}")
            zm2.metric("p-value", f"{res_z['p']:.4f}")

    # SECTION: Confidence Intervals
    with st.expander("Confidence Intervals for the Proportion", expanded=False):
        ci_col1, ci_col2 = st.columns(2)
        
        with ci_col1:
            st.markdown("**Clopper-Pearson (Exact)**")
            if st.button("Calculate Clopper-Pearson", key="btn_run_clopper"):
                with st.spinner("Computing exact interval..."):
                    (low, up), code = get_one_proportion_interval(
                        df=df, selected_column=selected_col, confidence=confidence,
                        success_term=success_term, method='beta'
                    )
                    state["clopper"] = {"ci": (low, up), "code": code}
            
            if "clopper" in state:
                r_c = state["clopper"]
                st.metric(f"CI ({confidence*100:.0f}%)", f"({r_c['ci'][0]:.4f}, {r_c['ci'][1]:.4f})")
                show_code(r_c["code"])

        with ci_col2:
            st.markdown("**Wilson Score**")
            if st.button("Calculate Wilson Score", key="btn_run_wilson"):
                with st.spinner("Computing Wilson interval..."):
                    (low, up), code = get_one_proportion_interval(
                        df=df, selected_column=selected_col, confidence=confidence,
                        success_term=success_term, method='wilson'
                    )
                    state["wilson"] = {"ci": (low, up), "code": code}
            
            if "wilson" in state:
                r_w = state["wilson"]
                st.metric(f"CI ({confidence*100:.0f}%)", f"({r_w['ci'][0]:.4f}, {r_w['ci'][1]:.4f})")
                show_code(r_w["code"])