import streamlit as st
from logic.independence_logic import (
    perform_chi_square_test, 
    perform_fisher_exact_test, 
    get_contingency_table
)
from gui.components import show_code

def render_independence_test_page():
    st.title("Independence Test (Chi-Square & Fisher)")

    # --- 1. Validaciones de Datos Iniciales ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    valid_cols = [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns to perform this test.")
        return

    # --- 2. Selección de Variables ---
    col_var1, col_var2 = st.columns(2)
    with col_var1:
        var1_col = st.selectbox("Select Variable 1", valid_cols, index=0)
    
    remaining_cols = [c for c in valid_cols if c != var1_col]
    with col_var2:
        var2_col = st.selectbox("Select Variable 2", remaining_cols if remaining_cols else valid_cols, index=0)

    apply_yates = st.checkbox(
        "Apply Yates' Continuity Correction (For Chi-Square)", 
        value=True, 
        help="Recommended for 2x2 tables."
    )

    # --- 3. Gestión de Identificadores y Estado ---
    current_context_id = f"{var1_col}_{var2_col}_{apply_yates}"
    
    if ("independence_state" not in st.session_state or 
        st.session_state.get("independence_context_id") != current_context_id):
        
        st.session_state.independence_state = {} 
        st.session_state.independence_context_id = current_context_id

    state = st.session_state.independence_state

    st.divider()

    # --- 4. Lógica de Evaluación y Límites Matemáticos ---
    clean_len = len(df[[var1_col, var2_col]].dropna())
    n_unique_v1 = df[var1_col].nunique()
    n_unique_v2 = df[var2_col].nunique()
    
    is_2x2_ui = (n_unique_v1 == 2 and n_unique_v2 == 2)
    total_cells = n_unique_v1 * n_unique_v2

    # --- 5. Contenedores de Ejecución Granular ---

    # SECCIÓN: TABLA DE CONTINGENCIA
    with st.expander("📊 1. Contingency Table", expanded=not state.get("table")):
        if st.button("Generate Contingency Table", key="btn_table"):
            with st.spinner("Calculating table..."):
                table, code = get_contingency_table(df, var1_col, var2_col)
                state["table"] = {"data": table, "code": code}
        
        if "table" in state:
            res = state["table"]
            show_code(res["code"])
            st.write("### Crosstab Results:")
            st.dataframe(res["data"])

    # SECCIÓN: CHI-SQUARE
    with st.expander("🧪 2. Chi-Square Test", expanded=False):
        if st.button("Run Chi-Square Test", key="btn_chi"):
            with st.spinner("Computing statistics..."):
                chi2, p, code = perform_chi_square_test(
                    df=df, var1_col=var1_col, var2_col=var2_col, correction=apply_yates
                )
                state["chi_square"] = {"stat": chi2, "p": p, "code": code}

        if "chi_square" in state:
            res = state["chi_square"]
            show_code(res["code"])
            c1, c2 = st.columns(2)
            c1.metric("Chi-Square Statistic", f"{res['stat']:.4f}")
            c2.metric("p-value", f"{res['p']:.4f}")

    # SECCIÓN: FISHER'S EXACT TEST
    with st.expander("🧬 3. Fisher's Exact Test", expanded=False):
        
        # Parámetro interactivo de remuestreo (solo visible si no es 2x2)
        n_resamples = 2000 # Valor por defecto seguro para 2x2
        if not is_2x2_ui:
            st.markdown("**Monte Carlo Simulation Settings:**")
            n_resamples = st.slider(
                "Number of resamples (n_resamples)", 
                min_value=500, max_value=20000, value=2000, step=500,
                help="Higher values increase precision but take longer to compute."
            )
            
        # MATEMÁTICA DE PREVENCIÓN Sincronizada con el input del usuario
        simulated_table_size = n_resamples * total_cells
        fisher_blocked = (not is_2x2_ui) and (simulated_table_size <= clean_len)

        # UI Reactiva para Fisher
        if is_2x2_ui:
            st.info("Standard Fisher's Exact Test will be applied (2x2 tables calculate exact factorials natively).")
        else:
            if fisher_blocked:
                st.error(
                    f"❌ **Test Blocked:** The SciPy heuristic failed. Your simulated table size "
                    f"({n_resamples} resamples × {total_cells} cells = **{simulated_table_size:,}**) "
                    f"must be strictly greater than your total valid rows (**{clean_len:,}**)."
                )
                st.info(
                    "💡 **How to fix this:** You can try increasing the number of resamples above, "
                    "or ideally, use the **Chi-Square Test** (above) which is the statistically correct "
                    "method for large sample sizes."
                )
            else:
                st.success(
                    f"✅ Simulation valid. Simulated size (**{simulated_table_size:,}**) > Sample size (**{clean_len:,}**)."
                )

        if st.button("Run Fisher Test", key="btn_fisher", disabled=fisher_blocked):
            with st.spinner("Executing Fisher simulation (this may take a moment)..."):
                f_stat, f_p, code = perform_fisher_exact_test(
                    df=df, var1_col=var1_col, var2_col=var2_col, 
                    alternative="two-sided", n_resamples=n_resamples
                )
                state["fisher"] = {"stat": f_stat, "p": f_p, "code": code}

        if "fisher" in state:
            res = state["fisher"]
            show_code(res["code"])
            cf1, cf2 = st.columns(2)
            
            val_stat = f"{res['stat']:.4f}" if res['stat'] is not None else "N/A"
            cf1.metric("Statistic (Odds Ratio)", val_stat)
            cf2.metric("p-value", f"{res['p']:.4f}")