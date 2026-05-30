import streamlit as st
from logic.independence_logic import (
    perform_chi_square_test, 
    perform_fisher_exact_test, 
    get_contingency_table
)
from gui.components import show_code

def render_independence_test_page():
    st.title("📊 Independence Test")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    valid_cols = [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns to perform this test.")
        return

    # --- 2. Configuration UI ---
    st.markdown("### Configuration")
    col_var1, col_var2 = st.columns(2)
    with col_var1:
        var1_col = st.selectbox("Select Variable 1", valid_cols, index=0, key="ind_var1")
    
    remaining_cols = [c for c in valid_cols if c != var1_col]
    with col_var2:
        var2_col = st.selectbox("Select Variable 2", remaining_cols if remaining_cols else valid_cols, index=0, key="ind_var2")

    # Pre-calculate shape and limits for conditional UI
    clean_len = len(df[[var1_col, var2_col]].dropna())
    n_unique_v1 = df[var1_col].nunique()
    n_unique_v2 = df[var2_col].nunique()
    
    is_2x2_ui = (n_unique_v1 == 2 and n_unique_v2 == 2)
    total_cells = n_unique_v1 * n_unique_v2

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        test_tooltip = (
            "**Test Recommendations:**\n\n"
            "📊 **Chi-Square Test:** Use this for larger sample sizes where the expected frequency in each cell is at least 5.\n\n"
            "🧬 **Fisher's Exact Test:** Use this for small sample sizes or when the table is sparse (expected frequencies < 5). Note: For tables larger than 2x2, this uses Monte Carlo simulation."
        )
        selected_test = st.selectbox(
            "Select Statistical Test",
            ["Chi-Square Test", "Fisher's Exact Test"],
            help=test_tooltip,
            key="ind_test_selector"
        )

    # Initialize default parameters for Context ID
    apply_yates = False
    n_resamples = 2000

    with col_m2:
        if selected_test == "Chi-Square Test":
            st.markdown("**Chi-Square Settings:**")
            apply_yates = st.checkbox(
                "Apply Yates' Continuity Correction", 
                value=True, 
                help="Recommended for 2x2 tables to prevent overestimation of statistical significance for small data.",
                key="ind_yates"
            )
        elif selected_test == "Fisher's Exact Test" and not is_2x2_ui:
            st.markdown("**Monte Carlo Simulation Settings:**")
            n_resamples = st.slider(
                "Number of resamples", 
                min_value=500, max_value=20000, value=2000, step=500,
                help="Higher values increase precision but take longer to compute.",
                key="ind_resamples"
            )
        elif selected_test == "Fisher's Exact Test" and is_2x2_ui:
            st.info("ℹ️ Standard Fisher's Exact Test will be applied natively for this 2x2 table.")

    # --- 4. Context ID and State Management ---
    current_context_id = f"{var1_col}_{var2_col}_{selected_test}_{apply_yates}_{n_resamples}"
    
    if ("independence_state" not in st.session_state or 
        st.session_state.get("independence_context_id") != current_context_id):
        
        st.session_state.independence_state = {} 
        st.session_state.independence_context_id = current_context_id

    state = st.session_state.independence_state

    st.divider()

    # --- 5. Data Exploration ---
    with st.expander("📊 Data Exploration: Contingency Table", expanded=False):
        if st.button("Generate Contingency Table", key="btn_ind_table"):
            with st.spinner("Calculating table..."):
                table, code_t = get_contingency_table(df, var1_col, var2_col)
                state["table"] = {"data": table, "code": code_t}
        
        if "table" in state:
            res_t = state["table"]
            st.dataframe(res_t["data"])
            show_code(res_t["code"])

    # --- 6. Unified Execution Section ---
    with st.expander(f"🧪 Execution: {selected_test}", expanded=True):
        
        fisher_blocked = False
        
        # Validación Didáctica de Suposiciones para Fisher (no 2x2)
        if selected_test == "Fisher's Exact Test" and not is_2x2_ui:
            simulated_table_size = n_resamples * total_cells
            fisher_blocked = (simulated_table_size <= clean_len)

            if fisher_blocked:
                st.error(
                    f"❌ **Test Blocked:** The SciPy heuristic failed. Your simulated table size "
                    f"({n_resamples} resamples × {total_cells} cells = **{simulated_table_size:,}**) "
                    f"must be strictly greater than your total valid rows (**{clean_len:,}**)."
                )
                st.info(
                    "💡 **How to fix this:** Try increasing the number of resamples above, "
                    "or ideally, switch to the **Chi-Square Test** which is statistically appropriate "
                    "for this sample size."
                )

        if st.button("Run Analysis", key="btn_run_ind", disabled=fisher_blocked):
            with st.spinner("Computing statistics (this may take a moment for simulations)..."):
                
                # Ejecutar el test seleccionado
                if selected_test == "Chi-Square Test":
                    stat, p_val, code_test = perform_chi_square_test(
                        df=df, var1_col=var1_col, var2_col=var2_col, correction=apply_yates
                    )
                    stat_label = "Chi-Square Statistic"
                else:
                    stat, p_val, code_test = perform_fisher_exact_test(
                        df=df, var1_col=var1_col, var2_col=var2_col, 
                        alternative="two-sided", n_resamples=n_resamples
                    )
                    stat_label = "Statistic (Odds Ratio)"

                # Guardar en el estado
                state["results"] = {
                    "stat": stat,
                    "stat_label": stat_label,
                    "p_val": p_val,
                    "code_test": code_test
                }

        # Mostrar Resultados
        if "results" in state:
            res = state["results"]
            
            c1, c2 = st.columns(2)
            
            # Manejar el caso donde Fisher's Exact puede retornar None para Odds Ratio (non-2x2)
            stat_display = f"{res['stat']:.4f}" if res['stat'] is not None else "N/A"
            
            c1.metric(res['stat_label'], stat_display)
            c2.metric("p-value", f"{res['p_val']:.4f}")
            
            show_code(res["code_test"])