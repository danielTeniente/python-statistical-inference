import streamlit as st
from logic.independence_logic import (
    perform_chi_square_test, 
    perform_fisher_exact_test, 
    get_contingency_table
)
from gui.components import show_code

def render_independence_test_page():
    st.title("Independence Test (Chi-Square & Fisher)")

    # --- Validaciones de Datos ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # Filtrar columnas categóricas (mismo criterio de 2 a 30 categorías)
    valid_cols = [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns to perform this test.")
        return

    # --- Selección de Variables ---
    col1, col2 = st.columns(2)
    
    with col1:
        var1_col = st.selectbox(
            "Select Variable 1", 
            valid_cols, 
            index=0, 
            help="First categorical variable to evaluate."
        )
    
    remaining_cols = [c for c in valid_cols if c != var1_col]
    
    with col2:
        if remaining_cols:
            var2_col = st.selectbox("Select Variable 2", remaining_cols, index=0)
        else:
            var2_col = st.selectbox("Select Variable 2", valid_cols, index=0)

    if var1_col == var2_col:
        st.warning("⚠️ Variable 1 and Variable 2 should be different.")
        return

    st.markdown("### Test Parameters")
    
    apply_yates = st.checkbox(
        "Apply Yates' Continuity Correction (For Chi-Square)", 
        value=True, 
        help="Recommended for 2x2 tables. Scipy automatically ignores this if the table is larger than 2x2."
    )

    st.divider()

    # --- Tabla de Contingencia ---
    with st.expander("Contingency Table", expanded=True):
        st.markdown("### Contingency Table (Crosstab)")
        contingency_table, code_ct = get_contingency_table(df, var1_col, var2_col)
        
        # Obtenemos la forma para las alertas dinámicas en Fisher
        r, c = contingency_table.shape
        is_2x2 = (r == 2 and c == 2)
        
        show_code(code_ct)
        st.write("### Crosstab Results:")
        st.dataframe(contingency_table)

    # --- Prueba de Fisher (Adaptada para Monte Carlo) ---
    with st.expander("2. Fisher's Exact Test", expanded=True):
        st.markdown("### Fisher's Exact Test")
        
        # Mensaje informativo dinámico para el usuario
        if is_2x2:
            st.info("💡 Standard Fisher's Exact Test applies to 2x2 tables.")
        else:
            st.info(f"💡 Table is {r}x{c}. Using **Monte Carlo simulation** to approximate Fisher's Exact p-value.")
            
        # Llamamos a tu función. Ignoramos la tabla retornada usando '_' porque ya la mostramos arriba
        fisher_stat, fisher_p_val, code_fisher = perform_fisher_exact_test(
            df=df, 
            var1_col=var1_col, 
            var2_col=var2_col,
            alternative="two-sided"
        )
        
        show_code(code_fisher)
        
        st.write("### Fisher's Test Results")
        
        col_f1, col_f2 = st.columns(2)
        
        # Validamos si el statistic existe (es None para > 2x2)
        if fisher_stat is not None:
            col_f1.metric("**Statistic (Odds Ratio):**", f"{fisher_stat:.4f}")
        else:
            col_f1.metric("**Statistic (Odds Ratio):**", "N/A (2x2 Only)")
            
        col_f2.metric("**p-value:**", f"{fisher_p_val:.4f}")

    # --- Prueba Chi-cuadrado ---
    with st.expander("1. Chi-Square Test of Independence", expanded=True):
        st.markdown("### Pearson's Chi-Square Test")
        
        chi2_stat, chi_p_val, code_chi2 = perform_chi_square_test(
            df=df, 
            var1_col=var1_col, 
            var2_col=var2_col, 
            correction=apply_yates
        )
        
        show_code(code_chi2)
        
        st.write("### Chi-Square Results")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("**Chi-Square Statistic:**", f"{chi2_stat:.4f}")
        col_res2.metric("**p-value:**", f"{chi_p_val:.4f}")
        
        
