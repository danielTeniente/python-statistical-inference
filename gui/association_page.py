import streamlit as st
from logic.association_logic import (
    perform_cramers_v_test,
    perform_pearsons_c_test,
    perform_phi_coefficient_test,
    perform_odds_ratio_test
)
# Asumiendo que get_contingency_table está en tu módulo de independencia o utilidad compartida
from logic.independence_logic import get_contingency_table 
from gui.components import show_code

def render_association_measures_page():
    st.title("Measures of Association (Effect Size)")

    # --- Validaciones de Datos ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # Filtrar columnas categóricas (mismo criterio de 2 a 30 categorías)
    valid_cols = [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns to perform this analysis.")
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

    st.divider()

    # --- Tabla de Contingencia ---
    with st.expander("Contingency Table", expanded=True):
        st.markdown("### Contingency Table (Crosstab)")
        contingency_table, code_ct = get_contingency_table(df, var1_col, var2_col)
        
        # Obtenemos la forma para las validaciones individuales posteriores
        r, c = contingency_table.shape
        is_2x2 = (r == 2 and c == 2)
        
        show_code(code_ct)
        st.write("### Crosstab Results:")
        st.dataframe(contingency_table)

    # --- 1. V de Cramér ---
    with st.expander("1. Cramér's V", expanded=True):
        st.markdown("### Cramér's V")
        st.info("💡 Valid for contingency tables of any size (R x C). Values range from 0 (no association) to 1 (perfect association).")
        
        cramers_v, p_val_cv, code_cv = perform_cramers_v_test(df, var1_col, var2_col)
        show_code(code_cv)
        
        st.write("### Results")
        col_cv1, col_cv2 = st.columns(2)
        col_cv1.metric("**Cramér's V:**", f"{cramers_v:.4f}")
        col_cv2.metric("**p-value (Ref):**", f"{p_val_cv:.4f}")
        
        st.caption("*Interpretation Guide: ~0.1 = Weak, ~0.3 = Moderate, >0.5 = Strong association.*")

    # --- 2. Coeficiente C de Pearson ---
    with st.expander("2. Pearson's Contingency Coefficient (C)", expanded=True):
        st.markdown("### Pearson's Contingency Coefficient")
        st.info("💡 Valid for tables of any size. Note that its maximum value is always less than 1 (depends on table size).")
        
        pearson_c, p_val_pc, code_pc = perform_pearsons_c_test(df, var1_col, var2_col)
        show_code(code_pc)
        
        st.write("### Results")
        col_pc1, col_pc2 = st.columns(2)
        col_pc1.metric("**Pearson's C:**", f"{pearson_c:.4f}")
        col_pc2.metric("**p-value (Ref):**", f"{p_val_pc:.4f}")

    # --- 3. Coeficiente Phi ---
    with st.expander("3. Phi Coefficient (φ)", expanded=True):
        st.markdown("### Phi Coefficient")
        
        if not is_2x2:
            st.warning(f"⚠️ **Skipped:** The Phi Coefficient requires exactly a 2x2 table. Your current selection generates a {r}x{c} table.")
        else:
            phi, p_val_phi, code_phi = perform_phi_coefficient_test(df, var1_col, var2_col)
            show_code(code_phi)
            
            st.write("### Results")
            col_phi1, col_phi2 = st.columns(2)
            col_phi1.metric("**Phi Coefficient:**", f"{phi:.4f}")
            col_phi2.metric("**p-value (Ref):**", f"{p_val_phi:.4f}")

    # --- 4. Odds Ratio ---
    with st.expander("4. Odds Ratio (OR)", expanded=True):
        st.markdown("### Odds Ratio")
        
        if not is_2x2:
            st.warning(f"⚠️ **Skipped:** The Odds Ratio requires exactly a 2x2 table. Your current selection generates a {r}x{c} table.")
        else:
            odds_ratio, ci_low, ci_high, p_val_or, code_or = perform_odds_ratio_test(df, var1_col, var2_col)
            show_code(code_or)
            
            st.write("### Results")
            col_or1, col_or2, col_or3 = st.columns(3)
            col_or1.metric("**Odds Ratio:**", f"{odds_ratio:.4f}")
            col_or2.metric("**95% CI Lower:**", f"{ci_low:.4f}")
            col_or3.metric("**95% CI Upper:**", f"{ci_high:.4f}")
            
            st.caption("*Interpretation Guide: OR = 1 implies no association. OR > 1 means higher odds of outcome in the first group. OR < 1 means lower odds.*")