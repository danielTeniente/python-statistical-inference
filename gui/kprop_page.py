import streamlit as st
from logic.independence_logic import perform_chi_square_test, get_contingency_table
from gui.components import show_code

def render_kprop_test_page():
    st.title("K Proportions Test (Chi-Square)")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # Para K proporciones, buscamos columnas categóricas. 
    # Filtramos columnas que tengan al menos 2 categorías, con un límite prudente para evitar IDs o variables continuas puras.
    valid_cols = [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns to perform this test.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        group_col = st.selectbox(
            "Select Grouping Variable (K groups)", 
            valid_cols, 
            index=0, 
            help="The variable that splits your data into K distinct groups."
        )
    
    remaining_cols = [c for c in valid_cols if c != group_col]
    
    with col2:
        if remaining_cols:
            outcome_col = st.selectbox("Select Outcome Variable", remaining_cols, index=0)
        else:
            outcome_col = st.selectbox("Select Outcome Variable", valid_cols, index=0)

    if group_col == outcome_col:
        st.warning("⚠️ Grouping variable and Outcome variable should be different.")
        return

    st.markdown("### Test Parameters")
    
    # Aquí implementamos la opción de Yates que discutimos
    apply_yates = st.checkbox(
        "Apply Yates' Continuity Correction", 
        value=True, 
        help="Recommended for 2x2 tables. Scipy automatically ignores this if the table is larger than 2x2, but it's good practice to keep it enabled."
    )

    st.divider()

    # --- Tabla de Contingencia ---
    with st.expander("Contingency Table", expanded=True):
        st.markdown("### Contingency Table (Crosstab) of the selected variables")
        contingency_table, code_ct = get_contingency_table(df, group_col, outcome_col)
        
        show_code(code_ct)
        
        st.write("### Contingency Table:")
        st.dataframe(contingency_table)

    # --- Prueba Chi-cuadrado ---
    with st.expander("Chi-Square Test of Homogeneity", expanded=True):
        st.markdown("### Chi-Square Test to compare K proportions")
        
        chi2_stat, p_val, code_chi2 = perform_chi_square_test(
            df=df, 
            var1_col=group_col, 
            var2_col=outcome_col, 
            correction=apply_yates
        )
        
        show_code(code_chi2)
        
        st.write("### Results")
        st.metric("**Chi-Square Statistic:**", f"{chi2_stat:.4f}")
        st.metric("**p-value:**", f"{p_val:.4f}")
        
        