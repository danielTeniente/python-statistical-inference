import streamlit as st
import pandas as pd
from logic.independence_logic import perform_chi_square_test, get_contingency_table
from gui.components import show_code

# --- HELPER FUNCTIONS CON CACHÉ ---
@st.cache_data(show_spinner=False)
def get_categorical_columns(df):
    """Escanea el DataFrame una sola vez para encontrar columnas con 2 a 30 categorías."""
    valid_cols = []
    for col in df.columns:
        if df[col].isnull().all():
            continue
        # dropna=True evita contar los nulos como una categoría válida
        if 2 <= df[col].nunique(dropna=True) <= 30:
            valid_cols.append(col)
    return valid_cols

@st.cache_data(show_spinner=False)
def get_unique_counts(df, col1, col2):
    """Obtiene rápidamente la cantidad de categorías para definir la forma de la tabla."""
    return df[col1].nunique(dropna=True), df[col2].nunique(dropna=True)


def render_kprop_test_page():
    st.title("K Proportions Test (Chi-Square)")

    # --- 1. Data Validations ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # --- 2. Lightweight Variable Filtering (Cached) ---
    valid_cols = get_categorical_columns(df)

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns (2 to 30 categories) to perform this test.")
        return

    # --- 3. Variable Selection ---
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        group_col = st.selectbox(
            "Select Grouping Variable (K groups)", 
            valid_cols, 
            index=0, 
            key="kprop_group_var"
        )
    
    remaining_cols = [c for c in valid_cols if c != group_col]
    
    with col2:
        outcome_col = st.selectbox(
            "Select Outcome Variable", 
            remaining_cols if remaining_cols else valid_cols, 
            index=0,
            key="kprop_outcome_var"
        )

    if group_col == outcome_col:
        st.warning("⚠️ Grouping variable and Outcome variable should be different.")
        return

    # Didáctica: Verificar si la tabla es 2x2 para la corrección de Yates
    k_groups, k_outcomes = get_unique_counts(df, group_col, outcome_col)
    is_2x2 = (k_groups == 2 and k_outcomes == 2)

    if is_2x2:
        apply_yates = st.checkbox("Apply Yates' Continuity Correction", value=True, key="kprop_yates")
    else:
        apply_yates = False
        st.info(f"💡 **Didactic Note:** Yates' Continuity Correction is mathematically designed only for 2x2 tables. Your selected variables form a **{k_groups}x{k_outcomes}** table, so standard Chi-Square will be used.")

    # --- 4. Context ID and State Management ---
    current_context_id = f"{group_col}_{outcome_col}_{apply_yates}"

    if ("kprop_state" not in st.session_state or 
        st.session_state.get("kprop_context_id") != current_context_id):
        
        st.session_state.kprop_state = {}  
        st.session_state.kprop_context_id = current_context_id

    state = st.session_state.kprop_state

    st.divider()

    # --- 5. Granular Execution (On-Demand) ---

    # SECTION: Contingency Table
    with st.expander("📊 1. Contingency Table", expanded=not state.get("table")):
        if st.button("Generate Contingency Table", key="btn_kprop_table"):
            with st.spinner("Calculating table..."):
                contingency_table, code_ct = get_contingency_table(df, group_col, outcome_col)
                state["table"] = {"data": contingency_table, "code": code_ct}

        if "table" in state:
            res_t = state["table"]
            show_code(res_t["code"])
            st.write("### Crosstab Results:")
            st.dataframe(res_t["data"])

    # SECTION: Chi-Square Test
    with st.expander("🧪 2. Chi-Square Test of Homogeneity", expanded=False):
        st.warning("🎓 **Assumption Reminder:** The Chi-Square test assumes that at least 80% of the cells have an expected frequency of 5 or more, and no cell has an expected frequency of less than 1. If this is violated, consider combining categories or using Fisher's Exact Test.")
        
        if st.button("Run Chi-Square Test", key="btn_kprop_chi"):
            with st.spinner("Computing statistics..."):
                chi2_stat, p_val, code_chi2 = perform_chi_square_test(
                    df=df, 
                    var1_col=group_col, 
                    var2_col=outcome_col, 
                    correction=apply_yates
                )
                state["chi_test"] = {"stat": chi2_stat, "p": p_val, "code": code_chi2}

        if "chi_test" in state:
            res_c = state["chi_test"]
            show_code(res_c["code"])
            
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Chi-Square Statistic", f"{res_c['stat']:.4f}")
            col_res2.metric("p-value", f"{res_c['p']:.4f}")