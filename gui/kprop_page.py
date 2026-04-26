import streamlit as st
from logic.independence_logic import perform_chi_square_test, get_contingency_table
from gui.components import show_code

def render_kprop_test_page():
    st.title("K Proportions Test (Chi-Square)")

    # --- 1. Data Validations ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # Lightweight filtering for UI selection
    valid_cols = [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns to perform this test.")
        return

    # --- 2. Variable Selection ---
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

    apply_yates = st.checkbox(
        "Apply Yates' Continuity Correction", 
        value=True, 
        key="kprop_yates"
    )

    # --- 3. Context ID and State Management ---
    # Create unique ID for the current selection
    current_context_id = f"{group_col}_{outcome_col}_{apply_yates}"

    # Reset results if the input parameters change
    if ("kprop_state" not in st.session_state or 
        st.session_state.get("kprop_context_id") != current_context_id):
        
        st.session_state.kprop_state = {}  # Isolated state dictionary
        st.session_state.kprop_context_id = current_context_id

    # Reference to the isolated state
    state = st.session_state.kprop_state

    st.divider()

    # --- 4. Granular Execution (On-Demand) ---

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