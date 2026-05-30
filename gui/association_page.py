import streamlit as st
from logic.association_logic import (
    perform_cramers_v_test,
    perform_pearsons_c_test,
    perform_phi_coefficient_test,
    perform_odds_ratio_test
)
from logic.independence_logic import get_contingency_table 
from gui.components import show_code

@st.cache_data(show_spinner=False)
def get_valid_categorical_columns(df):
    """Retorna columnas que tienen entre 2 y 30 categorías únicas."""
    return [col for col in df.columns if 2 <= df[col].dropna().nunique() <= 30]

def render_association_measures_page():
    st.title("📊 Measures of Association (Effect Size)")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # Usamos la función cacheada
    valid_cols = get_valid_categorical_columns(df)

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two categorical columns (between 2 and 30 categories).")
        return

    # --- 2. Configuration UI ---
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    with col1:
        var1_col = st.selectbox("Select Variable 1", valid_cols, key="assoc_var1")
    
    remaining_cols = [c for c in valid_cols if c != var1_col]
    with col2:
        var2_col = st.selectbox("Select Variable 2", remaining_cols if remaining_cols else valid_cols, key="assoc_var2")

    # Pre-cálculo seguro para is_2x2 (mucho más rápido usando nunique de Pandas sobre la serie ya cargada)
    is_2x2 = (df[var1_col].nunique() == 2 and df[var2_col].nunique() == 2)

    # Construir opciones dinámicamente basadas en la forma de la tabla
    measure_options = ["Cramér's V", "Pearson's Contingency Coefficient (C)"]
    if is_2x2:
        measure_options.extend(["Phi Coefficient (φ)", "Odds Ratio (OR)"])

    measure_tooltip = (
        "**Measure Recommendations:**\n\n"
        "📊 **Cramér's V:** Excellent general-purpose measure for tables of any size (0 to 1 scale).\n\n"
        "📈 **Pearson's C:** Another standard measure, but note its maximum value cannot reach 1.0 for non-square tables.\n\n"
        "📏 **Phi Coefficient (φ):** Specifically designed for 2x2 tables (equivalent to Cramér's V for 2x2).\n\n"
        "🎲 **Odds Ratio (OR):** Compares the relative odds of the occurrence of the outcome of interest given exposure to the variable of interest (only for 2x2)."
    )

    selected_measure = st.selectbox(
        "Select Measure of Association",
        measure_options,
        help=measure_tooltip,
        key="assoc_measure_selector"
    )

    if not is_2x2:
        st.info("ℹ️ **Note:** Phi Coefficient and Odds Ratio are not available because the selected variables do not form a strict 2x2 table.")

    # --- 4. Context ID and State Management ---
    analysis_id = f"{var1_col}_{var2_col}_{selected_measure}"

    if "assoc_state_id" not in st.session_state or st.session_state.assoc_state_id != analysis_id:
        st.session_state.assoc_state_id = analysis_id
        st.session_state.assoc_results = {}  

    state = st.session_state.assoc_results

    st.divider()

    # --- 5. Data Exploration ---
    with st.expander("📊 Data Exploration: Contingency Table", expanded=False):
        if st.button("Generate Contingency Table", key="btn_assoc_ct"):
            with st.spinner("Processing table..."):
                ct, code_ct = get_contingency_table(df, var1_col, var2_col)
                state["table"] = {"ct": ct, "code": code_ct}

        if "table" in state:
            res_t = state["table"]
            st.dataframe(res_t["ct"], use_container_width=True)
            show_code(res_t["code"])

    # --- 6. Unified Execution Section ---
    with st.expander(f"🧪 Execution: {selected_measure.split('(')[0].strip()}", expanded=True):
        if st.button("Calculate Measure", key="btn_run_assoc"):
            with st.spinner(f"Calculating {selected_measure}..."):
                
                try:
                    if selected_measure == "Cramér's V":
                        val, p, code = perform_cramers_v_test(df, var1_col, var2_col)
                        state["results"] = {"type": "standard", "label": "Cramér's V", "val": val, "p": p, "code": code}
                    
                    elif selected_measure == "Pearson's Contingency Coefficient (C)":
                        val, p, code = perform_pearsons_c_test(df, var1_col, var2_col)
                        state["results"] = {"type": "standard", "label": "Pearson's C", "val": val, "p": p, "code": code}
                    
                    elif selected_measure == "Phi Coefficient (φ)":
                        val, p, code = perform_phi_coefficient_test(df, var1_col, var2_col)
                        state["results"] = {"type": "standard", "label": "Phi (φ)", "val": val, "p": p, "code": code}
                    
                    elif selected_measure == "Odds Ratio (OR)":
                        val, low, high, p, code = perform_odds_ratio_test(df, var1_col, var2_col)
                        state["results"] = {
                            "type": "odds", "label": "Odds Ratio", "val": val, 
                            "low": low, "high": high, "p": p, "code": code
                        }
                except ZeroDivisionError:
                    st.error("❌ It was not possible to calculate the Odds Ratio due to zero counts in the contingency table. Consider adding a small constant (e.g., 0.5) to all cells or using a different measure.")
                    # Clean up failed state
                    if "results" in state:
                        del state["results"]
                except Exception as e:
                    st.error(f"❌ Error calculating measure: {e}")
                    if "results" in state:
                        del state["results"]

        # Renderizar Resultados
        if "results" in state:
            res = state["results"]
            
            if res["type"] == "standard":
                c1, c2 = st.columns(2)
                c1.metric(f"{res['label']}", f"{res['val']:.6f}")
                c2.metric("p-value", f"{res['p']:.4f}")
            elif res["type"] == "odds":
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Odds Ratio", f"{res['val']:.4f}")
                c2.metric("CI Lower (95%)", f"{res['low']:.4f}")
                c3.metric("CI Upper (95%)", f"{res['high']:.4f}")
                c4.metric("p-value", f"{res['p']:.4f}")

            show_code(res["code"])