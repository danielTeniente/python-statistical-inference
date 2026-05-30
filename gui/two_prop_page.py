import streamlit as st
import pandas as pd
from logic.proportions_logic import (
    perform_two_proportion_ztest, 
    get_two_proportion_confint
)
from logic.independence_logic import perform_fisher_exact_test, get_contingency_table
from gui.components import show_code

# --- HELPER FUNCTIONS CON CACHÉ ---
@st.cache_data(show_spinner=False)
def get_binary_columns(df):
    """Escanea el DataFrame una sola vez para encontrar columnas con exactamente 2 categorías."""
    valid_cols = []
    for col in df.columns:
        if df[col].isnull().all():
            continue
        # dropna=True evita que los nulos se cuenten como una tercera categoría
        if df[col].nunique(dropna=True) == 2:
            valid_cols.append(col)
    return valid_cols

@st.cache_data(show_spinner=False)
def get_column_unique_vals(df, col):
    """Extrae valores únicos de forma cacheada."""
    return df[col].dropna().unique()


def render_twoprop_test_page():
    st.title("📊 Two Proportions Test")

    # --- 1. Data Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
    
    df = st.session_state.df

    # --- 2. Lightweight Variable Filtering (Cached) ---
    valid_cols = get_binary_columns(df)

    if len(valid_cols) < 2:
        st.error("The dataset must contain at least two binary columns (e.g., Groups and Outcomes).")
        return

    # --- 3. Configuration UI ---
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        group_col = st.selectbox(
            "Select Grouping Variable", 
            valid_cols, 
            index=0, 
            help="The variable that splits your data into two groups.",
            key="twoprop_group"
        )
        
        # Filtrar para no seleccionar la misma variable en ambos campos
        remaining_cols = [c for c in valid_cols if c != group_col]
        outcome_col = st.selectbox(
            "Select Outcome Variable", 
            remaining_cols if remaining_cols else valid_cols, 
            index=0, 
            key="twoprop_outcome"
        )

    with col2:
        unique_vals = get_column_unique_vals(df, outcome_col)
        success_term = st.selectbox("Value for 'Success'", options=unique_vals, key="twoprop_success")
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"], key="twoprop_alt")

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="twoprop_conf")

    # --- 4. Dynamic Methodology Selection ---
    st.markdown("### Methodology Setup")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        test_tooltip = (
            "**Test Recommendations:**\n\n"
            "🛡️ **Fisher's Exact Test:** Use this for small sample sizes or when expected frequencies in any cell of the contingency table are less than 5. It is an exact test.\n\n"
            "📊 **Z-Test for Two Proportions:** Use this for large samples where expected successes and failures in both groups are at least 10."
        )
        selected_test = st.selectbox(
            "Select Statistical Test",
            ["Fisher's Exact Test", "Z-Test for Two Proportions"],
            help=test_tooltip,
            key="twoprop_test_selector"
        )

    with col_m2:
        ci_tooltip = (
            "**CI Recommendations:**\n\n"
            "🎯 **Newcombe Interval:** Highly recommended for differences in proportions. It performs well even for small sample sizes or extreme proportions.\n\n"
            "📊 **Wald Interval:** The standard textbook method, but can be inaccurate for small samples or proportions close to 0 or 1."
        )
        selected_ci = st.selectbox(
            "Select Confidence Interval Method",
            ["Newcombe Interval", "Wald Interval"],
            help=ci_tooltip,
            key="twoprop_ci_selector"
        )

    # --- 5. Context ID and State Management ---
    current_id = f"{group_col}_{outcome_col}_{success_term}_{confidence}_{alternative}_{selected_test}_{selected_ci}"
    
    if ("twoprop_state" not in st.session_state or 
        st.session_state.get("twoprop_id") != current_id):
        
        st.session_state.twoprop_state = {}  
        st.session_state.twoprop_id = current_id

    state = st.session_state.twoprop_state

    st.divider()

    # --- 6. Data Exploration ---
    with st.expander("📊 Data Exploration: Contingency Table", expanded=False):
        if st.button("Generate Contingency Table", key="btn_twoprop_table"):
            with st.spinner("Calculating table..."):
                table, code_t = get_contingency_table(df, group_col, outcome_col)
                state["table"] = {"data": table, "code": code_t}
        
        if "table" in state:
            res_t = state["table"]
            st.dataframe(res_t["data"])
            show_code(res_t["code"])

    # --- 7. Unified Execution Section ---
    with st.expander(f"🧪 Execution: {selected_test.split('(')[0].strip()} + {selected_ci.split('(')[0].strip()}", expanded=True):
        
        # Validación Didáctica de Suposiciones para el Z-Test
        if selected_test == "Z-Test for Two Proportions":
            stats = (df[outcome_col] == success_term).groupby(df[group_col]).agg(['sum', 'count'])
            successes = stats['sum'].values
            trials = stats['count'].values
            
            if len(trials) == 2:
                p_pool = successes.sum() / trials.sum()
                exp_successes = trials * p_pool
                exp_failures = trials * (1 - p_pool)
                
                if (exp_successes < 10).any() or (exp_failures < 10).any():
                    min_exp = min(exp_successes.min(), exp_failures.min())
                    st.warning(f"🎓 **Assumption Alert:** The minimum expected count under the null hypothesis is **{min_exp:.1f}** (less than 10). The normal approximation may be invalid. **Fisher's Exact Test** is highly recommended here.")
                else:
                    st.success("✔️ **Assumption Met:** All expected success and failure counts in both groups are $\\ge 10$.")

        if st.button("Run Analysis", key="btn_run_twoprop"):
            with st.spinner("Computing statistics..."):
                
                # 1. Ejecutar el test seleccionado
                if selected_test == "Fisher's Exact Test":
                    stat, p_val, code_test = perform_fisher_exact_test(
                        df=df, var1_col=group_col, var2_col=outcome_col, alternative=alternative
                    )
                    stat_label = "Odds Ratio"
                else:
                    stat_z, p_z, code_test = perform_two_proportion_ztest(
                        df=df, group_col=group_col, outcome_col=outcome_col, 
                        success_term=success_term, alternative=alternative
                    )
                    stat, p_val = stat_z, p_z
                    stat_label = "Z-Statistic"

                # 2. Ejecutar el IC seleccionado
                method_str = 'newcomb' if "Newcombe" in selected_ci else 'wald'
                (low, up), code_ci = get_two_proportion_confint(
                    df=df, group_col=group_col, outcome_col=outcome_col,
                    success_term=success_term, confidence=confidence, method=method_str
                )
                
                # Guardar en el estado
                state["results"] = {
                    "stat": stat,
                    "stat_label": stat_label,
                    "p_val": p_val,
                    "ci": (low, up),
                    "code_test": code_test,
                    "code_ci": code_ci
                }

        # Mostrar Resultados
        if "results" in state:
            res = state["results"]
            
            c1, c2, c3 = st.columns(3)
            
            # Manejar el caso donde Fisher's Exact puede retornar None para Odds Ratio
            stat_display = f"{res['stat']:.4f}" if res['stat'] is not None else "N/A"
            
            c1.metric(res['stat_label'], stat_display)
            c2.metric(f"p-value ({alternative})", f"{res['p_val']:.4f}")
            c3.metric("Confidence Interval", f"({res['ci'][0]:.4f}, {res['ci'][1]:.4f})")
            
            st.markdown("#### Logic")
            st.markdown(f"**Test ({selected_test}):**")
            show_code(res["code_test"])
            st.markdown(f"**Confidence Interval ({selected_ci}):**")
            show_code(res["code_ci"])