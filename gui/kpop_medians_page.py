import pandas as pd
import streamlit as st
from gui.components import show_code
from logic.basic_code import get_numeric_columns, get_categorical_columns
from logic.kpop_logic import perform_bootstrap_pairwise_median, perform_krustall_wallis

@st.cache_data(show_spinner=False)
def filter_kpop_data(df, cat_col, selected_cats):
    """Caches the heavy Pandas filtering operation to avoid repeating it on multiple button clicks."""
    return df[df[cat_col].isin(selected_cats)].copy()

def render_kpop_medians_page():
    st.title("k-Sample Median Tests")
    
    # --- 1. Initial Data Validations ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please upload a file in the 'Upload Dataset' section first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    all_categorical_cols = get_categorical_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return
    
    # Lightweight check for categories
    valid_categorical_cols = [col for col in all_categorical_cols if df[col].dropna().nunique() >= 3]
            
    if not valid_categorical_cols:
        st.error("Error: The dataset must contain at least one categorical column with 3 or more categories.")
        return

    # --- 2. Selection UI ---
    st.markdown("### Test Setup")
    col1, col2 = st.columns(2)
    with col1:
        selected_num_col = st.selectbox("Select numerical variable", numeric_cols, key="kpop_num")
    with col2:
        selected_cat_col = st.selectbox("Select grouping variable (≥ 3 categories)", valid_categorical_cols, key="kpop_cat")

    available_categories = df[selected_cat_col].dropna().unique().tolist()

    # --- LÍMITE FRONTAL 1: Prevención de Explosión Combinatoria ---
    MAX_CATEGORIES = 12
    st.info(
        f"ℹ️ **Rendering Limit:** To prevent the visualization from freezing your browser, "
        f"you can select a maximum of **{MAX_CATEGORIES} categories** for comparison."
    )

    # Safe default (max 5)
    default_cats = available_categories[:5] if len(available_categories) >= 5 else available_categories

    selected_categories = st.multiselect(
        "Select the specific categories to compare:", 
        available_categories, 
        default=default_cats,
        max_selections=MAX_CATEGORIES,
        key="kpop_categories"
    )

    if len(selected_categories) < 3:
        st.warning("⚠️ Please select at least 3 categories to perform K population tests.")
        return 

    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)

    # --- 3. Context Identification & State Management ---
    current_id = f"{selected_num_col}_{selected_cat_col}_{sorted(selected_categories)}_{confidence}"
    
    if ("kpop_medians_state" not in st.session_state or 
        st.session_state.get("kpop_medians_id") != current_id):
        
        st.session_state.kpop_medians_state = {} 
        st.session_state.kpop_medians_id = current_id

    state = st.session_state.kpop_medians_state

    st.divider()
    
    # --- 4. Granular Test Sections ---

    # SECTION: Kruskal-Wallis
    with st.expander("Kruskal-Wallis H-test for Equality of Medians", expanded=not state.get("kruskal")):
        if st.button("Run Kruskal-Wallis Test", key="btn_kruskal"):
            with st.spinner("Computing Kruskal-Wallis statistics..."):
                # Call cached filtering function
                filtered_df = filter_kpop_data(df, selected_cat_col, selected_categories)
                stat, p_value, code = perform_krustall_wallis(filtered_df, selected_num_col, selected_cat_col)
                state["kruskal"] = {"stat": stat, "p": p_value, "code": code}

        if "kruskal" in state:
            res = state["kruskal"]
            c1, c2 = st.columns(2)
            c1.metric("H Statistic", f"{res['stat']:.4f}")
            c2.metric("P-value", f"{res['p']:.4f}")
            show_code(res["code"])

    # SECTION: Bootstrap Pairwise
    with st.expander("Bootstrap Pairwise Confidence Intervals", expanded=False):
        
        # --- LÍMITE FRONTAL 2: Advertencia de Undersampling para Bootstrap ---
        group_counts = df[df[selected_cat_col].isin(selected_categories)].groupby(selected_cat_col).size()
        if (group_counts > 1500).any():
            st.info(
                "ℹ️ **Cloud Limit Note:** One or more selected groups contains over 1,500 records. "
                "Because bootstrapping is computationally intensive, a random sample of 1,500 records "
                "is being used for those groups to prevent the app from freezing."
            )

        n_resamples = st.slider(
            "Number of resamples", 
            min_value=500, max_value=5000, value=2000, step=500,
            help="Higher values yield more stable intervals but take longer."
        )

        if st.button("Execute Bootstrap Analysis", key="btn_bootstrap"):
            with st.spinner("Running Bootstrap simulations (this may take a few seconds)..."):
                # Call cached filtering function
                filtered_df = filter_kpop_data(df, selected_cat_col, selected_categories)
                
                results_df, fig, code = perform_bootstrap_pairwise_median(
                    filtered_df, selected_num_col, selected_cat_col, confidence, n_resamples
                )
                state["bootstrap"] = {"data": results_df, "fig": fig, "code": code}

        if "bootstrap" in state:
            res = state["bootstrap"]
            st.markdown("**Statistical Results:**")
            st.dataframe(res["data"], use_container_width=True)
            
            st.markdown("**Confidence Interval Plot:**")
            st.pyplot(res["fig"])
            show_code(res["code"])