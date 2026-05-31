import streamlit as st
from logic.basic_code import get_numeric_columns
from logic.descriptive_stats_page_logic import (
    describe_dataset, get_histogram, get_boxplot, get_grouped_boxplot,
    get_sample_size, get_dataset_size, get_mean, 
    get_median, get_mode, get_std, get_variance,
    get_min, get_max, get_range, get_quartiles,
    get_iqr, get_skewness, get_kurtosis
)
from gui.components import show_code

def render_descriptive_numerical_page():
    # --- 1. Initial Validation ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first to see descriptive statistics.")
        return

    df = st.session_state.df
    st.title("📈 Descriptive Statistics: Numerical Variables")
    
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("The dataset does not contain any numeric columns.")
        return

    # --- 2. State Initialization ---
    if "desc_state" not in st.session_state:
        st.session_state.desc_state = {
            "summary": {},
            "individual": {},
            "plots": {}
        }
    
    state = st.session_state.desc_state

    # --- SECTION 1: STATISTICAL SUMMARY ---
    st.subheader("1. Statistical Summary")
    
    # Context ID for Section 1 (based on dataset shape)
    summary_id = f"summary_{df.shape}"
    if st.session_state.get("id_summary") != summary_id:
        state["summary"] = {}
        st.session_state.id_summary = summary_id

    if st.button("Generate General Summary", key="btn_summary"):
        with st.spinner("Processing dataset summary..."):
            size, size_code = get_sample_size(df)
            shape, shape_code = get_dataset_size(df)
            desc, desc_code = describe_dataset(df)
            
            state["summary"] = {
                "size": size, "size_code": size_code,
                "shape": shape, "shape_code": shape_code,
                "desc": desc, "desc_code": desc_code
            }

    if state["summary"]:
        res = state["summary"]

        st.markdown("**Shape of the dataset (rows, columns)**")
        # two columns for better layout
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Number of Rows/Observations", value=res["shape"][0])
        with c2:
            st.metric(label="Number of Columns/Variables", value=res["shape"][1])
        show_code(res["shape_code"])

        st.markdown("**Summary table**")
        show_code(res["desc_code"])
        st.dataframe(res["desc"], use_container_width=True)

    st.divider()

    # --- SECTION 2: DESCRIPTIVE STATISTICS FOR EACH COLUMN ---
    st.subheader("2. Descriptive Statistics for Each Column")
    selected_column = st.selectbox("Select column to analyze", numeric_cols, key="sel_col_stats")
    
    # Context ID for Section 2 (based on selected column)
    stats_id = f"stats_{selected_column}"
    if st.session_state.get("id_stats") != stats_id:
        state["individual"] = {}
        st.session_state.id_stats = stats_id

    if st.button("Calculate Statistics", key="btn_stats"):
        with st.spinner(f"Computing stats for {selected_column}..."):
            # Execute all small operations
            m, mc = get_mean(df, selected_column)
            med, medc = get_median(df, selected_column)
            mo, moc = get_mode(df, selected_column)
            sd, sdc = get_std(df, selected_column)
            v, vc = get_variance(df, selected_column)
            mi, mic = get_min(df, selected_column)
            ma, mac = get_max(df, selected_column)
            r, rc = get_range(df, selected_column)
            q, qc = get_quartiles(df, selected_column)
            i, ic = get_iqr(df, selected_column)
            sk, skc = get_skewness(df, selected_column)
            k, kc = get_kurtosis(df, selected_column)
            state["individual"] = {
                "mean": (m, mc), "median": (med, medc), "mode": (mo, moc),
                "std": (sd, sdc), "var": (v, vc), "min": (mi, mic), "max": (ma, mac),
                "range": (r, rc), "quartiles": (list(q), qc), "iqr": (i, ic), "skew": (sk, skc), "kurtosis": (k, kc)
            }

    if state["individual"]:
        res = state["individual"]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mean", f"{res['mean'][0]:.4f}")
            show_code(res["mean"][1])
        with c2:
            st.metric("Median", f"{res['median'][0]:.4f}")
            show_code(res["median"][1])
        with c3:
            st.metric("Mode", f"{res['mode'][0]:.4f}")
            show_code(res["mode"][1])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Standard Deviation", f"{res['std'][0]:.4f}")
            show_code(res["std"][1])
        with c2:
            st.metric("Variance", f"{res['var'][0]:.4f}")
            show_code(res["var"][1])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Minimum", f"{res['min'][0]:.4f}")
            show_code(res["min"][1])
        with c2:
            st.metric("Maximum", f"{res['max'][0]:.4f}")
            show_code(res["max"][1])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Range", f"{res['range'][0]:.4f}")
            show_code(res["range"][1])
        with c2:
            qs = res["quartiles"][0]
            st.markdown(f"Quartiles: \n- Q1: **{qs[0]}**\n- Q2: **{qs[1]}**\n- Q3: **{qs[2]}**")
            show_code(res["quartiles"][1])
        with c3:
            st.metric("IQR", f"{res['iqr'][0]:.4f}")
            show_code(res["iqr"][1])

        st.metric("Skewness", f"{res['skew'][0]:.4f}")
        show_code(res["skew"][1])
        st.metric("Kurtosis", f"{res['kurtosis'][0]:.4f}")
        show_code(res["kurtosis"][1])

    st.divider()

    # --- SECTION 3: DATA VISUALIZATION ---
    st.subheader("3. Data Visualization")
    
    # Dividimos la interfaz en dos columnas para mejor diseño
    col1, col2 = st.columns(2)
    
    with col1:
        selected_col_plot = st.selectbox("Select column to plot", numeric_cols, key="sel_col_plot")
        bins_count = st.slider("Number of bins (Histogram)", 5, 50, 20, key="bins_slider")
        
    with col2:
        # Obtenemos las columnas categóricas dinámicamente
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Checkbox condicional
        group_by_cat = st.checkbox("Group Boxplot by Category", key="chk_group_box")
        selected_cat_col = None
        
        if group_by_cat:
            if not cat_cols:
                st.warning("No categorical columns available for grouping in this dataset.")
            else:
                selected_cat_col = st.selectbox("Select categorical column", cat_cols, key="sel_cat_col")

    # Context ID for Section 3 (actualizado para incluir los nuevos estados)
    plot_id = f"plot_{selected_col_plot}_{bins_count}_{group_by_cat}_{selected_cat_col}"
    if st.session_state.get("id_plot") != plot_id:
        state["plots"] = {}
        st.session_state.id_plot = plot_id

    if st.button("Generate Visualizations", key="btn_plots"):
        with st.spinner("Generating plots..."):
            h_fig, h_code = get_histogram(df, selected_col_plot, bins=bins_count)
            
            # Condicional para generar el boxplot normal o agrupado
            if group_by_cat and selected_cat_col:
                b_fig, b_code = get_grouped_boxplot(df, selected_col_plot, selected_cat_col)
            else:
                b_fig, b_code = get_boxplot(df, selected_col_plot)
            
            state["plots"] = {
                "hist": (h_fig, h_code),
                "box": (b_fig, b_code)
            }

    if state["plots"]:
        res = state["plots"]
        
        st.markdown("#### Histogram")
        show_code(res["hist"][1])
        st.pyplot(res["hist"][0])
        
        st.markdown("#### Boxplot")
        show_code(res["box"][1])
        st.pyplot(res["box"][0])