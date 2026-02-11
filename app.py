import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.set_page_config(page_title="Python package – Normality Tests")

st.title("Python package – Inferential Statistics")
st.subheader("Normality tests")

# --------------------------
# 1. Load dataset
# --------------------------
uploaded_file = st.file_uploader("Load dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the dataset:")
    st.dataframe(df.head())

    # --------------------------
    # 2. Select column
    # --------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available.")
        st.stop()

    col = st.selectbox(
        "Select numeric variable",
        numeric_cols
    )

    data = df[col].dropna().values

    # --------------------------
    # 3. Test for normality menu
    # --------------------------
    st.markdown("### Test for normality")

    test = st.selectbox(
        "Select test",
        [
            "Shapiro–Wilk",
            "D’Agostino–Pearson"
        ]
    )

    # --------------------------
    # 4. Run test + show code
    # --------------------------
    if st.button("Run test"):

        st.markdown("### Generated Python code")

        if test == "Shapiro–Wilk":
            code = f"""from scipy import stats

stat, p = stats.shapiro(data)
print(f"Shapiro–Wilk Test: statistic={{stat:.4f}}, p-value={{p:.4f}}")
"""
            stat, p = stats.shapiro(data)

        else:  # D’Agostino–Pearson
            code = f"""from scipy import stats

stat, p = stats.normaltest(data)
print(f"D’Agostino–Pearson Test: statistic={{stat:.4f}}, p-value={{p:.4f}}")
"""
            stat, p = stats.normaltest(data)

        # Show code block
        st.code(code, language="python")

        # --------------------------
        # 5. Results
        # --------------------------
        st.markdown("### Results")
        st.write(f"**Statistic:** {stat:.4f}")
        st.write(f"**p-value:** {p:.4f}")

        alpha = 0.05
        if p < alpha:
            st.error("Null hypothesis rejected: data is NOT normally distributed.")
        else:
            st.success("Fail to reject null hypothesis: data may be normally distributed.")
