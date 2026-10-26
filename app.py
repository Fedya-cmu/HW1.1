import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Social Media & Student Life Explorer", layout="wide")
st.title("Social Media & Student Life Explorer")
st.caption(
    "Upload a CSV, explore a column with a chart, then fit a linear regression "
    "with two inputs and a target. Built with a student social media survey dataset."
)

st.info(
    "**What is this model for, and what happens when it is wrong?** It gives a rough estimate of a "
    "student's mental health index from their daily social media hours and sleep, to illustrate how "
    "these habits relate, but when it is wrong it can falsely reassure or alarm someone, so it "
    "must never replace a real conversation or professional assessment."
)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("This file needs at least three numeric columns.")
        st.stop()

    # 1. Preview
    st.subheader("1. Preview")
    st.dataframe(df.head(10), hide_index=True, use_container_width=True)
    st.caption(f"{len(df):,} rows, {len(df.columns)} columns, {len(numeric_cols)} numeric.")

    # 2. Chart
    st.subheader("2. Chart")
    c1, c2 = st.columns(2)
    chart_type = c1.selectbox("Chart type", ["Histogram", "Scatter", "Line"])
    column = c2.selectbox("Column", numeric_cols)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
    elif chart_type == "Line":
        fig = px.line(df.reset_index(drop=True), y=column, title=f"{column} by row order")
        fig.update_xaxes(title="Row")
    else:
        other_options = [c for c in numeric_cols if c != column]
        other = st.selectbox("Compare against", other_options)
        fig = px.scatter(df, x=other, y=column, opacity=0.5, title=f"{column} versus {other}")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Regression
    st.subheader("3. Predict with linear regression")
    default_target = "Mental_Health_Index" if "Mental_Health_Index" in numeric_cols else numeric_cols[-1]
    target = st.selectbox("Target: what you want to predict", numeric_cols, index=numeric_cols.index(default_target))
    input_options = [c for c in numeric_cols if c != target]
    preferred = [c for c in ("Daily_Usage_Hours", "Sleep_Duration_Hours") if c in input_options]
    inputs = st.multiselect(
        "Inputs: choose exactly two",
        input_options,
        default=(preferred + input_options)[:2],
        max_selections=2,
    )

    if len(inputs) != 2:
        st.warning("Choose exactly two input columns.")
        st.stop()

    data = df[inputs + [target]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    # Same column names and order for training and prediction.
    X = data[inputs]
    y = data[target]
    model = LinearRegression().fit(X, y)

    m1, m2, m3 = st.columns(3)
    m1.metric("R2 on this data", f"{model.score(X, y):.3f}")
    m2.metric(f"Effect of {inputs[0]}", f"{model.coef_[0]:,.4g}")
    m3.metric(f"Effect of {inputs[1]}", f"{model.coef_[1]:,.4g}")
    st.caption(f"Trained on {len(data):,} rows with no missing values. Each effect is the change in the target per one-unit increase, holding the other input fixed.")

    st.markdown("**Try it:** type values for the inputs.")
    v1, v2 = st.columns(2)
    val1 = v1.number_input(inputs[0], value=float(data[inputs[0]].median()))
    val2 = v2.number_input(inputs[1], value=float(data[inputs[1]].median()))

    new_row = pd.DataFrame([[val1, val2]], columns=inputs)
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {target}", f"{prediction:,.2f}")

    for name, val in zip(inputs, (val1, val2)):
        low, high = data[name].min(), data[name].max()
        if val < low or val > high:
            st.warning(
                f"{name} = {val:,.2f} is outside the range the model learned from "
                f"({low:,.2f} to {high:,.2f}). Treat the prediction with caution."
            )
else:
    st.info("Upload a CSV to get started. Try the sample file `social_media.csv` from the repository.")
