import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, pull

# Title of the app
st.title("AutoML with PyCaret")

# Load the dataset
data = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if data is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(data)
    st.write("Dataset Loaded Successfully:")
    st.write(df)

    # Select the target variable
    target = st.selectbox("Select the target variable", df.columns)

    # Check the distribution of the target variable
    class_counts = df[target].value_counts()
    st.write("Class distribution:")
    st.write(class_counts)

    # Filter out classes with fewer than 2 instances
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df[target].isin(valid_classes)]

    # Show the new class distribution
    st.write("New class distribution after filtering:")
    st.write(df[target].value_counts())

    # Train Model button
    if st.button("Train Model"):
        if df[target].nunique() < 2:
            st.write("Error: At least two instances are required for each class in the target variable.")
        else:
            st.write("Training model... This may take a while.")

            # Set up the PyCaret environment
            clf = setup(data=df, target=target, html=False)

            # Train the model using compare_models
            best_model = compare_models()

            # Show the best model
            st.write("Model trained successfully!")
            st.write("Best Model:")
            st.write(best_model)

            # Show model results
            st.write("Model Performance:")
            model_performance = pull()
            st.write(model_performance)
