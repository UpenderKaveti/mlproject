import streamlit as st
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler

# Set page title
st.title("Student Performance Input Form")

# Input for categorical columns
gender = st.selectbox("Gender", options=["male", "female"])
race_ethnicity = st.selectbox("Race/Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    options=[
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ]
)
lunch = st.selectbox("Lunch", options=["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", options=["none", "completed"])

# Input for numerical columns
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=0)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=0)

# Submit button
submit = st.button("Submit")

# Output user input
if submit:
    st.write("### Submitted Data")
    st.write(f"Gender: {gender}")
    st.write(f"Race/Ethnicity: {race_ethnicity}")
    st.write(f"Parental Level of Education: {parental_level_of_education}")
    st.write(f"Lunch: {lunch}")
    st.write(f"Test Preparation Course: {test_preparation_course}")
    st.write(f"Reading Score: {reading_score}")
    st.write(f"Writing Score: {writing_score}")

    # Prepare data for further processing
    data = {
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course,
        "reading_score": reading_score,
        "writing_score": writing_score,
    }

    obj = CustomData(**data)
    pred_df = obj.get_data_as_data_frame()

    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)
    st.write("Prediction value: ", results)