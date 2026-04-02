import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Movie Rating Predictor", layout="centered")

st.title("🎬 Movie Rating Predictor")
st.write("Select movie details")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load encoders
le_genre = pickle.load(open("genre.pkl", "rb"))
le_director = pickle.load(open("director.pkl", "rb"))
le_actor1 = pickle.load(open("actor1.pkl", "rb"))
le_actor2 = pickle.load(open("actor2.pkl", "rb"))
le_actor3 = pickle.load(open("actor3.pkl", "rb"))

# Dropdowns (REAL NAMES)
genre_name = st.selectbox("Select Genre", le_genre.classes_)
director_name = st.selectbox("Select Director", le_director.classes_)
actor1_name = st.selectbox("Select Actor 1", le_actor1.classes_)
actor2_name = st.selectbox("Select Actor 2", le_actor2.classes_)
actor3_name = st.selectbox("Select Actor 3", le_actor3.classes_)

# Numeric inputs
duration = st.slider("Duration (minutes)", 60, 200)
year = st.slider("Year", 1950, 2025)

# Convert names → encoded values
genre = le_genre.transform([genre_name])[0]
director = le_director.transform([director_name])[0]
actor1 = le_actor1.transform([actor1_name])[0]
actor2 = le_actor2.transform([actor2_name])[0]
actor3 = le_actor3.transform([actor3_name])[0]

# Predict
if st.button("Predict Rating"):
    input_data = np.array([[genre, director, actor1, actor2, actor3, duration, year]])
    prediction = model.predict(input_data)

    st.success(f"⭐ Predicted Rating: {prediction[0]:.2f}")