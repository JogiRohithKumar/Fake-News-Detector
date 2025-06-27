import streamlit as st
import joblib

model = joblib.load('../model/fake_news_model.pkl')
vectorizer = joblib.load('../model/tfidf_vectorizer.pkl')

st.title("📰 Fake News Detector")
user_input = st.text_area("Paste the news article here:")

if st.button("Predict"):
    cleaned_input = user_input.lower()
    vectorized = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        st.success("✅ This news is **Real**.")
    else:
        st.error("❌ This news is **Fake**.")
