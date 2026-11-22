import streamlit as st
import joblib
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Load the model
model = joblib.load('predictor.pkl')

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")
st.title("😊 Sentiment Analysis")
st.write("Enter text to analyze its sentiment.")

# Text input
user_text = st.text_area("Input Text")

if st.button("Predict Sentiment", type="primary"):
    if user_text:
        # Preprocess
        cleaned_text = clean_text(user_text)
        
        # Predict
        prediction = model.predict([cleaned_text])[0]
        
        # Display result
        st.subheader("Result:")
        if prediction == 'positive':
            st.success(f"Sentiment: {prediction.capitalize()}")
        elif prediction == 'negative':
            st.error(f"Sentiment: {prediction.capitalize()}")
        else:
            st.info(f"Sentiment: {prediction.capitalize()}")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Sentiment Analysis App</p>
</div>
""", unsafe_allow_html=True)

