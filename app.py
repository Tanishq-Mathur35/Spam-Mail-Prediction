import streamlit as st
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_regression.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")
st.markdown("Check whether an email is **Spam** or **Not Spam (Ham)** using ML.")

email_text = st.text_area(
    "Enter Email Content:",
    placeholder="Paste or type email text here...",
    height=180
)

if st.button("🔍 Check Email"):
    if not email_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            email_features = vectorizer.transform([email_text])
            prediction = model.predict(email_features)[0]
            probability = model.predict_proba(email_features)[0]

            if prediction == 1:
                st.success(f"✅ Not Spam (Ham)\n\nConfidence: {max(probability)*100:.2f}%")
            else:
                st.error(f"🚨 Spam Email\n\nConfidence: {max(probability)*100:.2f}%")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Built with scikit-learn & Streamlit")
