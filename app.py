import streamlit as st
import joblib

model = joblib.load("logistic_regression.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Spam Mail Checker")

st.title("Spam Mail Checker")
st.write("Enter an email message below to check whether it is spam or not.")

email_text = st.text_area(
    "Email Text",
    placeholder="Type or paste the email content here",
    height=150
)

if st.button("Check"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        email_features = vectorizer.transform([email_text])
        result = model.predict(email_features)

        if result[0] == 1:
            st.success("This email is not spam.")
        else:
            st.error("This email is spam.")
