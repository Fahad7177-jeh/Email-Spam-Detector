import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
# Initialize stemmer
ps = PorterStemmer()

# --- Text Preprocessing Function ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# --- Load model and vectorizer ---
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- Page Configuration ---
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main-title {
        text-align: center;
        font-size: 42px;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #7f8c8d;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 class='main-title'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Instantly detect whether an email message is Spam or Not Spam ⚡</p>", unsafe_allow_html=True)

# --- User Input ---
input_sms = st.text_area("✉️ Enter the message:", height=120, placeholder="Type or paste your message here...")

# --- Predict Button ---
if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before prediction.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.markdown("### 🚨 <span style='color:red;'>This message is likely **Spam**!</span>", unsafe_allow_html=True)
        else:
            st.markdown("### ✅ <span style='color:green;'>This message looks **Safe (Not Spam)**.</span>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<br><hr><center>🔒 Built with Streamlit | by Shaik Fahad Jahangir</center>", unsafe_allow_html=True)
