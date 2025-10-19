import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# --- Hardcoded English Stopwords ---
STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves',
    'he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their',
    'theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was',
    'were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and',
    'but','if','or','because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out','on','off',
    'over','under','again','further','then','once','here','there','when','where','why','how','all','any',
    'both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
])

# Initialize stemmer and tokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# --- Text Preprocessing Function ---
def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)

    filtered = [ps.stem(word) for word in text if word.isalnum() and word not in STOPWORDS and word not in string.punctuation]
    return " ".join(filtered)

# --- Load model and vectorizer ---
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- Page Configuration ---
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body { background-color: #f5f7fa; }
    .main-title { text-align: center; font-size: 42px; color: #2c3e50; font-weight: 700; margin-bottom: 0; }
    .sub-title { text-align: center; color: #7f8c8d; font-size: 18px; margin-bottom: 30px; }
    .stButton button { background-color: #3498db; color: white; border-radius: 8px; padding: 0.6em 1.2em; font-weight: 600; }
    .stButton button:hover { background-color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 class='main-title'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Instantly detect whether an email message is Spam or Not Spam ⚡</p>", unsafe_allow_html=True)

# --- User Input ---
input_sms = st.text_area("✉️ Enter the message:", height=120, placeholder="Type or paste your message here...")

# --- Predict Button with Spinner ---
if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before prediction.")
    else:
        with st.spinner('🔎 Analyzing message...'):
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
        # 4. Display Result
        if result == 1:
            st.markdown("### 🚨 <span style='color:red;'>This message is likely **Spam**!</span>", unsafe_allow_html=True)
        else:
            st.markdown("### ✅ <span style='color:green;'>This message looks **Safe (Not Spam)**.</span>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<br><hr><center>🔒 Built with Streamlit | by Shaik Fahad Jahangir</center>", unsafe_allow_html=True)
