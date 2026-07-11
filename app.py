# import streamlit as st
# import pickle
# import string

# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize

# ps = PorterStemmer()

# # Load TF-IDF Vectorizer and Model
# tfidf = pickle.load(open("models/vectorizer.pkl", "rb"))
# model = pickle.load(open("models/model.pkl", "rb"))


# # Text Preprocessing Function
# def transform_text(text):

#     # Lowercase
#     text = text.lower()

#     # Tokenization
#     text = word_tokenize(text)

#     # Remove special characters
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     # Remove stopwords and punctuation
#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words("english") and i not in string.punctuation:
#             y.append(i)

#     # Stemming
#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)


# # Streamlit UI
# st.title("📩 Bilingual Spam Detector")

# input_sms = st.text_area("Enter the message")

# if st.button("Predict"):

#     # 1. Preprocess
#     transformed_sms = transform_text(input_sms)

#     # 2. Vectorize
#     vector_input = tfidf.transform([transformed_sms]).toarray()

#     # 3. Predict
#     result = model.predict(vector_input)[0]

#     # 4. Show Result
#     if result == 1:
#         st.error("🚨 Spam Message")
#     else:
#         st.success("✅ Ham Message")



import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

# Load TF-IDF Vectorizer and Model
tfidf = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))


import re

def transform_text(text):

    # Hindi/Marathi असेल तर preprocessing करू नको
    if re.search(r'[\u0900-\u097F]', text):
        return text.strip()

    # English preprocessing
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.set_page_config(
    page_title="Bilingual Spam Detector",
    page_icon="📩",
    layout="centered"
)

with st.sidebar:
    st.title("📌 About")

    st.write("""
    ### Bilingual Spam Detector

    This project detects whether a message is:

    ✅ Ham (Normal Message)

    🚨 Spam Message

    ---
    """)

    st.write("### 🛠 Tech Stack")
    st.write("""
    - Python
    - Scikit-Learn
    - TF-IDF
    - Streamlit
    """)

    st.write("---")
    st.success("Model Accuracy: 96.44%")
    
st.title("📩 Bilingual Spam Detector")

st.markdown("""
### 🤖 AI-Powered Spam Detection System

Detect **Spam** and **Ham** messages using Machine Learning.

**Supported Languages (Coming Soon):**
- 🇬🇧 English ✅
- 🇮🇳 Hindi 🚧
- 🇮🇳 Marathi 🚧
""")

st.divider()

input_sms = st.text_area(
    "✍️ Enter your message",
    height=180,
    placeholder="Type your message here..."
)
if st.button("🔍 Predict", use_container_width=True):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    #     # Debug
    # st.write("Original:", input_sms)
    # st.write("Transformed:", transformed_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray()

    # 3. Predict
    result = model.predict(vector_input)[0]

    # Debug
    # st.write("Prediction:", result)


    # 4. Show Result
    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Ham Message")