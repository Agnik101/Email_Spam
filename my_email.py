import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import re

# Set app styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #ffcccb, #ff4b4b);
}
[data-testid="stSidebar"] {
    background-color: #ff4b4b;
    padding: 1rem;
}
.highlight {
    background-color: #ff0000;
    color: white;
    padding: 0.1em;
    border-radius: 3px;
    font-weight: bold;
}
.spam-word {
    color: #ff0000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

# Comprehensive list of spam words/phrases with pattern matching
SPAM_WORDS = [
    r'(?i)\bsex\b', r'(?i)\bfree\s*sex\b', r'(?i)\bgay\b', 
    r'(?i)\bmotherfucker\b', r'(?i)\bfucker\b', r'(?i)\bdate\s*me\b',
    r'(?i)\bviagra\b', r'(?i)\bporn\b', r'(?i)\bnude\b', 
    r'(?i)\bhot\s*girls\b', r'(?i)\bsingle\s*now\b',
    r'(?i)\bmeet\s*girls\b', r'(?i)\bcasino\b', r'(?i)\bcredit\b',
    r'(?i)\bloan\b', r'(?i)\bclick\s*here\b', r'(?i)\bwinner\b',
    r'(?i)\bprize\b', r'(?i)\bmoney\b', r'(?i)\badult\s*dating\b',
    r'(?i)\bsexy\b', r'(?i)\bhot\s*singles\b', r'(?i)\bhot\b'
]

def detect_spam_words(text):
    found_words = set()
    highlighted_text = text
    
    for pattern in SPAM_WORDS:
        for match in re.finditer(pattern, text):
            matched_word = match.group()
            found_words.add(matched_word.lower())
            start, end = match.span()
            highlighted_text = (highlighted_text[:start] + 
                              f'<span class="highlight">{matched_word}</span>' + 
                              highlighted_text[end:])
    
    return highlighted_text, sorted(found_words)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english')]
    text = [word for word in text if word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Main app
st.title("📧 Email Spam Classifier")
input_email = st.text_area("Enter your email text:")

if st.button('🔍 Predict'):
    if not input_email.strip():
        st.warning("Please enter some text to classify")
    else:
        # First check for spam words
        highlighted_text, found_words = detect_spam_words(input_email)
        
        if found_words:
            st.error("🚨 This is SPAM (contains spam words)")
            with st.expander("View details", expanded=True):
                st.markdown("**Detected spam words:**")
                for word in found_words:
                    st.markdown(f"- <span class='spam-word'>{word}</span>", unsafe_allow_html=True)
                st.markdown("**Text with highlighted spam words:**")
                st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            processed_text = transform_text(input_email)
            vector = tfidf.transform([processed_text])
            prediction = model.predict(vector)[0]
            
            if prediction == 1:
                st.error("🚨 This is SPAM (ML model prediction)")
            else:
                st.success("✅ This is HAM (Not Spam)")

if st.button('📊 Show Distribution'):
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        fig, ax = plt.subplots()
        df['v1'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                    colors=['#2d6a4f','#ff4d6d'], ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error showing distribution: {e}")


# Sidebar remains unchanged from working version

# Red sidebar with expandable sections
with st.sidebar:
    # Team section
    with st.expander("👥 Meet Our Team", expanded=False):
        team_members = [
            {"name": "Agnik Gupta", "designation": "Data Scientist", "contribution": "Developed the NLP model"},
            {"name": "Arunava Ghosh", "designation": "ML Engineer", "contribution": "Optimized the algorithm"},
            {"name": "Souhardya Nandy", "designation": "Developer", "contribution": "Built the interface"},
            {"name": "Bitan Bannerjee", "designation": "Data Analyst", "contribution": "Refined the dataset"}
        ]
        
        for member in team_members:
            with st.expander(member["name"], expanded=False):
                st.write(f"**Designation:** {member['designation']}")
                st.write(f"**Contribution:** {member['contribution']}")

    # About section
    with st.expander("ℹ️ About This App", expanded=False):
        st.write("""
        Features:
        - NLP Text Processing
        - Spam Word Detection
        - Naive Bayes Classification
        - Visual Analytics
        """)
    
    # Contact section
    with st.expander("📨 Contact Us", expanded=False):
        st.write("**Email:** spam.filter@example.com")
        st.write("**Support Hours:** 9AM-5PM EST")
