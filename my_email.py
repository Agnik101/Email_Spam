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

/* Style for the sidebar with increased width */
[data-testid="stSidebar"] {
    background-color: #ff4b4b;
    padding: 1.5rem;
    color: white;
    width: 250px;  /* Increase the sidebar width */
    min-width: 300px; /* Ensure it doesn't shrink too much */
    transition: all 0.3s ease;
}

/* Responsive adjustments for smaller screens */
@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        padding: 1rem;
        font-size: 0.9rem;
        width: auto;
    }
    .team-member {
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
}

/* App background and font */
.stApp {
    background: linear-gradient(to bottom, #ffcccb, #ff4b4b);
    font-family: 'Arial', sans-serif;
    transition: all 0.3s ease;
}

/* Highlight spam words */
.highlight {
    background-color: #ff0000;
    color: white;
    padding: 0.1em;
    border-radius: 3px;
    font-weight: bold;
}

/* Tip boxes */
.tip-box {
    background-color:#FFFFE0;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

/* Team member box style */
.team-member {
    background-color:#FFFFE0;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Initialize NLTK
try:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

ps = PorterStemmer()

# Comprehensive list of spam words/phrases
SPAM_WORDS = [
    r'(?i)\bsex\b', r'(?i)\bfree\s*sex\b', r'(?i)\bgay\b',
    r'(?i)\bmotherfucker\b', r'(?i)\bfucker\b', r'(?i)\bdate\s*me\b',
    r'(?i)\bviagra\b', r'(?i)\bporn\b', r'(?i)\bnude\b',
    r'(?i)\bhot\s*girls\b', r'(?i)\bsingle\s*now\b',
    r'(?i)\bmeet\s*girls\b', r'(?i)\bcasino\b', r'(?i)\bcredit\b',
    r'(?i)\bloan\b' , r'(?i)\badult\s*dating\b',
    r'(?i)\bsexy\b', r'(?i)\bhot\s*singles\b', r'(?i)\bhot\b',
    r'(?i)\rape\b' , r'(?i)\blottery\b', r'(?i)\bfree\b' , 
    r'(?i)\bsubscribe\b' , r'(?i)\bdigital\s*arrest\b'
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

# Main app content
def main():

    st.title("📧 Advanced Email Spam Classifier")
    

    # Input area
    input_email = st.text_area("Enter email text to analyze:", height=150)

    if st.button('🔍 Analyze Email'):
        if not input_email.strip():
            st.warning("Please enter some text to analyze")
        else:
            # Check for spam words
            highlighted_text, found_words = detect_spam_words(input_email)
            
            if found_words:
                st.error("🚨 This is classified as SPAM by our ML model ")
                with st.expander("View details", expanded=True):
                    st.markdown("**Detected spam words:**")
                    for word in found_words:
                        st.markdown(f"- {word}")
                    
                    st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                # If no spam words, use ML model
                try:
                    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
                    model = pickle.load(open('model.pkl', 'rb'))
                    
                    processed_text = transform_text(input_email)
                    vector = tfidf.transform([processed_text])
                    prediction = model.predict(vector)[0]
                    
                    if prediction == 1:
                        st.error("🚨 This is classified as SPAM by our ML model")
                    else:
                        st.success("✅ This is HAM (Not Spam)")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    # Visualization section
    if st.button('📊 Show Spam/Ham Distribution'):
        try:
            df = pd.read_csv('spam22.csv', encoding='latin-1')
            fig, ax = plt.subplots()
            df['v1'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                        colors=['#2d6a4f','#ff4d6d'], ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error showing distribution: {e}")

# Sidebar content
def sidebar_content():
    with st.sidebar:
        st.header("About This Tool")
        
        # Educational section about spam
        with st.expander("ℹ️ Understanding Spam", expanded=True):
            st.write("""
            **Spam** refers to unsolicited messages, typically sent in bulk, that may contain:
            - Commercial advertisements
            - Fraudulent schemes
            - Malicious links or attachments
            - Inappropriate content
            
            Our classifier combines keyword detection and machine learning to identify these unwanted messages.
            """)
        
        # Tips section
        with st.expander("🛡️ Spam Prevention Tips"):
            st.write("""
            - ✅ **Use spam filters** provided by your email service
            - ✅ **Never reply** to suspicious emails
            - ✅ **Check sender addresses** carefully
            - ✅ **Avoid clicking links** in unsolicited emails
            - ✅ **Keep software updated** to prevent vulnerabilities
            - ✅ **Report spam** to your email provider
            """)
        
        # Team section
        with st.expander("👥 Our Team"):
            team_col1, team_col2 = st.columns(2)
            
            with team_col1:
                st.markdown("""
                **Agnik Gupta**  
                *Data Scientist*  
                Model development
                """)
                
                st.markdown("""
                **Arunava Ghosh**  
                *Data Scientist*  
                Algorithm optimization
                """)
                st.markdown("""
                **Souhardya Nandy**  
                *Developer*  
                App interface
                """)
                
            with team_col2:
                
                
                st.markdown("""
                **Bitan Banerjee**  
                *Data Analyst*  
                Data processing
                """)

        
                st.markdown("""
                **Debrik Debnath**  
                *Developer*  
                App interface
                """)
        
        # Contact section
        with st.expander("📨 Contact Us"):
            st.write("""
            **Email:** contact@spamfilter.com  
            **Support:** 24/7 via help portal  
            **HQ:** Data Security Tower, Tech City
            """)

# Run the app
if __name__ == "__main__":
    main()
    sidebar_content()
