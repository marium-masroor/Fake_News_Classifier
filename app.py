import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

# Function to load models with caching
@st.cache(allow_output_mutation=True)
def load_objects():
    try:
        with open('vector.pkl', 'rb') as f:
            vector = pickle.load(f)
    except FileNotFoundError:
        st.error("The file 'vector.pkl' was not found. Please ensure it exists in the correct directory.")
        vector = None

    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("The file 'best_model.pkl' was not found. Please ensure it exists in the correct directory.")
        model = None

    return vector, model

# Load the vectorizer and model
vector_form, load_model = load_objects()

# Define the stemming function
def stemming(content):
    # Remove non-alphabetic characters
    con = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase
    con = con.lower()
    # Split the text into words
    con = con.split()
    # Stem words and remove stopwords
    stop_words = set(stopwords.words('english'))
    con = [port_stem.stem(word) for word in con if word not in stop_words]
    # Join the stemmed words back into a single string
    con = ' '.join(con)
    return con

# Define the prediction function
def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

# Streamlit App Layout
def main():
    st.title('Fake News Classification App')
    st.subheader("Input the News Content Below")
    
    # Text area for user input
    sentence = st.text_area("Enter your News Content Here", "Some News", height=200)
    
    # Predict button
    if st.button("Predict"):
        if sentence.strip() == "":
            st.warning("Please enter some news content to analyze.")
        elif vector_form is None or load_model is None:
            st.error("Model components are not loaded properly.")
        else:
            try:
                prediction_class = fake_news(sentence)
                if prediction_class == [0]:
                    st.success("Real News")
                elif prediction_class == [1]:
                    st.warning("Fake News")
                else:
                    st.info("Unable to classify the news content.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
