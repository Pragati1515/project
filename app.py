import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ============================
# 🔹 Safe NLTK Downloads (No punkt_tab)
# ============================
def download_nltk_resources():
    resources = [
        "punkt",  # tokenizer
        "wordnet",
        "stopwords",
        "averaged_perceptron_tagger"
    ]
    for res in resources:
        try:
            if res == "punkt":
                nltk.data.find(f"tokenizers/{res}")
            else:
                nltk.data.find(res)
        except LookupError:
            nltk.download(res)

download_nltk_resources()

# ============================
# 🔹 Preprocessing
# ============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

from nltk.tokenize import word_tokenize, sent_tokenize

def lexical_preprocess(text):
    text = str(text)
    # force standard punkt
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens if w.lower() not in stop_words and w not in string.punctuation]
    return " ".join(tokens)

def syntactic_features(text):
    text = str(text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([tag for word, tag in pos_tags])


def semantic_features(text):
    text = str(text)
    blob = TextBlob(text)
    return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"

def discourse_features(text):
    text = str(text)
    sentences = nltk.sent_tokenize(text)
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

def pragmatic_features(text):
    text = str(text)
    tokens = []
    for w in pragmatic_words:
        count = text.lower().count(w)
        tokens.extend([w] * count)
    return " ".join(tokens)

# ============================
# 🔹 Train Multiple Models
# ============================
def train_models(X_features, y):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel="linear"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    return results

# ============================
# 🔹 Streamlit UI
# ============================
st.title("📰 Fake vs Real Detection - NLP Phase-wise with ML Models")

uploaded_file = st.file_uploader("Upload your CSV file (must have 'Statement' & 'BinaryTarget')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    st.write("### Select Columns for Analysis")
    text_col = st.selectbox("Select the text column", df.columns)
    label_col = st.selectbox("Select the target column", df.columns)

    X = df[text_col].fillna("").astype(str)
    y = df[label_col].astype(str)

    st.write("### Running Phase-wise Analysis...")

    phases = {
        "Lexical & Morphological": (X.apply(lexical_preprocess), CountVectorizer()),
        "Syntactic": (X.apply(syntactic_features), CountVectorizer()),
        "Semantic": (X.apply(semantic_features), TfidfVectorizer()),
        "Discourse": (X.apply(discourse_features), CountVectorizer()),
        "Pragmatic": (X.apply(pragmatic_features), CountVectorizer())
    }

    all_results = {}

    for phase, (X_phase, vectorizer) in phases.items():
        vec = vectorizer.fit_transform(X_phase)
        res = train_models(vec, y)
        all_results[phase] = res

    st.write("### 📊 Phase-wise Accuracies")
    results_df = pd.DataFrame(all_results).T
    st.dataframe(results_df.style.format("{:.4f}"))

    st.write("### 🔎 Accuracy Comparison")
    plt.figure(figsize=(10,6))
    results_df.plot(kind="bar", figsize=(10,6))
    plt.ylabel("Accuracy")
    plt.title("Model Accuracies per NLP Phase")
    plt.xticks(rotation=30)
    st.pyplot(plt.gcf())

