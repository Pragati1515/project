import streamlit as st
import pandas as pd
import numpy as np
import string
import re
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

# ----------------------------
# Stop words & pragmatic words
# ----------------------------
stop_words = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were','be',
    'been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for',
    'with','about','against','between','into','through','during','before','after',
    'above','below','to','from','up','down','in','out','on','off','over','under',
    'again','further','then','once','here','there','when','where','why','how','all',
    'any','both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','s','t','can','will','just','don',
    'should','now'
])
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ----------------------------
# Lexical preprocessing without NLTK
# ----------------------------
def lexical_preprocess(text):
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)  # simple regex tokenizer
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    return " ".join(tokens)

# ----------------------------
# Syntactic Features (TextBlob POS)
# ----------------------------
def syntactic_features(text):
    text = str(text)
    blob = TextBlob(text)
    return " ".join([tag for word, tag in blob.tags])

# ----------------------------
# Semantic Features
# ----------------------------
def semantic_features(text):
    blob = TextBlob(str(text))
    return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"

# ----------------------------
# Discourse Features
# ----------------------------
def discourse_features(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s for s in sentences if s.strip()]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

# ----------------------------
# Pragmatic Features
# ----------------------------
def pragmatic_features(text):
    tokens = []
    text_lower = str(text).lower()
    for w in pragmatic_words:
        count = text_lower.count(w)
        tokens.extend([w]*count)
    return " ".join(tokens)

# ----------------------------
# Train multiple ML models
# ----------------------------
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

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📰 Fake vs Real Detection - NLP Phase-wise with ML Models")

uploaded_file = st.file_uploader("Upload your CSV file (must have 'Statement' & 'BinaryTarget')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    st.write("### Select Columns for Analysis")
    text_col = st.selectbox("Text column", df.columns)
    label_col = st.selectbox("Target column", df.columns)

    X = df[text_col].fillna("").astype(str)
    y = df[label_col]

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
    results_df.plot(kind="bar", figsize=(10,6))
    plt.ylabel("Accuracy")
    plt.title("Model Accuracies per NLP Phase")
    plt.xticks(rotation=30)
    st.pyplot(plt.gcf())
