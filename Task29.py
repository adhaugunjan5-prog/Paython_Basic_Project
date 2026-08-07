import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
import joblib

from datasets import load_dataset
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# Download NLTK Resources
# -----------------------------
nltk.download("stopwords")

st.set_page_config(
    page_title="Task29 - NLP Vectorization",
    layout="wide"
)

st.title("😊 Task29 - NLP Vectorization using Streamlit")

# =====================================================
# Load Dataset
# =====================================================

@st.cache_data
def load_data():

    dataset = load_dataset("dair-ai/emotion")

    df = pd.DataFrame(dataset["train"])

    labels = dataset["train"].features["label"].names

    df.rename(columns={"label":"encoded_label"}, inplace=True)

    df["emotion"] = df["encoded_label"].apply(
        lambda x: labels[x]
    )

    stop_words = set(stopwords.words("english"))

    def clean_text(text):

        text = text.lower()

        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

        text = re.sub(r"\d+", "", text)

        text = text.encode(
            "ascii",
            "ignore"
        ).decode()

        words = text.split()

        words = [
            word
            for word in words
            if word not in stop_words
        ]

        return " ".join(words)

    df["cleaned_text"] = df["text"].apply(clean_text)

    return df


df = load_data()

st.success("Dataset Loaded Successfully")

# =====================================================
# Q1
# =====================================================

st.header("Q1. Data Preparation")

X = df["cleaned_text"]

y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

st.write("### Shapes")

st.write("X_train :", X_train.shape)

st.write("X_test :", X_test.shape)

st.write("y_train :", y_train.shape)

st.write("y_test :", y_test.shape)

# =====================================================
# Q2
# =====================================================

st.header("Q2. Bag of Words")

bow = CountVectorizer()

X_train_bow = bow.fit_transform(X_train)

X_test_bow = bow.transform(X_test)

st.write("Training Matrix Shape")

st.write(X_train_bow.shape)

st.write("Testing Matrix Shape")

st.write(X_test_bow.shape)

st.write("First 20 Vocabulary Words")

st.write(bow.get_feature_names_out()[:20])

# =====================================================
# Q3
# =====================================================

st.header("Q3. Multinomial Naive Bayes (BoW)")

bow_model = MultinomialNB()

bow_model.fit(
    X_train_bow,
    y_train
)

bow_prediction = bow_model.predict(
    X_test_bow
)

bow_accuracy = accuracy_score(
    y_test,
    bow_prediction
)

st.success(
    f"Bag of Words Accuracy : {bow_accuracy:.4f}"
)

# =====================================================
# Q4
# =====================================================

st.header("Q4. Vocabulary Analysis")

vocabulary = bow.get_feature_names_out()

st.write("Total Vocabulary Size")

st.write(len(vocabulary))

st.write("15 Sample Words")

st.write(vocabulary[:15])

sample_text = X_train.iloc[0]

sample_vector = bow.transform([sample_text])

st.write("Sample Document")

st.write(sample_text)

st.write("Bag of Words Vector")

st.write(sample_vector.toarray())

# =====================================================
# Q5. Bag of Words with N-Grams
# =====================================================

from sklearn.feature_extraction.text import TfidfVectorizer

st.header("Q5. N-Grams with Bag of Words")

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))

X_train_bigram = bigram_vectorizer.fit_transform(X_train)

X_test_bigram = bigram_vectorizer.transform(X_test)

st.write("### Bigram Feature Matrix Shape")
st.write("Training:", X_train_bigram.shape)
st.write("Testing :", X_test_bigram.shape)

st.write("### First 20 Bigram Features")
st.write(bigram_vectorizer.get_feature_names_out()[:20])

# =====================================================
# Q6. Bigram Model Training
# =====================================================

st.header("Q6. MultinomialNB using Bigrams")

bigram_model = MultinomialNB()

bigram_model.fit(X_train_bigram, y_train)

bigram_prediction = bigram_model.predict(X_test_bigram)

bigram_accuracy = accuracy_score(
    y_test,
    bigram_prediction
)

st.success(
    f"Bigram Accuracy : {bigram_accuracy:.4f}"
)

st.write("### Comparison")

comparison = pd.DataFrame({

    "Model":[
        "Bag of Words",
        "Bag of Words + Bigrams"
    ],

    "Accuracy":[
        bow_accuracy,
        bigram_accuracy
    ]

})

st.dataframe(comparison)

# =====================================================
# Q7. TF-IDF
# =====================================================

st.header("Q7. TF-IDF Vectorization")

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

st.write("Training Shape")

st.write(X_train_tfidf.shape)

st.write("Testing Shape")

st.write(X_test_tfidf.shape)

st.write("First 15 TF-IDF Features")

st.write(
    tfidf.get_feature_names_out()[:15]
)

# =====================================================
# Q8. TF-IDF Model
# =====================================================

st.header("Q8. TF-IDF + MultinomialNB")

tfidf_model = MultinomialNB()

tfidf_model.fit(
    X_train_tfidf,
    y_train
)

tfidf_prediction = tfidf_model.predict(
    X_test_tfidf
)

tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_prediction
)

st.success(
    f"TF-IDF Accuracy : {tfidf_accuracy:.4f}"
)

# =====================================================
# Q9. Comparison of Vectorizers
# =====================================================

st.header("Q9. Comparison of Vectorizers")

comparison_df = pd.DataFrame({
    "Vectorizer": [
        "Bag of Words (Unigram)",
        "Bag of Words (Unigram + Bigram)",
        "TF-IDF"
    ],
    "Accuracy": [
        bow_accuracy,
        bigram_accuracy,
        tfidf_accuracy
    ]
})

st.dataframe(comparison_df, use_container_width=True)

# ----------------------------
# Accuracy Chart
# ----------------------------
fig, ax = plt.subplots(figsize=(7,4))

ax.bar(
    comparison_df["Vectorizer"],
    comparison_df["Accuracy"]
)

ax.set_title("Accuracy Comparison")
ax.set_xlabel("Vectorizer")
ax.set_ylabel("Accuracy")

plt.xticks(rotation=15)

st.pyplot(fig)

# ----------------------------
# Best Method
# ----------------------------
best_accuracy = max(
    bow_accuracy,
    bigram_accuracy,
    tfidf_accuracy
)

if best_accuracy == bow_accuracy:
    best_method = "Bag of Words (Unigram)"
    best_model = bow_model
    best_vectorizer = bow

elif best_accuracy == bigram_accuracy:
    best_method = "Bag of Words (Unigram + Bigram)"
    best_model = bigram_model
    best_vectorizer = bigram_vectorizer

else:
    best_method = "TF-IDF"
    best_model = tfidf_model
    best_vectorizer = tfidf

st.success(f"Best Method : {best_method}")
st.success(f"Accuracy : {best_accuracy:.4f}")

st.subheader("Observation")

st.info(f"""
Among the three vectorization techniques,
**{best_method}** achieved the highest accuracy.

Bag of Words converts text into word counts,
while Bigrams capture two-word combinations.

TF-IDF reduces the importance of very common words
and often produces better performance because it
focuses on more informative terms.
""")

# =====================================================
# Q10. Mini Project
# =====================================================

st.header("Q10. Complete Vectorization Pipeline")

# Save Best Model
joblib.dump(
    best_model,
    "best_model.pkl"
)

# Save Best Vectorizer
joblib.dump(
    best_vectorizer,
    "best_vectorizer.pkl"
)

st.success("best_model.pkl Saved Successfully")
st.success("best_vectorizer.pkl Saved Successfully")

st.write("### Files Created")

st.code("""
best_model.pkl
best_vectorizer.pkl
""")

st.balloons()

st.success("""
🎉 Congratulations!

Task29 Completed Successfully.

✔ Data Preparation

✔ Bag of Words

✔ Naive Bayes

✔ Vocabulary Analysis

✔ N-Grams

✔ Bigram Model

✔ TF-IDF

✔ TF-IDF Model

✔ Accuracy Comparison

✔ Best Model Saved

✔ Best Vectorizer Saved
""")