import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import string
import re
import nltk

from datasets import load_dataset
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

st.set_page_config(
    page_title="Emotion NLP Preprocessing",
    layout="wide"
)

st.title("😊 Emotion Dataset NLP Preprocessing")
st.write("## Q1 to Q10")


@st.cache_data
def load_data():

    dataset = load_dataset("dair-ai/emotion")

    train = dataset["train"]

    df = pd.DataFrame(train)

    labels = dataset["train"].features["label"].names

    df.rename(
        columns={
            "label": "encoded_label"
        },
        inplace=True
    )

    df["emotions"] = df["encoded_label"].apply(
        lambda x: labels[x]
    )

    return df, labels


try:

    df, label_names = load_data()

    st.success("Dataset Loaded Successfully!")

except Exception as e:

    st.error(e)

    st.stop()

# Q1

st.header("Q1. Loading Dataset")

st.subheader("First 10 Rows")

st.dataframe(df.head(10), use_container_width=True)

st.subheader("Shape")

st.write(df.shape)

st.subheader("Missing Values")

st.write(df.isnull().sum())

# Q2

st.header("Q2. Exploring Target Labels")

st.subheader("Unique Labels")

st.write(df["emotions"].unique())

mapping = {
    emotion: i
    for i, emotion in enumerate(label_names)
}

st.subheader("Label Mapping")

st.write(mapping)

st.dataframe(
    df[
        [
            "text",
            "emotions",
            "encoded_label"
        ]
    ].head(10)
)

# Q3

st.header("Q3. Lowercasing")

before = df["text"].head(5)

df["lower_text"] = df["text"].str.lower()

after = df["lower_text"].head(5)

compare = pd.DataFrame({

    "Before": before,

    "After": after

})

st.dataframe(compare)

st.info("""
Lowercasing converts all words into one format.

Happy and happy become identical.

This reduces vocabulary size and improves NLP performance.
""")


# Q4

st.header("Q4. Remove Punctuation")

def remove_punctuation(text):

    return text.translate(

        str.maketrans(

            "",

            "",

            string.punctuation

        )

    )

df["punct_removed"] = df["lower_text"].apply(
    remove_punctuation
)

compare = pd.DataFrame({

    "Before": df["lower_text"].head(),

    "After": df["punct_removed"].head()

})

st.dataframe(compare)

# Q5


st.header("Q5. Remove Numbers")

def remove_numbers(text):
    return re.sub(r"\d+", "", text)

df["number_removed"] = df["punct_removed"].apply(remove_numbers)

compare = pd.DataFrame({
    "Before": df["punct_removed"].head(5),
    "After": df["number_removed"].head(5)
})

st.dataframe(compare, use_container_width=True)


# Q6

st.header("Q6. Remove Emojis & Special Characters")

def remove_emojis(text):
    return text.encode("ascii", "ignore").decode()

df["emoji_removed"] = df["number_removed"].apply(remove_emojis)

st.write("### Cleaned Examples")
st.dataframe(
    df[["number_removed", "emoji_removed"]].head(5),
    use_container_width=True
)


# Q7


st.header("Q7. Remove Stopwords")

stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["stopwords_removed"] = df["emoji_removed"].apply(remove_stopwords)

st.write("### Samples")

st.dataframe(
    df[["emoji_removed", "stopwords_removed"]].head(5),
    use_container_width=True
)


# Q8


st.header("Q8. Complete Text Cleaning Pipeline")

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove emojis
    text = text.encode(
        "ascii",
        "ignore"
    ).decode()

    # Remove stopwords
    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["cleaned_text"] = df["text"].apply(clean_text)

st.write("### Original vs Cleaned Text")

st.dataframe(
    df[["text", "cleaned_text"]].head(10),
    use_container_width=True
)

# Q9

st.header("Q9. Text Length Analysis")

# Count words in cleaned text
df["text_length"] = df["cleaned_text"].apply(
    lambda x: len(x.split())
)

st.subheader("Statistics")

avg_len = df["text_length"].mean()
min_len = df["text_length"].min()
max_len = df["text_length"].max()

col1, col2, col3 = st.columns(3)

col1.metric("Average Length", round(avg_len, 2))
col2.metric("Minimum Length", min_len)
col3.metric("Maximum Length", max_len)

st.subheader("Histogram")

fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    df["text_length"],
    bins=30
)

ax.set_title("Histogram of Text Length")
ax.set_xlabel("Number of Words")
ax.set_ylabel("Frequency")

st.pyplot(fig)

st.subheader("Observations")

st.success("""
1. Most cleaned sentences contain a small number of words.

2. Cleaning removes unnecessary symbols, digits and stopwords.

3. Vocabulary becomes cleaner and smaller.

4. The dataset is now suitable for NLP and Machine Learning models.
""")

# Q10

st.header("Q10. Mini Project")

output_file = "cleaned_emotions.csv"

df.to_csv(
    output_file,
    index=False
)

st.success("✅ cleaned_emotions.csv saved successfully!")

st.subheader("Emotion Counts")

st.dataframe(
    df["emotions"].value_counts()
)

st.subheader("Download Cleaned Dataset")

with open(output_file, "rb") as file:

    st.download_button(
        label="Download cleaned_emotions.csv",
        data=file,
        file_name="cleaned_emotions.csv",
        mime="text/csv"
    )

st.balloons()
