"""
SMS Spam Assignment — Text Preprocessing & Vectorization
Reproducible standalone script applying:
1. Lowercasing & Punctuation Removal (List Comprehension)
2. Tokenization (NLTK word_tokenize)
3. Stopwords Removal (List Comprehension)
4. Lemmatization (WordNetLemmatizer + List Comprehension)
5. Vectorization: CountVectorizer, N-Grams (1, 3), and TF-IDF (1, 2)
"""

import os
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def setup_nltk():
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, quiet=True)


def main():
    setup_nltk()
    os.makedirs("figures", exist_ok=True)
    pd.set_option("display.max_colwidth", 200)

    # 1. Load dataset
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "body_text"]
    print(f"Dataset Shape: {df.shape}")
    print("Class Distribution:\n", df["label"].value_counts())

    # Plot 1: Class Distribution
    counts = df["label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#2ecc71", "#e74c3c"]
    counts.plot.pie(
        autopct="%1.1f%%",
        shadow=True,
        startangle=60,
        counterclock=False,
        colors=colors,
        ax=axes[0],
    )
    axes[0].set_ylabel("")
    axes[0].set_title("Ham vs. Spam Percentage", fontweight="bold")

    sns.barplot(x=counts.index, y=counts.values, palette=colors, ax=axes[1])
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 80, f"{v:,}", ha="center", fontweight="bold")
    axes[1].set_title("Message Count by Class", fontweight="bold")
    axes[1].set_xlabel("Class Label")
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(0, max(counts.values) * 1.12)
    plt.tight_layout()
    plt.savefig("figures/01_class_distribution.png", dpi=150)
    plt.close()

    # 2. Step-by-Step Text Preprocessing (List Comprehension + Lambda)
    stopwords_En = set(stopwords.words("english"))
    wn = nltk.WordNetLemmatizer()

    def remove_punct(text):
        return "".join([char.lower() for char in text if char not in string.punctuation])

    def tokenize(text):
        return word_tokenize(text)

    def remove_stopwords(tokenized_list):
        return [word for word in tokenized_list if word not in stopwords_En]

    def lemmatizing(tokenized_text):
        return [wn.lemmatize(word) for word in tokenized_text]

    df["body_text_nopunc"] = df["body_text"].apply(lambda x: remove_punct(x))
    df["body_text_tokenized"] = df["body_text_nopunc"].apply(lambda x: tokenize(x))
    df["body_text_nostop"] = df["body_text_tokenized"].apply(lambda x: remove_stopwords(x))
    df["body_text_lemmatized"] = df["body_text_nostop"].apply(lambda x: lemmatizing(x))

    # Unified clean_text function
    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = word_tokenize(text)
        text = " ".join([wn.lemmatize(word) for word in tokens if word not in stopwords_En])
        return text

    df["cleaned_text"] = df["body_text"].apply(lambda x: clean_text(x))
    print("\n--- Text Before vs. After Preprocessing (First 5 Rows) ---")
    print(df[["label", "body_text", "cleaned_text"]].head(5))

    # Plot 2: Text Length Before vs. After Preprocessing
    raw_len = df["body_text"].str.len()
    clean_len = df["cleaned_text"].str.len()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(raw_len, fill=True, color="#3498db", label=f"Before Preprocessing (Mean: {raw_len.mean():.1f} chars)", ax=ax)
    sns.kdeplot(clean_len, fill=True, color="#9b59b6", label=f"After Preprocessing (Mean: {clean_len.mean():.1f} chars)", ax=ax)
    ax.set_xlim(0, 300)
    ax.set_title("Character Length Distribution: Before vs. After Preprocessing", fontweight="bold")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/02_preprocessing_length_reduction.png", dpi=150)
    plt.close()

    # 3. Vectorization
    vectorizer = CountVectorizer()
    features_CountVec = vectorizer.fit_transform(df["cleaned_text"])

    ngram_vect = CountVectorizer(ngram_range=(1, 3))
    features_ngram = ngram_vect.fit_transform(df["cleaned_text"])

    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    features_tfidf = tfidf.fit_transform(df["cleaned_text"])

    print("\n--- Vectorization Shapes ---")
    print(f"CountVectorizer (1,1): {features_CountVec.shape} (nnz={features_CountVec.nnz:,})")
    print(f"N-Grams (1,3)        : {features_ngram.shape} (nnz={features_ngram.nnz:,})")
    print(f"TF-IDF (1,2)         : {features_tfidf.shape} (nnz={features_tfidf.nnz:,})")

    # Plot 3: Vectorization Comparison
    methods = ["CountVectorizer\n(Unigrams)", "TF-IDF\n(1-2 Grams)", "N-Grams\n(1-3 Grams)"]
    vocab_sizes = [features_CountVec.shape[1], features_tfidf.shape[1], features_ngram.shape[1]]
    nnz_counts = [features_CountVec.nnz, features_tfidf.nnz, features_ngram.nnz]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_colors = ["#1abc9c", "#f39c12", "#34495e"]

    axes[0].bar(methods, vocab_sizes, color=bar_colors)
    for i, v in enumerate(vocab_sizes):
        axes[0].text(i, v + 1200, f"{v:,}", ha="center", fontweight="bold")
    axes[0].set_title("Vocabulary Size (Features)", fontweight="bold")
    axes[0].set_ylabel("Number of Features")
    axes[0].set_ylim(0, max(vocab_sizes) * 1.15)

    axes[1].bar(methods, nnz_counts, color=bar_colors)
    for i, v in enumerate(nnz_counts):
        axes[1].text(i, v + 2500, f"{v:,}", ha="center", fontweight="bold")
    axes[1].set_title("Non-Zero Matrix Entries (NNZ)", fontweight="bold")
    axes[1].set_ylabel("Non-Zero Entries")
    axes[1].set_ylim(0, max(nnz_counts) * 1.15)

    plt.tight_layout()
    plt.savefig("figures/03_vectorization_comparison.png", dpi=150)
    plt.close()
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
