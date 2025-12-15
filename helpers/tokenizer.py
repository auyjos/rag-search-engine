import string

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def load_stopwords() -> set[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        return {word for word in f.read().splitlines() if word}


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    text = remove_punctuation(text.lower())
    tokens = [token for token in text.split() if token]
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens
