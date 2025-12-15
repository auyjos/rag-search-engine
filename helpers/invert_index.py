import os
import pickle

from helpers.tokenizer import tokenize


class InvertedIndex:
    def __init__(self, stopwords: set[str] | None = None):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.stopwords = stopwords or set()

    def __add_document(self, doc_id:int, text: str) -> None:
        for token in tokenize(text, self.stopwords):
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term:str)->list[int]:
        tokens = tokenize(term, self.stopwords)
        if not tokens:
            return []
        token = tokens[0]
        return sorted(self.index.get(token, set()))
    
    def build(self, movies: list[dict]) -> None:
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id,f"{movie['title']} {movie['description']}")

    def save(self, index_path: str = "cache/index.pkl", docmap_path: str = "cache/docmap.pkl") -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    @classmethod
    def load(cls, stopwords: set[str] | None = None, index_path: str = "cache/index.pkl", docmap_path: str = "cache/docmap.pkl") -> "InvertedIndex":
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            docmap = pickle.load(f)
        inst = cls(stopwords or set())
        inst.index = index
        inst.docmap = docmap
        return inst