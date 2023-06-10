import re
import numpy as np
from collections import Counter


class CountVectorizer:
    def __init__(self, stop_words: list = []) -> None:
        self.stop_words = stop_words

    def fit(self, lemma_texts: list) -> None:
        words = set()
        for text in lemma_texts:
            words.update(text.split())
        self.words = sorted(words)
        return self

    def transform(self, lemma_texts: list[str]):
        matrix = np.zeros((len(lemma_texts), len(self.words)))
        for word_num, word in enumerate(self.words):
            for doc_num, doc in enumerate(lemma_texts):
                matrix[doc_num][word_num] += doc.split().count(word)
        self.matrix = matrix
        return matrix

    def fit_transform(self, lemma_texts: list):
        _ = self.fit(lemma_texts)
        return self.transform(lemma_texts)

    @property
    def vocabulary_(self):
        word_sum = np.sum(self.matrix, axis=0, dtype=int)
        return dict(zip(self.words, word_sum))
