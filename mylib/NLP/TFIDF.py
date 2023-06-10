from pathlib import Path
import os
import sys

sys.path.append(str(Path(os.getcwd())))


import numpy as np
from mylib.NLP.CountVectorizer import CountVectorizer


class TFIDFVectorizer:
    def __init__(self, stop_words: list = []) -> None:
        self.stop_words = stop_words
        self.bow = CountVectorizer(self.stop_words)

    def fit(self, lemma_texts: list) -> None:
        self.bow.fit(lemma_texts)
        return self

    def transform(self, lemma_texts: list) -> np.ndarray:
        matrix = self.bow.transform(lemma_texts)
        tf = matrix / np.sum(matrix, axis=1, keepdims=True)
        idf = np.log((matrix.shape[0] + 1)/(np.sum(matrix > 0, axis=0, keepdims=True) + 1)) + 1
        return tf * idf

    def fit_transform(self, lemma_texts: list) -> np.ndarray:
        _ = self.fit(lemma_texts)
        return self.transform(lemma_texts)


a = TFIDFVectorizer()
print(a.fit_transform(["Hello Ivan My name is Ivan", "Yesterday I met a new friend his name is Ivan", "Our meating was very nice I like make new friends"]))
# print(a.vocabulary_, a.words)