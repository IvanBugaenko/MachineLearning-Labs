import numpy as np
import spacy as sp


nlp = sp.load("en_core_web_sm")


class CountVectorizer:
    def __init__(self, ngram_range: tuple[int, int], min_df=1, max_features=None) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features

    def fit(self, texts: np.ndarray) -> None:
        lemma_texts = [self.__lemmatization(text) for text in texts]
        print(lemma_texts)
        # for text_num, text in enumerate(lemma_texts)
        return lemma_texts


    def transform(self, ):
        ...

    def fit_transform(self, X: np.ndarray):
        _ = self.fit(X)
        return ...

    def __lemmatization(self, text: np.ndarray) -> np.ndarray:
        return [token.lemma_ for token in nlp(text) if (token.lower_ not in nlp.Defaults.stop_words) and (not token.is_punct)]

    def todense(self):
        ...


a = CountVectorizer((1,))
print(a.fit(["Hello, Ivan. My name is Ivan!", "Yesterday I met a new friend, his name is Ivan", "Our meating was very nice, I like make new friends!"]))