from typing import List

from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer

from training.algorithms.tf_idf import Tfidf


class TfidfKeywordExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(2, 3),
            token_pattern=r"(?u)\b\w+\b",
        )

    def fit(self, corpus):
        self.vectorizer.fit(corpus)

    def extract(self, text: List[str], top_k: int = 5) -> list:
        """Trích xuất từ khóa dựa trên điểm TF-IDF"""
        tfidf_matrix = self.vectorizer.transform(text)
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(feature_names[i], scores[i]) for i in top_indices]


class LDAExtractor:
    def __init__(self):
        pass

    def lda_keyword_extraction(
        self, texts: str, num_topics: int, top_k: int
    ) -> dict:
        """Chạy LDA để trích xuất từ khóa"""

        # Tạo từ điển và ma trận từ liệu
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = LdaModel(
            corpus, num_topics=num_topics, id2word=dictionary, passes=10
        )

        topic_keywords = {}
        for idx, topic in lda_model.show_topics(
            formatted=False, num_words=top_k
        ):
            for word, score in topic:
                topic_keywords[word] = score

        return topic_keywords
