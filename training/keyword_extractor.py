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

    def extract(self, text: str, top_k: int = 5) -> list:
        """Trích xuất từ khóa dựa trên điểm TF-IDF"""
        tfidf_matrix = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(feature_names[i], scores[i]) for i in top_indices]
