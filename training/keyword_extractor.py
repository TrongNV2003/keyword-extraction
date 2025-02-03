from typing import List

import matplotlib.pyplot as plt
import torch
from gensim import corpora
from gensim.models import LdaModel
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
from wordcloud import WordCloud


class TfidfKeywordExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            # ngram_range=(1, 3),
            token_pattern=r"(?u)\b\w+\b",
        )

    def fit(self, corpus):
        self.vectorizer.fit(corpus)

    def extract(self, filtered_text: str, top_k: int = 5) -> List[str]:
        """
        Trích xuất từ khóa dựa trên điểm TF-IDF

        Args:
            filtered_text (str): Text đã được preprocessing
            top_k (int): Số lượng từ khóa cần trích xuất

        Returns:
            list: Danh sách các từ khóa
        """
        tfidf_matrix = self.vectorizer.transform(filtered_text)
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(feature_names[i], scores[i]) for i in top_indices]


class BertKeywordExtractor:
    def __init__(self):
        pass

    def extract(
        self, filtered_text: str, model: str, top_k: int = 5
    ) -> List[str]:
        """
        Args:
            filtered_text (str): Text đã được preprocessing
            top_k (int): Số lượng từ khóa cần trích xuất

        Returns:
            list: Danh sách các từ khóa
        """
        kw_model = KeyBERT(model=model)

        # hoặc có thể dùng hàm model embedding (model=self.phobert_embedding)

        keywords = kw_model.extract_keywords(
            filtered_text,
            use_maxsum=True,
            nr_candidates=20,
            top_n=top_k,
        )
        return keywords

    # PhoBERT embedding
    def phobert_embedding(self, texts: str) -> List[float]:
        MODEL = "models/phobert-base"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL)
        inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = model(**inputs)[0]  # Last hidden state
        return embeddings.numpy()


class WordCloudVisualizer:
    def __init__(self):
        pass

    def visualize_wordcloud(self, keyword_dict: dict) -> None:
        """
        Hàm vẽ wordcloud từ danh sách keyword

        Args:
            keyword_dict (dict): Danh sách keyword và score tương ứng

        Returns:
            None
        """
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(keyword_dict)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(
            "GHTK Keyword Extraction",
            fontsize=16,
            fontweight="bold",
            color="green",
        )
        plt.show()


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
