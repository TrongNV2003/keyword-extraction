import re
from typing import List

from underthesea import word_tokenize


class TextPreprocess:
    def __init__(self):
        self.stopwords_file = "stopwords/vietnamese-stopwords.txt"

    def process_text(self, text: List[str]) -> str:
        """
        This function will process the text by removing stopwords, punctuation, and whitespace
        """

        cleaned_text = self._remove_whitespace(text)
        cleaned_text = self._remove_stopwords(cleaned_text)
        # cleaned_text = self._lowercase(cleaned_text)
        cleaned_text = self._remove_punctuation(cleaned_text)
        cleaned_text = self._remove_whitespace(cleaned_text)
        return cleaned_text

    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the text
        - Tokenize into word phrase for correct removing stopwords

        Parameters:
            text: str

        Returns:
            text after removing stopwords
        """

        stopwords = self.load_stopwords(self.stopwords_file)

        # tokenize into word phrase for correct removing stopwords
        words = self.word_segment(text)

        filtered_words = [word for word in words if word not in stopwords]
        return " ".join(filtered_words)

    def load_stopwords(self, stopword_file: str) -> List[str]:
        with open(stopword_file, "r", encoding="utf-8") as f:
            stopwords = f.read().splitlines()
        return stopwords

    def word_segment(self, text: str) -> List[str]:
        """
        This function will segment the text into words or phrases
        Tokenize into phrases for correct removing stopwords

        Parameters:
            text (str): context to be segmented

        Returns:
            token (list): list of words
        """

        tokens = word_tokenize(text)
        return tokens

    def _remove_punctuation(self, text: str) -> str:
        """
        This function will remove punctuation from the text

        Parameters:
            text: str

        Returns:
            text without punctuation
        """

        cleaned_text = re.sub(r"[^\w\s]", "", text)
        return cleaned_text

    def _remove_whitespace(self, text: str) -> str:
        """
        This function will remove whitespace from the text

        Parameters:
            text: str

        Returns:
            text without whitespace
        """

        cleaned_text = re.sub(r"\s+", " ", text).strip()
        return cleaned_text

    def _lowercase(self, text: str) -> str:
        """
        This function will convert the text to lowercase

        Parameters:
            text: str

        Returns:
            lower text
        """

        return text.lower()
