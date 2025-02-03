from typing import List

from underthesea import pos_tag


class TextPreprocess:
    def __init__(self):
        pass

    def filter_pos_tagging(
        self, text: str, tags=["N", "Np", "V", "A"]
    ) -> List[str]:
        """Giữ lại các cụm từ có POS tagging phù hợp

        Args:

        text (str): Đoạn văn bản cần lọc
        tags (list): Danh sách các loại POS cần giữ lại

        Returns:

        list: Danh sách các từ hoặc cụm từ
        """

        tagged = pos_tag(text)
        return [word for word, tag in tagged if tag in tags]
