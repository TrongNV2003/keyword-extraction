import json
from typing import List, Mapping, Tuple

import numpy as np

from training.preprocessing import TextPreprocess

preprocess = TextPreprocess()


class Dataset:
    def __init__(self, json_file: str) -> None:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> str:
        """
        Get the item at the given index

        Returns:
            text: the text of the item
            label: the label of the item
        """

        item = self.data[index]
        clean_title = preprocess.process_text(item["title"])
        clean_summary = preprocess.process_text(item["summary"])
        clean_body = preprocess.process_text(item["body"])

        return clean_title, clean_summary, clean_body
