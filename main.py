import argparse

import matplotlib.pyplot as plt
from underthesea import chunk, pos_tag, word_tokenize
from wordcloud import WordCloud

from training.data_crawler import DataCrawler
from training.data_loader import Dataset
from training.keyword_extractor import TfidfKeywordExtractor

crawler = DataCrawler()
extractor = TfidfKeywordExtractor()

parser = argparse.ArgumentParser(
    description="Inference script for keyword extraction"
)
parser.add_argument(
    "--url", type=str, help="url of a newspaper from zingnews", required=True
)
parser.add_argument(
    "--top_k", type=str, help="select top k words to extract", default=20
)
args = parser.parse_args()


def filter_phrases_by_pos(text, tags=["N", "Np", "Nu", "V", "A"]):
    """Giữ lại các cụm từ có POS phù hợp"""
    tagged = pos_tag(text)
    return " ".join([word for word, tag in tagged if tag in tags])


if __name__ == "__main__":
    data_crawl = crawler.crawl_data(args.url)

    dataset = Dataset(data_crawl)
    title = [dataset[i][0] for i in range(len(dataset))]
    summary = [dataset[i][1] for i in range(len(dataset))]
    body = [dataset[i][2] for i in range(len(dataset))]

    body = word_tokenize(body[0])
    filtered_text = filter_phrases_by_pos(" ".join(body))

    extractor.fit([filtered_text])
    keywords = extractor.extract(filtered_text, int(args.top_k))
    keyword_dict = {word: score for word, score in keywords}
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(keyword_dict)

    # Plot Word Cloud
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
