import argparse

from training.data_crawler import DataCrawler
from training.data_loader import Dataset
from training.keyword_extractor import (
    BertKeywordExtractor,
    TfidfKeywordExtractor,
    WordCloudVisualizer,
)
from training.preprocessing import TextPreprocess

crawler = DataCrawler()
preprocess = TextPreprocess()

tfidf_extractor = TfidfKeywordExtractor()
bert_extractor = BertKeywordExtractor()

wordcloud = WordCloudVisualizer()

parser = argparse.ArgumentParser(
    description="Inference script for keyword extraction"
)
parser.add_argument(
    "--url", type=str, help="url of a newspaper from zingnews", required=True
)
parser.add_argument(
    "--top_k", type=int, help="select top k words to extract", default=20
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="required embedding language model",
    default="paraphrase-multilingual-MiniLM-L12-v2",
)
parser.add_argument(
    "--method", type=str, help="select method to extract", default="tfidf"
)
args = parser.parse_args()


if __name__ == "__main__":
    data_crawl = crawler.crawl_data(args.url, record=False)

    dataset = Dataset(data_crawl)
    title = [dataset[i][0] for i in range(len(dataset))]
    summary = [dataset[i][1] for i in range(len(dataset))]
    body = [dataset[i][2] for i in range(len(dataset))]

    corpus = preprocess.filter_pos_tagging(body[0])

    tokens = [token.replace(" ", "_") for token in corpus]
    filtered_text = [" ".join(tokens)]

    if args.method == "tfidf":
        # Extract keywords tf-idf
        tfidf_extractor.fit(filtered_text)
        keywords = tfidf_extractor.extract(filtered_text, top_k=args.top_k)
    elif args.method == "bert":
        # Extract keywords PhoBERT
        keywords = bert_extractor.extract(
            filtered_text, top_k=args.top_k, model=args.model_name_or_path
        )

    keyword_dict = {word.replace("_", " "): score for word, score in keywords}

    # Visualize keyword to wordcloud
    wordcloud.visualize_wordcloud(keyword_dict)
