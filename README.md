# keyword-extraction
- Extract word or phrase from a newspaper from Zingnews
    + Crawl data from url of a newspaper
    + Extract words or phrases: (e.g: Noun, Adj, Verb...) from a newspaper
    + Visualize keyword into word cloud

## Installation
- You have to install browser driver to crawl data. In this repo i use edge driver, download from [edge](https://developer.microsoft.com/vi-vn/microsoft-edge/tools/webdriver/?cs=394355647&form=MA13LH)

```sh
pip install -r requirements.txt
```

### Usage
```sh
bash main.sh
```

- Note: you can change parameters in file main.sh:
    + url: you can select another url from [Zingnews](https://znews.vn/) to crawl text
    + models: embedding models
    + top k: select top k words to extract
    + method: you can use tf-idf method or bert method (embedding model only works in bert method)

### Future plans
- Tokenize các từ chính xác hơn, sử dụng các phương pháp khác để extract keyword
