import datetime
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service


class DataCrawler:
    def __init__(self) -> None:
        self.service = "/home/trongnv130/Desktop/keyword-extraction/edgedriver_linux64/msedgedriver"

    def crawl_data(self, url: str, record: bool = True) -> None:
        """
        This function crawl data from a given url

        Args:
            url (str): url to crawl data

        Returns:
            json data: data crawled from the url
        """

        service = Service(self.service)
        driver = webdriver.Edge(service=service)

        driver.get(url)

        data_crawl = []

        # get title
        elems_title = driver.find_elements(
            By.CSS_SELECTOR, ".the-article-title"
        )
        title = [elem_title.text for elem_title in elems_title][0]

        # get sumary
        elem_sumary = driver.find_elements(
            By.CSS_SELECTOR, ".the-article-summary"
        )
        sumary = [elem_sumary.text for elem_sumary in elem_sumary][0]

        # get body
        elem_body = driver.find_elements(By.CSS_SELECTOR, ".the-article-body")
        body = [elem_body.text for elem_body in elem_body][0]

        driver.quit()

        data_crawl.append({"title": title, "summary": sumary, "body": body})
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H_%M")

        if record:
            with open(
                f"data/data_crawl-{current_time}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(data_crawl, f, ensure_ascii=False, indent=4)

        return data_crawl
