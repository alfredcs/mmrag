import requests # Required to make HTTP requests
from bs4 import BeautifulSoup  # Required to parse HTML
import numpy as np # Required to dedupe sites
from urllib.parse import unquote # Required to unquote URLs
from xml.etree import ElementTree

# Saerch google and bing with a query and return urls
class SearchEngine:
    def __init__(self):
        self.google_url = "https://www.google.com/search?q="
        self.bing_url = "https://www.bing.com/search?q="
        #self.bing_url = "https://www.bing.com/search?q={query.replace(' ', '+')}"

    def search(self, query, count: int=10):
        google_urls = self.search_goog(query, count)
        bing_urls = self.search_bing(query, count)
        combined_urls = google_urls + bing_urls
        return list(set(combined_urls))  # Remove duplicates

    def search_goog(self, query, count):
        #response = requests.get(f"https://www.google.com/search?q={query}") # Make the request
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser") # Parse the HTML
        links = soup.find_all("a") # Find all the links in the HTML
        urls = []
        for l in [link for link in links if link["href"].startswith("/url?q=")]:
            # get the url
            url = l["href"]
            # remove the "/url?q=" part
            url = url.replace("/url?q=", "")
            # remove the part after the "&sa=..."
            url = unquote(url.split("&sa=")[0])
            # special case for google scholar
            if url.startswith("https://scholar.google.com/scholar_url?url=http"):
                url = url.replace("https://scholar.google.com/scholar_url?url=", "").split("&")[0]
            elif 'google.com/' in url: # skip google links
                continue
            elif 'youtube.com/' in url:
                continue
            elif 'search?q=' in url:
                continue
            if url.endswith('.pdf'): # skip pdf links
                continue
            if '#' in url: # remove anchors (e.g. wikipedia.com/bob#history and wikipedia.com/bob#genetics are the same page)
                url = url.split('#')[0]
            # print the url
            urls.append(url)
        return urls
        
    def search_google(self, query, count):
        params = {
            "q": query,
            "num": count  # Number of results to retrieve
        }
        response = requests.get(self.google_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".yuRUbf a")]
        return urls

    def search_bing(self, query, count):
        params = {
            "q": query,
            "count": count # Number of results to retrieve
        }
        response = requests.get(self.bing_url, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [link.get("href") for link in soup.select(".b_algo h2 a")]
        return urls[:count]

from xml.etree import ElementTree

class ArxivSearcher:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query?"

    def search(self, query, max_results=10):
        """Search arXiv for the given query and return a list of article URLs."""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return self.parse_response(response.text)
        else:
            print(f"Error fetching results from arXiv: {response.status_code}")
            return []

    def parse_response(self, xml_data):
        """Parse the XML response from arXiv and extract article URLs."""
        namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
        root = ElementTree.fromstring(xml_data)
        urls = []
        for entry in root.findall('arxiv:entry', namespace):
            id_tag = entry.find('arxiv:id', namespace)
            if id_tag is not None:
                urls.append(id_tag.text)
        return urls