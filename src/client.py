# src/client.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from typing import List, Dict, Set, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent
import time
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class RufusClient:
    def __init__(self, api_key: str, max_pages: int = 10, concurrent_requests: int = 3):
        """Initialize Rufus client with API key and configuration."""
        self.api_key = api_key
        self.max_pages = max_pages
        self.concurrent_requests = concurrent_requests
        self.visited_urls: Set[str] = set()
        self.user_agent = UserAgent()
        self.setup_logging()

        # Initialize LangChain chat model for content analysis
        self.llm = ChatOpenAI(
            openai_api_key=api_key, temperature=0.1, model="gpt-4o-mini"
        )

    def setup_logging(self):
        """Configure logging for the client."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def is_valid_url(self, url: str, base_url: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)
            return parsed_url.netloc == parsed_base.netloc and not url.endswith(
                (".pdf", ".jpg", ".png", ".gif")
            )
        except Exception:
            return False

    def get_page_content(self, url: str) -> Optional[str]:
        """Fetch page content with error handling and rate limiting."""
        try:
            headers = {"User-Agent": self.user_agent.random}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract valid links from page."""
        links = []
        for a_tag in soup.find_all("a", href=True):
            url = urljoin(base_url, a_tag["href"])
            if self.is_valid_url(url, base_url):
                links.append(url)
        return links

    def analyze_content_relevance(self, content: str, instructions: str) -> Dict:
        """Use LLM to analyze content relevance and structure data."""
        prompt = f"""
        Analyze the following web content and determine its relevance to this task:
        {instructions}
        
        If the content is relevant, extract and structure the key information.
        If not relevant, return empty data.
        
        Content: {content[:4000]}  # Truncate to avoid token limits
        
        Return as JSON with schema:
        {{
            "is_relevant": bool,
            "extracted_data": {{}},
            "relevance_score": float  # 0-1
        }}
        """

        response = self.llm.invoke(
            [
                SystemMessage(
                    content="You are a web content analyzer. Return only valid JSON."
                ),
                HumanMessage(content=prompt),
            ]
        )

        try:
            return json.loads(response.content)
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return {"is_relevant": False, "extracted_data": {}, "relevance_score": 0}

    def process_page(self, url: str, instructions: str) -> Dict:
        """Process a single page: fetch, analyze, and extract data."""
        content = self.get_page_content(url)
        if not content:
            return {"url": url, "data": None}

        soup = BeautifulSoup(content, "html.parser")

        # Extract main content (remove navigation, footers, etc.)
        main_content = ""
        for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
            main_content += f"{tag.get_text()} "

        analysis = self.analyze_content_relevance(main_content, instructions)

        return {
            "url": url,
            "data": analysis["extracted_data"] if analysis["is_relevant"] else None,
            "relevance_score": analysis.get("relevance_score", 0),
        }

    def scrape(self, base_url: str, instructions: str) -> List[Dict]:
        """Main method to scrape website based on instructions."""
        self.logger.info(f"Starting scrape of {base_url}")
        self.visited_urls.clear()
        results = []
        urls_to_visit = [base_url]

        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            while urls_to_visit and len(self.visited_urls) < self.max_pages:
                current_url = urls_to_visit.pop(0)
                if current_url in self.visited_urls:
                    continue

                self.visited_urls.add(current_url)
                self.logger.info(f"Processing {current_url}")

                # Process page
                result = executor.submit(self.process_page, current_url, instructions)
                processed_data = result.result()

                if processed_data["data"]:
                    results.append(processed_data)

                # Get new links
                content = self.get_page_content(current_url)
                if content:
                    soup = BeautifulSoup(content, "html.parser")
                    new_links = self.extract_links(soup, base_url)
                    urls_to_visit.extend(
                        [url for url in new_links if url not in self.visited_urls]
                    )

                time.sleep(1)  # Rate limiting

        return results

    def save_results(self, results: List[Dict], output_file: str):
        """Save scraped results to file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
