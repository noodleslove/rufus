# src/client2.py
import os
import time
import json
import logging
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Optional
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title


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
        """
        Use LLM to analyze content relevance and structure data with few-shot learning
        and structured prompt engineering.
        """
        system_prompt = """You are a precise web content analyzer that evaluates content relevance and extracts structured data.
        You must always return valid JSON matching the exact schema provided.
        Focus on identifying key information related to the user's instructions.
        
        Rules:
        1. Always return valid JSON
        2. Follow the exact schema provided
        3. Be conservative with relevance scores
        4. Extract only information that directly relates to the instructions
        5. Clean and normalize extracted text
        """

        few_shot_examples = """
        Example 1:
        Instructions: "Find information about product pricing and features"
        Content: "Our premium plan costs $29/month and includes cloud storage, 24/7 support, and automated backups."
        Output: {
            "is_relevant": true,
            "relevance_score": 0.9,
            "extracted_data": {
                "pricing": {
                    "plan_type": "premium",
                    "cost": "$29/month"
                },
                "features": [
                    "cloud storage",
                    "24/7 support",
                    "automated backups"
                ]
            }
        }

        Example 2:
        Instructions: "Find information about product pricing and features"
        Content: "Check out our latest blog post about industry trends and news updates!"
        Output: {
            "is_relevant": false,
            "relevance_score": 0.1,
            "extracted_data": {}
        }

        Example 3:
        Instructions: "Find HR policies and employee benefits"
        Content: "Employees are entitled to 20 days of paid vacation, health insurance, and 401k matching up to 5%."
        Output: {
            "is_relevant": true,
            "relevance_score": 0.95,
            "extracted_data": {
                "benefits": {
                    "vacation_days": 20,
                    "insurance": ["health"],
                    "retirement": "401k with 5% matching"
                }
            }
        }
        """

        analysis_prompt = f"""Based on the examples above, analyze the following content and return structured data:

        Instructions: "{instructions}"
        
        Content: "{content[:4000]}"  # Truncate to avoid token limits

        Return valid JSON that exactly matches this schema:
        {{
            "is_relevant": boolean,  // true if content matches instructions
            "relevance_score": float,  // 0.0 to 1.0
            "extracted_data": {{  // structured data based on content type
                // dynamically structured based on content
            }}
        }}
        
        Important: Return ONLY the JSON object, no additional text or explanation."""

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=few_shot_examples),
                    HumanMessage(content=analysis_prompt),
                ]
            )

            # Clean the response to handle potential formatting issues
            cleaned_response = self._clean_json_response(response.content)
            result = json.loads(cleaned_response)

            # Validate response structure
            required_fields = {"is_relevant", "relevance_score", "extracted_data"}
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in response")

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.debug(f"Failed response content: {response.content}")
            return {"is_relevant": False, "relevance_score": 0.0, "extracted_data": {}}
        except Exception as e:
            self.logger.error(f"Error in content analysis: {str(e)}")
            return {"is_relevant": False, "relevance_score": 0.0, "extracted_data": {}}

    def _clean_json_response(self, response: str) -> str:
        """Clean and normalize JSON response from LLM."""
        # Remove any markdown code block indicators
        response = response.replace("```json", "").replace("```", "")

        # Find the first '{' and last '}' to extract just the JSON object
        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No valid JSON object found in response")

        return response[start : end + 1]

    def _validate_json_structure(self, data: Dict) -> bool:
        """Validate that the JSON response has the required structure."""
        if not isinstance(data.get("is_relevant"), bool):
            return False

        relevance_score = data.get("relevance_score")
        if not isinstance(relevance_score, (int, float)):
            return False
        if not 0 <= relevance_score <= 1:
            return False

        if not isinstance(data.get("extracted_data"), dict):
            return False

        return True

    def process_page(self, url: str, instructions: str) -> Dict:
        """Process a single page: fetch, analyze, and extract data."""
        content = self.get_page_content(url)
        if not content:
            return {"url": url, "data": None}

        # Extract main content (remove navigation, footers, etc.)
        elements = partition_html(url=url, skip_headers_and_footers=True)
        chunks = chunk_by_title(
            elements, combine_text_under_n_chars=100, max_characters=3000
        )
        chunk_dicts = [chunk.to_dict() for chunk in chunks]

        # Analyze content relevance
        analysis = self.analyze_content_relevance(content, instructions)

        return {
            "url": url,
            "data": chunk_dicts,
            "is_relevant": analysis.get("is_relevant", False),
            "relevance_score": analysis.get("relevance_score", 0.0),
            "llm_extracted_data": analysis.get("extracted_data", {}),
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
