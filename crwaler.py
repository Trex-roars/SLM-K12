import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Set
from dataclasses import dataclass
import json
import logging
import re
from urllib.parse import urljoin
import asyncio
from async_timeout import timeout
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class SubjectLink:
    class_name: str
    subject_name: str
    url: str
    chapter_links: List[str] = None

    def __post_init__(self):
        if self.chapter_links is None:
            self.chapter_links = []

class LearnCBSECrawler:
    def __init__(self, max_connections=50):
        self.base_url = "https://www.learncbse.in/ncert-solutions-2/"
        self.max_connections = max_connections
        self.session = None
        self.semaphore = None
        self.visited_urls = set()
        self.class_links = {}  # Dictionary to store class -> subject links
        self.subject_data = []  # List to store all subject data with their chapters

    async def init_session(self):
        """Initialize aiohttp session with connection pooling."""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            ssl=False
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.semaphore = asyncio.Semaphore(self.max_connections)

    async def fetch_page(self, url: str) -> str:
        """Fetch page content with retries and error handling."""
        if url in self.visited_urls:
            return None

        retries = 3
        while retries > 0:
            try:
                async with self.semaphore:
                    async with timeout(30):
                        async with self.session.get(url) as response:
                            if response.status == 200:
                                self.visited_urls.add(url)
                                return await response.text()
                            elif response.status == 429:  # Too Many Requests
                                await asyncio.sleep(5)
                                retries -= 1
                                continue
            except Exception as e:
                logging.error(f"Error fetching {url}: {e}")
                retries -= 1
                if retries > 0:
                    await asyncio.sleep(1)
        return None

    def extract_class_links(self, html: str) -> Dict[str, List[str]]:
        """Extract class links and their corresponding subject links."""
        soup = BeautifulSoup(html, 'html.parser')
        class_links = {}

        # Find class sections using common patterns
        for anchor in soup.find_all('a', href=True):
            text = anchor.get_text(strip=True)
            href = anchor['href']

            # Match class patterns (Class 6-12)
            class_match = re.search(r'class[- ](1[0-2]|[6-9])', href.lower())
            if class_match:
                class_num = class_match.group(1)
                class_name = f"Class {class_num}"
                if class_name not in class_links:
                    class_links[class_name] = []
                class_links[class_name].append(href)

        return class_links

    def extract_subject_links(self, html: str, class_name: str) -> List[SubjectLink]:
        """Extract subject links for a specific class."""
        soup = BeautifulSoup(html, 'html.parser')
        subject_links = []

        # Common subject patterns
        subject_patterns = ['maths', 'science', 'physics', 'chemistry', 'biology', 'english', 'hindi', 'sanskrit']

        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            text = anchor.get_text(strip=True)

            # Check if link contains subject pattern
            if any(subject in href.lower() for subject in subject_patterns):
                subject_name = text.strip()
                subject_links.append(SubjectLink(
                    class_name=class_name,
                    subject_name=subject_name,
                    url=href
                ))

        return subject_links

    def extract_chapter_links(self, html: str) -> List[str]:
        """Extract chapter links from subject page."""
        soup = BeautifulSoup(html, 'html.parser')
        chapter_links = []

        # Look for links that might be chapters
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            text = anchor.get_text(strip=True)

            # Common chapter indicators
            if any(pattern in href.lower() for pattern in [
                'chapter', 'exercise', 'solutions', 'ncert'
            ]):
                chapter_links.append(href)

        return list(set(chapter_links))  # Remove duplicates

    async def process_subject_page(self, subject: SubjectLink):
        """Process a subject page to extract chapter links."""
        html = await self.fetch_page(subject.url)
        if html:
            with ThreadPoolExecutor() as executor:
                chapter_links = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.extract_chapter_links,
                    html
                )
                subject.chapter_links = chapter_links

    async def crawl(self):
        """Main crawling function."""
        await self.init_session()

        try:
            # Fetch main page
            html = await self.fetch_page(self.base_url)
            if not html:
                logging.error("Failed to fetch main page")
                return

            # Extract class links
            with ThreadPoolExecutor() as executor:
                self.class_links = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.extract_class_links,
                    html
                )

            # Process each class and its subjects
            for class_name, links in self.class_links.items():
                for link in links:
                    html = await self.fetch_page(link)
                    if html:
                        # Extract subject links
                        with ThreadPoolExecutor() as executor:
                            subject_links = await asyncio.get_event_loop().run_in_executor(
                                executor,
                                self.extract_subject_links,
                                html,
                                class_name
                            )

                        # Process each subject page in parallel
                        tasks = [self.process_subject_page(subject) for subject in subject_links]
                        await asyncio.gather(*tasks)

                        self.subject_data.extend(subject_links)

        finally:
            await self.session.close()

    def save_results(self, filename: str = 'learncbse_links.json'):
        """Save crawled data to JSON file."""
        data = {
            "classes": {}
        }

        for subject in self.subject_data:
            if subject.class_name not in data["classes"]:
                data["classes"][subject.class_name] = {}

            if subject.subject_name not in data["classes"][subject.class_name]:
                data["classes"][subject.class_name][subject.subject_name] = {
                    "subject_url": subject.url,
                    "chapter_links": subject.chapter_links
                }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

async def main():
    logging.info("Starting LEARN CBSE crawler...")

    crawler = LearnCBSECrawler()
    await crawler.crawl()

    crawler.save_results()

    logging.info(f"Crawling complete. Visited {len(crawler.visited_urls)} pages.")
    logging.info("Results have been saved to 'learncbse_links.json'")

if __name__ == "__main__":
    asyncio.run(main())
