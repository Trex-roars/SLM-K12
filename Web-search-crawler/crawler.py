from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator, Literal
import asyncio
import os
import json
from datetime import datetime
import httpx
import cohere
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import streamlit as st
from streamlit_chat import message
import time
import nest_asyncio

# Enable nested async loops (needed for Streamlit)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Verify environment variables
print("\nChecking environment variables:")
print(f"GOOGLE_API_KEY set: {'GOOGLE_API_KEY' in os.environ}")
print(f"COHERE_API_KEY set: {'COHERE_API_KEY' in os.environ}")

@dataclass
class WebContent:
    """Represents crawled web content."""
    title: str
    content: str
    url: str
    source_quality: float

# WebCrawler class remains the same...
#
#
class WebCrawler:
    """Handles web crawling functionality with educational focus."""

    def __init__(self):
        print("Initializing WebCrawler")
        # Educational domains to prioritize
        self.trusted_domains = {
            'wikipedia.org': 0.8,
            'britannica.com': 0.9,
            'khanacademy.org': 0.95,
            'nasa.gov': 0.9,
            'nationalgeographic.com': 0.85,
            'sciencedaily.com': 0.8,
            'education.com': 0.75,
            'scholastic.com': 0.85
        }
        self.session = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={'User-Agent': 'Educational Content Bot 1.0'}
        )
        print(f"Initialized with {len(self.trusted_domains)} trusted domains")

    def get_domain_score(self, url: str) -> float:
        """Calculate domain reliability score."""
        domain = urlparse(url).netloc.lower()
        base_domain = '.'.join(domain.split('.')[-2:])
        return self.trusted_domains.get(base_domain, 0.5)

    async def clean_text(self, soup: BeautifulSoup) -> str:
        """Clean and extract relevant text from HTML."""
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # Extract text from paragraphs and headers
        content = []
        for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'article']):
            text = elem.get_text().strip()
            if text and len(text) > 20:  # Filter out short snippets
                content.append(text)

        return '\n\n'.join(content)

    async def crawl_url(self, url: str) -> Optional[WebContent]:
        """Crawl a single URL and extract content."""
        print(f"\nCrawling URL: {url}")
        try:
            response = await self.session.get(url)
            response.raise_for_status()
            print(f"Successfully fetched {url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
            content = await self.clean_text(soup)

            if not content:
                print(f"No content found for {url}")
                return None

            result = WebContent(
                title=title,
                content=content[:2000],  # Limit content length
                url=url,
                source_quality=self.get_domain_score(url)
            )
            print(f"Successfully extracted content from {url} (quality: {result.source_quality})")
            return result

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            print(f"Full exception: {repr(e)}")
            return None

    async def search(self, query: str, num_results: int = 3) -> List[WebContent]:
        """Perform web search using DuckDuckGo and crawl results."""
        print(f"\nStarting web search for query: {query}")
        search_url = f"https://html.duckduckgo.com/html/?q={query}+site:({'+OR+'.join(self.trusted_domains.keys())})"

        try:
            print(f"Sending request to DuckDuckGo")
            response = await self.session.get(search_url)
            response.raise_for_status()
            print("Successfully received search results")

            soup = BeautifulSoup(response.text, 'html.parser')
            result_links = []

            for result in soup.find_all('a', class_='result__url'):
                url = result.get('href')
                if url:
                    full_url = urljoin('https://', url)
                    print(f"Found result URL: {full_url}")
                    result_links.append(full_url)
                if len(result_links) >= num_results:
                    break

            print(f"Found {len(result_links)} result links")

            tasks = [self.crawl_url(url) for url in result_links]
            results = await asyncio.gather(*tasks)

            valid_results = [r for r in results if r is not None]
            valid_results.sort(key=lambda x: x.source_quality, reverse=True)

            print(f"Successfully processed {len(valid_results)} valid results")
            return valid_results[:num_results]

        except Exception as e:
            print(f"Search error: {str(e)}")
            print(f"Full exception: {repr(e)}")
            return []

    async def close(self):
        """Close the HTTP session."""
        await self.session.aclose()

@dataclass
class EducationLevel:
    """Defines grade-specific parameters for content adaptation."""
    grade: int
    reading_level: Literal["basic", "intermediate", "advanced"]
    vocabulary_level: Literal["simple", "moderate", "advanced"]
    explanation_style: Literal["storytelling", "conceptual", "technical"]

    @classmethod
    def from_grade(cls, grade: int) -> "EducationLevel":
        if grade <= 4:
            return cls(grade, "basic", "simple", "storytelling")
        elif grade <= 8:
            return cls(grade, "intermediate", "moderate", "conceptual")
        else:
            return cls(grade, "advanced", "advanced", "technical")

class EducationalLLMWrapper:
    """Enhanced LLM wrapper with improved error handling."""

    def __init__(self):
        print("\nInitializing EducationalLLMWrapper")
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel('gemini-pro')
            print("Successfully initialized Gemini model")

            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
            print("Successfully initialized Cohere model")

            self.web_crawler = WebCrawler()
            print("Successfully initialized WebCrawler")

        except Exception as e:
            print(f"Error initializing EducationalLLMWrapper: {str(e)}")
            raise

    async def get_web_content(self, query: str) -> str:
        """Get content from web crawling."""
        results = await self.web_crawler.search(query)
        if not results:
            return ""

        content = "\n\n".join([
            f"Source: {result.url}\nTitle: {result.title}\nContent: {result.content}"
            for result in results
        ])
        return content

    async def _try_gemini_response(self, prompt: str) -> AsyncIterator[str]:
        """Attempt to get a response from Gemini."""
        chat = self.model.start_chat(history=[])

        try:
            response = chat.send_message(prompt)
            if response.text:
                yield response.text
        except Exception as e:
            print(f"Gemini error: {str(e)}")
            raise

    async def _try_cohere_response(self, prompt: str) -> str:
        """Get a response from Cohere as backup."""
        try:
            response = await asyncio.to_thread(
                self.cohere_client.generate,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            return response.generations[0].text
        except Exception as e:
            print(f"Cohere error: {str(e)}")
            raise

    async def generate_response(self, query: str, grade: int) -> AsyncIterator[str]:
        """Generate educational response with fallback options."""
        try:
            context = await self.get_web_content(query)
            education_level = EducationLevel.from_grade(grade)

            prompt = f"""
            As a helpful teacher for grade {grade} students, answer the following question.
            Adapt your response for:
            - Reading level: {education_level.reading_level}
            - Vocabulary level: {education_level.vocabulary_level}
            - Explanation style: {education_level.explanation_style}

            Context:
            {context}

            Question:
            {query}

            Provide an educational response appropriate for grade {grade}.
            """

            try:
                async for chunk in self._try_gemini_response(prompt):
                    yield chunk
            except Exception as e:
                print("Falling back to Cohere")
                try:
                    backup_response = await self._try_cohere_response(prompt)
                    yield backup_response
                except Exception as e:
                    yield "I apologize, but I'm having trouble generating a response. Please try again."

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            yield "I apologize, but I encountered an error. Please try again."

    async def close(self):
        """Clean up resources."""
        await self.web_crawler.close()

class StreamlitEducationalApp:
    """Streamlit interface for the Educational LLM Wrapper."""

    def __init__(self):
        self.llm = EducationalLLMWrapper()
        self.setup_page_config()
        self.initialize_session_state()
        self.create_custom_theme()

    def setup_page_config(self):
        try:
            st.set_page_config(
                page_title="Educational AI Assistant",
                page_icon="📚",
                layout="wide"
            )
        except:
            pass

    def create_custom_theme(self):
        st.markdown("""
            <style>
            .stTextInput > div > div > input {
                border-radius: 10px;
            }
            .stButton > button {
                border-radius: 10px;
                background-color: #4CAF50;
                color: white;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
            .stMarkdown {
                font-size: 16px;
            }
            </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_grade" not in st.session_state:
            st.session_state.current_grade = 6

    def setup_sidebar(self):
        with st.sidebar:
            st.title("Settings")
            st.session_state.current_grade = st.slider(
                "Select Grade Level",
                min_value=1,
                max_value=12,
                value=st.session_state.current_grade,
                help="Adjust the grade level to get age-appropriate responses"
            )

            education_level = EducationLevel.from_grade(st.session_state.current_grade)
            st.write("Current Education Level:")
            st.write(f"- Reading: {education_level.reading_level}")
            st.write(f"- Vocabulary: {education_level.vocabulary_level}")
            st.write(f"- Style: {education_level.explanation_style}")

            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

    def display_chat_interface(self):
        st.title("Educational AI Assistant 📚")
        st.write(f"Currently helping Grade {st.session_state.current_grade} students")

        for msg in st.session_state.messages:
            message(
                msg["content"],
                is_user=msg["role"] == "user",
                key=str(msg["timestamp"])
            )

    async def process_user_input(self, prompt: str):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.time()
        })

        placeholder = st.empty()
        full_response = ""

        try:
            async for response in self.llm.generate_response(prompt, st.session_state.current_grade):
                if response:
                    full_response = response
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            if full_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": time.time()
                })

        except Exception as e:
            print(f"Error: {str(e)}")
            placeholder.markdown("I apologize, but I encountered an error. Please try again.")

    def run(self):
        try:
            self.setup_sidebar()
            self.display_chat_interface()

            if prompt := st.chat_input("Ask me anything! I'm here to help you learn 🌟"):
                message(prompt, is_user=True)
                with st.chat_message("assistant"):
                    asyncio.run(self.process_user_input(prompt))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            if hasattr(self, 'llm'):
                asyncio.run(self.llm.close())

def main():
    try:
        app = StreamlitEducationalApp()
        app.run()
    except Exception as e:
        st.error("A fatal error occurred. Please check the logs and try again.")
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
