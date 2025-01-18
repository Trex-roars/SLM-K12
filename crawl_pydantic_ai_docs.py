import os
import sys
import json
import asyncio
import requests
import cohere
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

load_dotenv()

# Initialize clients (Supabase, Cohere, Google Gemini)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash-8b')

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using Google Gemini."""
    prompt = f"""Extract a title and summary from this documentation chunk.
    URL: {url}

    Content:
    {chunk[:1000]}...

    Return your response strictly in this JSON format:
    {{
        "title": "brief but descriptive title",
        "summary": "concise summary of main points"
    }}
    """

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
            else:
                return {
                    "title": "Error processing title",
                    "summary": "Error processing summary"
                }

    except Exception as e:
        print(f"Error getting title and summary from Gemini: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary"
        }


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Cohere."""
    try:
        # Use embed-multilingual-v3.0 which outputs 1536 dimensions
        response = await asyncio.to_thread(
            cohere_client.embed,
            texts=[text],
            model='embed-multilingual-v3.0',
            input_type='search_document'
        )

        # Verify embedding dimensions
        embedding = response.embeddings[0]
        if len(embedding) != 1536:
            print(f"Warning: Got unexpected embedding dimension: {len(embedding)}")
            # Pad or truncate to 1536 dimensions if necessary
            if len(embedding) < 1536:
                embedding = embedding + [0] * (1536 - len(embedding))
            else:
                embedding = embedding[:1536]

        return embedding
    except Exception as e:
        print(f"Error getting embedding from Cohere: {e}")
        return [0] * 1536

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    try:
        extracted = await get_title_and_summary(chunk, url)
        embedding = await get_embedding(chunk)

        if len(embedding) != 1536:
            print(f"Warning: Incorrect embedding dimension {len(embedding)} for chunk {chunk_number}")

        metadata = {
            "source": "education_content",
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
            "embedding_dimension": len(embedding)  # Add dimension to metadata for debugging
        }

        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
    except Exception as e:
        print(f"Error processing chunk {chunk_number}: {e}")
        raise

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        if len(chunk.embedding) != 1536:
            print(f"Warning: Skipping chunk {chunk.chunk_number} due to incorrect embedding dimension: {len(chunk.embedding)}")
            return None

        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table("site_pages").insert(data).execute()
        print(f"Successfully inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk {chunk.chunk_number}: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    chunks = chunk_text(markdown)
    tasks = [
        process_chunk(chunk, i, url)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

def extract_urls_from_json(data: Dict[str, Any]) -> List[str]:
    """Extract all URLs from the nested JSON structure."""
    urls = []

    def process_subject(subject_data):
        if isinstance(subject_data, dict):
            if 'subject_url' in subject_data:
                urls.append(subject_data['subject_url'])
            if 'chapter_links' in subject_data:
                urls.extend(subject_data['chapter_links'])

            for value in subject_data.values():
                if isinstance(value, dict):
                    process_subject(value)

    for class_data in data['classes'].values():
        process_subject(class_data)

    return urls

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                try:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        await process_and_store_document(url, result.markdown_v2.raw_markdown)
                    else:
                        print(f"Failed: {url} - Error: {result.error_message}")
                except Exception as e:
                    print(f"Error processing {url}: {e}")

        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

async def main():
    with open("learncbse_links_deduped.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)

    urls = extract_urls_from_json(json_data)

    if not urls:
        print("No URLs found in the JSON structure")
        return

    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
