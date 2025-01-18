from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import cohere
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from supabase import Client, create_client

load_dotenv()

# Initialize clients with error handling
def init_cohere_client() -> Optional[cohere.Client]:
    try:
        return cohere.Client(os.getenv("COHERE_API_KEY"))
    except Exception as e:
        print(f"Error initializing Cohere client: {e}")
        return None

def init_gemini_model() -> Optional[Any]:
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel('gemini-1.5-flash-8b')
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return None

cohere_client = init_cohere_client()
model_gemini = init_gemini_model()
logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AIAgentDeps:
    supabase: Client

class DocumentationAgent:
    def __init__(self, deps: AIAgentDeps):
        self.deps = deps
        self.fallback_cache = {}  # Simple cache for fallback responses

    async def get_embedding(self, text: str) -> Tuple[List[float], bool]:
        """Get embedding vector from Cohere with fallback."""
        if not cohere_client:
            return [0] * 1536, False

        try:
            response = await asyncio.to_thread(
                cohere_client.embed,
                texts=[text],
                model='embed-multilingual-v3.0',
                input_type='search_document'
            )
            embedding = response.embeddings[0]
            if len(embedding) != 1536:
                embedding = self._normalize_embedding(embedding)
            return embedding, True
        except Exception as e:
            print(f"Error getting embedding from Cohere: {e}")
            return [0] * 1536, False

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to required dimension."""
        if len(embedding) < 1536:
            return embedding + [0] * (1536 - len(embedding))
        return embedding[:1536]

    async def generate_content(self, prompt: str, attempt_structured: bool = False) -> str:
        """Generate content using Gemini with enhanced capabilities."""
        if not model_gemini:
            return "AI model unavailable. Please try again later."

        try:
            if attempt_structured:
                structured_prompt = f"""
                Please provide a structured response with the following:
                1. Main points and key concepts
                2. Technical details and implementation notes
                3. Practical examples or use cases
                4. Best practices and recommendations

                Question/Topic: {prompt}
                """
                response = await asyncio.to_thread(
                    model_gemini.generate_content,
                    structured_prompt
                )
            else:
                response = await asyncio.to_thread(
                    model_gemini.generate_content,
                    prompt
                )
            return response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return await self._get_fallback_response(prompt)

    async def _get_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response using Gemini's knowledge."""
        try:
            fallback_prompt = f"""
            As an AI assistant, provide a helpful response about:
            {prompt}

            Focus on providing general knowledge and best practices,
            even without access to specific documentation.
            """
            if prompt in self.fallback_cache:
                return self.fallback_cache[prompt]

            response = await asyncio.to_thread(
                model_gemini.generate_content,
                fallback_prompt
            )
            self.fallback_cache[prompt] = response.text
            return response.text
        except Exception:
            return "Unable to generate response. Please try again later."

async def retrieve_relevant_documentation(self, user_query: str) -> str:
         """Enhanced documentation retrieval with improved fallback to Gemini."""
         try:
             query_embedding, embedding_success = await self.get_embedding(user_query)

             if embedding_success:
                 result = self.deps.supabase.rpc(
                     'match_site_pages',
                     {
                         'query_embedding': query_embedding,
                         'match_count': 5,
                         'filter': {'source': 'pydantic_ai_docs'}
                     }
                 ).execute()

                 if result.data:
                     return await self._format_documentation_results(result.data)

                 print(query_embedding)
             # Enhanced fallback to Gemini-generated response
             fallback_prompt = f"""
             As a technical documentation assistant, provide detailed technical information about:
             {user_query}

             If the query is about technical implementation or coding:
             1. Provide code examples
             2. Explain implementation details
             3. Offer best practices
             4. Include common pitfalls and solutions

             If the query is about concepts or theory:
             1. Provide detailed technical explanations
             2. Include relevant examples
             3. Explain related concepts
             4. Reference industry standards or common practices

             Focus on being specific and technical rather than saying the information isn't available.
             """

             generated_response = await self.generate_content(fallback_prompt, attempt_structured=True)

             # Format the response to look like documentation
             return f"""
 # Generated Technical Documentation

 ## Overview
 {generated_response}

 Note: This information is generated based on the AI model's knowledge. For the most up-to-date and accurate information, please refer to official documentation or technical resources.
 """
         except Exception as e:
             print(f"Error retrieving documentation: {e}")
             return await self._get_fallback_response(user_query)

     async def _get_fallback_response(self, prompt: str) -> str:
         """Generate a more technical fallback response."""
         try:
             if prompt in self.fallback_cache:
                 return self.fallback_cache[prompt]

             fallback_prompt = f"""
             Provide a detailed technical response about:
             {prompt}

             Include:
             1. Technical specifications and details
             2. Implementation considerations
             3. Best practices and recommendations
             4. Examples and use cases

             Format the response in a clear, structured manner.
             """

             response = await asyncio.to_thread(
                 model_gemini.generate_content,
                 fallback_prompt
             )

             formatted_response = f"""
 # Technical Documentation

 ## Detailed Information
 {response.text}

 Note: This is AI-generated technical documentation. Please verify critical information from official sources.
 """

             self.fallback_cache[prompt] = formatted_response
             return formatted_response
         except Exception:
             return "Unable to generate technical documentation. Please try again later."

    async def _format_documentation_results(self, docs: List[Dict[str, Any]]) -> str:
        """Format documentation results with enhanced summaries."""
        formatted_chunks = []
        for doc in docs:
            summary = await self.generate_content(
                f"Provide a technical summary of:\n{doc['content']}\n"
                "Include key concepts, implementation details, and practical examples."
            )
            chunk_text = f"""
# {doc['title']}
## Summary
{summary}

## Original Content
{doc['content']}
"""
            formatted_chunks.append(chunk_text)
        return "\n\n---\n\n".join(formatted_chunks)

    async def list_documentation_pages(self) -> List[str]:
        """List documentation pages with fallback."""
        try:
            result = self.deps.supabase.from_('site_pages') \
                .select('url') \
                .eq('metadata->>source', 'pydantic_ai_docs') \
                .execute()

            if result.data:
                return sorted(set(doc['url'] for doc in result.data))

            # Fallback to cached or generated content
            return ["No documentation pages available. Using AI-generated responses."]
        except Exception as e:
            print(f"Error listing documentation pages: {e}")
            return ["Documentation system temporarily unavailable."]

    async def get_page_content(self, url: str) -> str:
        """Get page content with enhanced fallback mechanism."""
        try:
            result = self.deps.supabase.from_('site_pages') \
                .select('title, content, chunk_number') \
                .eq('url', url) \
                .eq('metadata->>source', 'pydantic_ai_docs') \
                .order('chunk_number') \
                .execute()

            if not result.data:
                return await self._generate_synthetic_page_content(url)

            return await self._format_page_content(result.data)
        except Exception as e:
            print(f"Error retrieving page content: {e}")
            return await self._generate_synthetic_page_content(url)

    async def _generate_synthetic_page_content(self, url: str) -> str:
        """Generate synthetic page content when actual content is unavailable."""
        topic = url.split('/')[-1].replace('-', ' ').title()
        return await self.generate_content(
            f"Create comprehensive documentation for: {topic}\n"
            "Include detailed explanations, examples, and best practices.",
            attempt_structured=True
        )

    async def _format_page_content(self, chunks: List[Dict[str, Any]]) -> str:
        """Format page content with enhanced structure."""
        page_title = chunks[0]['title'].split(' - ')[0]
        full_content = "\n\n".join(chunk['content'] for chunk in chunks)

        outline = await self.generate_content(
            f"Create a detailed outline for:\nTitle: {page_title}\nContent: {full_content[:2000]}...",
            attempt_structured=True
        )

        formatted_content = [
            f"# {page_title}",
            "## Table of Contents",
            outline,
            "## Detailed Documentation",
            full_content,
            "\n## Additional Resources and Examples",
            await self.generate_content(f"Provide practical examples and use cases for: {page_title}")
        ]

        return "\n\n".join(formatted_content)

async def main():
    # Example initialization with error handling
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not all([supabase_url, supabase_key]):
            raise ValueError("Missing Supabase credentials")

        supabase_client = create_client(supabase_url, supabase_key)
        deps = AIAgentDeps(supabase=supabase_client)
        agent = DocumentationAgent(deps)

        # Example usage with error handling
        pages = await agent.list_documentation_pages()
        print("Available pages:", pages)

        if pages:
            content = await agent.get_page_content(pages[0])
            print("\nFirst page content:", content)

        docs = await agent.retrieve_relevant_documentation("How to use agents?")
        print("\nRelevant documentation:", docs)

    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
