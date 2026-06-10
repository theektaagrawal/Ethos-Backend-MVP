import asyncio
import base64
import time
import json
from openai import AsyncOpenAI
from app.services.openrag_client import get_openrag_client
from app.config import settings

# Brand-relevant search queries to pull holistic brand context
BRAND_CONTEXT_QUERIES = [
    "What are the guidelines for the brand trademark and logo?",
    "What is the primary brand color palette?",
    "What are the brand aesthetics, visual style, and photography guidelines?",
    "What are the rules for brand voice, tone, language, and copywriting?",
    "What are the brand's 'never do's', constraints, and prohibitions?",
    "What is the brand's heritage, origin, identity, and core values?",
    "What are the preferences for brand taste, product design sensibility, and materials?",
]


class GeneratorService:
    def __init__(self):
        self.openrag_client = get_openrag_client()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _fetch_brand_guidelines(self) -> str:
        """Pulls broad brand context from OpenRAG across all brand primitives."""

        async def search(query: str):
            try:
                response = await self.openrag_client.client.post(
                    "/v1/chat",
                    json={"message": query},
                )
                if response.status_code == 200:
                    chat_response = response.json().get("response", "").strip()
                    if chat_response:
                        return [chat_response]
            except Exception:
                pass
            return []

        all_results = await asyncio.gather(*(search(q) for q in BRAND_CONTEXT_QUERIES))

        seen: set[str] = set()
        sections: list[str] = []
        for query, snippets in zip(BRAND_CONTEXT_QUERIES, all_results):
            unique_snippets = []
            for s in snippets:
                key = s[:120]
                if key not in seen:
                    seen.add(key)
                    unique_snippets.append(s[:500])
            if unique_snippets:
                sections.append(f"--- {query} ---")
                sections.extend(unique_snippets)

        return "\n\n".join(sections)

    async def generate_image(self, user_prompt: str) -> dict:
        """
        Enriches the user prompt with brand guidelines from OpenRAG,
        then generates an image with gpt-image-2.
        """
        yield f"data: {json.dumps({'status': 'Fetching brand guidelines from Ethos...'})}\n\n"
        brand_guidelines = await self._fetch_brand_guidelines()

        yield f"data: {json.dumps({'status': 'Generating image with brand context...'})}\n\n"

        enriched_prompt = f"""<role_definition>
You are a professional creative director generating brand-compliant imagery.
</role_definition>

<brand_guidelines>
{brand_guidelines}
</brand_guidelines>

<user_request>
{user_prompt}
</user_request>

<instructions>
Generate a photorealistic, high-fidelity image that faithfully executes the user request while strictly adhering to the brand guidelines above.
Structure your generation conceptually as: background/scene → subject → key details → constraints.
Ensure the result feels premium, authentic, and wholly aligned with the brand's aesthetic philosophy, voice, and refusals.
</instructions>
"""

        if settings.debug:
            print("--- DEBUG: GENERATOR SERVICE PROMPT ---")
            print(enriched_prompt)
            print("---------------------------------------")
            start_time = time.time()

        result = await self.openai_client.images.generate(
            model=settings.openai_image_model,
            prompt=enriched_prompt,
            n=1,
            size="1024x1024",
            quality="high",
        )

        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: GENERATOR API CALL TOOK {elapsed:.2f} seconds ---")

        image_b64 = result.data[0].b64_json
        yield f"data: {json.dumps({'status': 'Complete', 'result': {'image_base64': f'data:image/png;base64,{image_b64}', 'brand_context_used': len(brand_guidelines) > 0}})}\n\n"


_generator_service_instance = None


def get_generator_service() -> GeneratorService:
    global _generator_service_instance
    if _generator_service_instance is None:
        _generator_service_instance = GeneratorService()
    return _generator_service_instance
