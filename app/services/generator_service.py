import asyncio
import base64
from openai import AsyncOpenAI
from app.services.openrag_client import get_openrag_client
from app.config import settings

# Brand-relevant search queries to pull holistic brand context
BRAND_CONTEXT_QUERIES = [
    "brand aesthetics visual style photography guidelines",
    "brand voice tone language copywriting rules",
    "brand refusals never do constraints prohibitions",
    "brand heritage origin identity core values",
    "brand taste product design sensibility material preferences",
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
                    "/v1/search",
                    json={"query": query, "limit": 5, "score_threshold": 0},
                )
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    return [r.get("text", "").strip() for r in results if r.get("text")]
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
        brand_guidelines = await self._fetch_brand_guidelines()

        enriched_prompt = f"""You are a professional creative director generating brand-compliant imagery.

=== BRAND GUIDELINES ===
{brand_guidelines}
=== END BRAND GUIDELINES ===

=== USER REQUEST ===
{user_prompt}
=== END USER REQUEST ===

Generate an image that faithfully executes the user request while strictly adhering to the brand guidelines above.
Ensure the result feels premium, authentic, and wholly aligned with the brand's aesthetic philosophy, voice, and refusals.
"""

        result = await self.openai_client.images.generate(
            model="gpt-image-2",
            prompt=enriched_prompt,
            n=1,
            size="1024x1024",
        )

        image_b64 = result.data[0].b64_json
        return {
            "image_base64": f"data:image/png;base64,{image_b64}",
            "brand_context_used": len(brand_guidelines) > 0,
        }


_generator_service_instance = None


def get_generator_service() -> GeneratorService:
    global _generator_service_instance
    if _generator_service_instance is None:
        _generator_service_instance = GeneratorService()
    return _generator_service_instance
