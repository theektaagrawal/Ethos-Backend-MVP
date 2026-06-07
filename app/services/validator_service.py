import os
import json
import base64
import asyncio
from openai import AsyncOpenAI
from app.services.openrag_client import get_openrag_client
from app.routers.knowledge import BRAND_PRIMITIVES

from app.config import settings

class ValidatorService:
    def __init__(self):
        self.openrag_client = get_openrag_client()
        
        api_key = settings.openai_api_key
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def _analyze_image_for_queries(self, image_base64: str, description: str) -> list[str]:
        prompt = f"""Analyze the visual contents, aesthetic, themes, and specific elements present in this draft image.
Description: {description}

Output ONLY a JSON object with a single key "queries" containing a list of 3-5 short search queries we can use to look up the brand's rules regarding these specific elements.
Example elements to notice: lighting style, presence of models/celebrities, specific colors, packaging types (ribbons, boxes), text elements, logos, etc.

Example Output:
{{
  "queries": ["celebrity endorsement", "bright studio lighting", "ribbon wrapping", "exclamation marks in text"]
}}
"""
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ]
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("queries", [])

    async def _get_targeted_brand_knowledge(self, queries: list[str]) -> str:
        """Gathers snippets from OpenRAG based on dynamic image analysis queries."""
        context_parts = []
        
        async def search_query(query):
            try:
                response = await self.openrag_client.client.post(
                    "/v1/search",
                    json={"query": query, "limit": 4, "score_threshold": 0},
                )
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    snippets = [item.get("text", "").strip() for item in results if item.get("text")]
                    return query, snippets
            except Exception:
                pass
            return query, []

        query_results = await asyncio.gather(*(search_query(q) for q in queries))
        
        for query, snippets in query_results:
            if snippets:
                context_parts.append(f"--- Context for: {query} ---")
                for snippet in snippets:
                    context_parts.append(snippet[:500]) # Keep it bounded

        return "\n".join(context_parts)

    async def audit_image_draft(self, image_base64: str, description: str):
        # Remove data:image/...;base64, prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Step 1: Analyze image to generate targeted queries
        queries = await self._analyze_image_for_queries(image_base64, description)
        
        # Step 2: Fetch targeted brand knowledge
        brand_knowledge = await self._get_targeted_brand_knowledge(queries)
        
        prompt = f"""You are a strict brand guardian and creative director for the house.
Your ONLY job is to audit the provided draft image against our strict brand rules and primitives. 
Do NOT invent generic fashion advice or marketing cliches. Every improvement or rejection MUST be directly rooted in a specific rule, refusal, or philosophy found in the Brand Knowledge Context below.

### Brand Knowledge Context:
{brand_knowledge}

### Draft Description from user:
{description}

Analyze the provided image. 
1. Does it violate any of the specific refusals, voice guidelines, or aesthetic rules in the context? (e.g., using banned words/elements, looking too commercial, complex "luxury unboxing" style, celebrity focus, etc.).
2. How can it be improved to strictly reflect the core principles (e.g., "Less, but for a longer time", silence, restraint)?

Return ONLY a JSON object with two keys:
- "improvements": A list of strings, each detailing a specific visual or thematic improvement to align with the brand. You MUST reference the specific brand rule you are applying.
- "rejections": A list of strings, each detailing a specific element in the draft that violates brand rules. You MUST reference the specific brand rule that is being violated.

Example Output:
{{
  "improvements": ["Remove the exclamation marks to align with the brand's 'What We Never Say' guidelines.", "Simplify the composition to reflect the 'Less, but for a longer time' philosophy."],
  "rejections": ["The image features a celebrity, which violates the 'Celebrity Dressing' refusal rule.", "The packaging looks too complex, conflicting with our restrained, durable packaging standards."]
}}
"""
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        content = response.choices[0].message.content
        return json.loads(content)

    async def apply_image_improvements(self, image_base64: str, description: str, improvements: list, rejections: list):
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        image_bytes = base64.b64decode(image_base64)
        
        prompt = f"Original Idea: {description}\n\n"
        
        if rejections:
            prompt += "THINGS TO AVOID (Rejections from previous draft):\n"
            for r in rejections:
                prompt += f"- {r}\n"
                
        if improvements:
            prompt += "\nMANDATORY IMPROVEMENTS to incorporate:\n"
            for i in improvements:
                prompt += f"- {i}\n"
                
        prompt += "\nPlease generate an image that faithfully executes the Original Idea while strictly adhering to the mandatory improvements and avoiding the rejections. Ensure a highly professional, brand-aligned aesthetic."

        result = await self.openai_client.images.edit(
            model="gpt-image-2",
            image=[("image.png", image_bytes)],
            prompt=prompt
        )
        
        image_base64 = result.data[0].b64_json
        return {"image_base64": f"data:image/png;base64,{image_base64}"}

_validator_service_instance = None

def get_validator_service() -> ValidatorService:
    global _validator_service_instance
    if _validator_service_instance is None:
        _validator_service_instance = ValidatorService()
    return _validator_service_instance
