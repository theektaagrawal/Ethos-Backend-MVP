import os
import json
import base64
import asyncio
import time
import re
from openai import AsyncOpenAI
from app.services.openrag_client import get_openrag_client
from app.routers.knowledge import BRAND_PRIMITIVES

from app.config import settings

def replace_mckinley_brand(content, brand_name: str):
    if not brand_name:
        return content
    if isinstance(content, str):
        return re.sub(r'mckinley', brand_name, content, flags=re.IGNORECASE)
    elif isinstance(content, list):
        return [replace_mckinley_brand(item, brand_name) for item in content]
    elif isinstance(content, dict):
        return {k: replace_mckinley_brand(v, brand_name) for k, v in content.items()}
    return content


class ValidatorService:
    def __init__(self):
        self.openrag_client = get_openrag_client()
        
        api_key = settings.openai_api_key
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def _analyze_image_for_queries(self, image_base64: str, description: str) -> list[str]:
        prompt = f"""<role_definition>
You are a visual analyzer. Your job is to analyze the visual contents, aesthetic, themes, and specific elements present in this draft image to determine what brand rules we should look up.
</role_definition>

<draft_description>
{description}
</draft_description>

<instructions>
Analyze the visual details of the image. Identify key elements such as:
- Lighting style
- Presence of models or celebrities
- Specific colors
- Packaging types (ribbons, boxes, etc.)
- Text elements, logos, and voice
</instructions>

<structured_output_contract>
Output ONLY a JSON object with a single key "queries" containing a list of 3-5 keyword-rich search phrases optimized for semantic vector search (NOT natural language questions). Each phrase should be 4-8 words combining the visual element observed and the rule domain it falls under.

Example Output:
{{
  "queries": ["celebrity endorsement rules prohibitions refusals", "product photography lighting style guidelines", "ribbon packaging wrapping visual constraints", "exclamation marks punctuation copywriting voice rules"]
}}
</structured_output_contract>
"""
        if settings.debug:
            print("--- DEBUG: VALIDATOR QUERY PROMPT ---")
            print(prompt)
            print("-------------------------------------")
            start_time = time.time()

        response = await self.openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.0,
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
        
        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: VALIDATOR QUERY API CALL TOOK {elapsed:.2f} seconds ---")

        content = response.choices[0].message.content
        data = json.loads(content)
        
        if settings.debug:
            print(f"--- DEBUG: QUERIES EXTRACTED: {data.get('queries', [])} ---")
            
        return data.get("queries", [])

    async def _get_targeted_brand_knowledge(self, queries: list[str]) -> str:
        """Gathers raw document chunks from OpenRAG based on image analysis queries."""
        async def search_query(query):
            try:
                response = await self.openrag_client.client.post(
                    "/v1/search",
                    json={"query": query, "limit": 5, "score_threshold": 0.3},
                )
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    snippets = []
                    for r in results:
                        text = r.get("text", "").strip()
                        filename = r.get("filename", "")
                        if text:
                            snippets.append((text, filename))
                    return query, snippets
            except Exception:
                pass
            return query, []

        if settings.debug:
            start_time = time.time()

        query_results = await asyncio.gather(*(search_query(q) for q in queries))

        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: OPENRAG QUERIES TOOK {elapsed:.2f} seconds ---")

        seen: set[str] = set()
        context_parts: list[str] = []
        for query, snippets in query_results:
            if snippets:
                context_parts.append(f"--- Context for: {query} ---")
                for text, filename in snippets:
                    key = text[:120]
                    if key not in seen:
                        seen.add(key)
                        source = f"\n[Source: {filename}]" if filename else ""
                        context_parts.append(f"{text[:600]}{source}")

        return "\n".join(context_parts)

    async def audit_image_draft(self, image_base64: str, description: str, brand_name: str = "McKINLEY"):
        # Remove data:image/...;base64, prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Clean description
        description = replace_mckinley_brand(description, brand_name)

        yield f"data: {json.dumps({'status': 'Analyzing visual elements...'})}\n\n"
        # Step 1: Analyze image to generate targeted queries
        queries = await self._analyze_image_for_queries(image_base64, description)
        
        yield f"data: {json.dumps({'status': 'Consulting Loci Agent for brand rules...'})}\n\n"
        # Augment dynamic queries with systematic primitive coverage so the audit
        # always has broad brand context even when visual analysis misses a dimension.
        primitive_queries = [
            "brand refusals prohibitions never do constraints",
            "brand color palette approved colors logo usage",
            "brand typography fonts capitalization text rules",
            "brand aesthetic philosophy visual style photography",
        ]
        all_queries = queries + [q for q in primitive_queries if q not in queries]

        # Fetch two types of context in parallel:
        # 1. Raw chunks via /v1/search → real source citations
        # 2. A single /v1/chat call → LightRAG relational/thematic synthesis (no citation required)
        async def fetch_relational_context():
            try:
                response = await self.openrag_client.client.post(
                    "/v1/chat",
                    json={"message": "Summarize the brand's core aesthetic identity, key visual prohibitions, and defining design principles across all brand primitives."},
                )
                if response.status_code == 200:
                    return response.json().get("response", "").strip()
            except Exception:
                pass
            return ""

        factual_chunks_task = self._get_targeted_brand_knowledge(all_queries)
        relational_ctx, factual_chunks = await asyncio.gather(
            fetch_relational_context(), factual_chunks_task
        )

        sections = []
        if relational_ctx:
            sections.append(f"=== Brand Identity Summary (thematic, no citation required) ===\n{relational_ctx}")
        if factual_chunks:
            sections.append(f"=== Specific Brand Rules (cite [Source:] from these) ===\n{factual_chunks}")
        brand_knowledge = "\n\n".join(sections)

        # Clean brand_knowledge retrieved from database
        brand_knowledge = replace_mckinley_brand(brand_knowledge, brand_name)

        if not brand_knowledge.strip():
            yield f"data: {json.dumps({'status': 'Warning: No brand knowledge retrieved. Check that OpenRAG has indexed documents.'})}\n\n"

        yield f"data: {json.dumps({'status': 'Auditing against brand guidelines...'})}\n\n"
        
        prompt = f"""<role_definition>
You are a strict brand guardian and creative director. Your sole job is to audit the provided draft image against our strict brand rules and primitives.
</role_definition>

<grounding_rules>
- Do NOT invent generic fashion advice, creative feedback, or marketing clichés.
- Every suggested improvement or rejection MUST be directly rooted in a specific rule, refusal, or philosophy found in the <brand_knowledge_context> below.
- If the brand knowledge context is insufficient or irrelevant to an element in the image, do not create a rejection or improvement for that element.
</grounding_rules>

<missing_context_gating>
- If required brand context is missing to evaluate an element, do NOT guess.
- Label any assumptions explicitly.
</missing_context_gating>

<brand_knowledge_context>
{brand_knowledge}
</brand_knowledge_context>

<draft_description>
{description}
</draft_description>

<instructions>
Analyze the visual content of the provided image and its description. Perform a rigorous, element-by-element brand audit:
1. **Inspect the Logo/Trademark Color and Styling**:
   - Check the exact color of the logo in the image (e.g., is it red, green, blue, white, black?).
   - Cross-reference this color with the approved color versions in <brand_knowledge_context> (e.g., white, black, blue).
   - If the logo color violates the guidelines (such as using a red logo when only white, black, or blue are allowed), flag it as a rejection and add an improvement to change it to an approved color.
2. **Inspect Typography and Text overlays**:
   - Check the fonts, capitalization, textures (e.g., distressed, clean), and placement.
   - Verify against typographic rules in the context.
3. **Inspect Layout, Sizing, and Safe Area**:
   - Check the size and position of the logo, promotional badges, and other graphical overlays.
4. **Identify Violations & Formulate Improvements**:
   - Identify specific elements that violate any refusals, voice rules, color rules, or layout guidelines in the context.
   - Formulate highly specific, literal visual instructions for how the image MUST be edited to fix these violations.
   - Do NOT use generic creative-director feedback (like "Simplify the composition" or "Adjust color grade to be more natural"). Instead, dictate exactly what needs to change visually and literally (e.g., "Change the red logo in the top-left corner to white", "Remove the red '2025' arrival badge from the top-right", "Remove the distressed/grunge effect from the text").
</instructions>

<structured_output_contract>
Return ONLY a JSON object with two keys:
- "improvements": A list of strings, each providing a concrete, literal visual instruction for DALL-E to edit the image. You MUST cite the specific document from the context for each improvement (e.g., "Change the headline text to 'Escape to nature' using Work Sans Regular. [Source: BrandBook.pdf]").
- "rejections": A list of strings, each detailing a specific element in the draft that violates brand rules. You MUST cite the specific document from the context (e.g., "[Source: BrandBook.pdf]").

Do not add any prose or markdown formatting outside of the JSON object.
</structured_output_contract>

Example Output:
{{
  "improvements": ["Remove the exclamation marks from the text overlay. [Source: BrandVoice.pdf]", "Replace the complex packaging with a plain matte black box. [Source: PackagingStandards.pdf]"],
  "rejections": ["The image features a celebrity, which violates the 'Celebrity Dressing' refusal rule. [Source: Refusals.pdf]", "The packaging looks too complex, conflicting with our restrained, durable packaging standards. [Source: PackagingStandards.pdf]"]
}}
"""
        if settings.debug:
            print("--- DEBUG: AUDIT PROMPT ---")
            print(prompt)
            print("---------------------------")
            start_time = time.time()

        response = await self.openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.0,
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
        
        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: AUDIT API CALL TOOK {elapsed:.2f} seconds ---")
        
        content = response.choices[0].message.content
        audit_data = json.loads(content)

        yield f"data: {json.dumps({'status': 'Gathering stakeholder feedback...'})}\n\n"

        # Generate simulated stakeholder feedback based on the audit
        feedback_prompt = f"""<role_definition>
You are simulating the reactions of three key stakeholders to a drafted brand image that has just been audited.
</role_definition>

<context>
Original Description: {description}
Violations found: {audit_data.get('rejections', [])}
Suggested Improvements: {audit_data.get('improvements', [])}
</context>

<instructions>
Provide a short, 1-2 sentence realistic, in-character reaction from each of these three roles reacting to the draft's flaws. They should sound like they are reviewing the draft and agreeing with the audit findings.
- Founder: Focuses on core ethos, mission, and long-term brand legacy.
- CBO (Chief Brand Officer): Focuses on alignment with brand guidelines, color palettes, and structural correctness.
- Brand Critic: A slightly skeptical external or internal voice who is hard to please, focusing on avoiding genericness, contrast, and subtle aesthetic nuances.
</instructions>

<structured_output_contract>
Output ONLY a JSON object with the exact keys: "founder", "cbo", "brand_critic". Do not add any markdown formatting or explanation outside the JSON.
</structured_output_contract>
"""
        if settings.debug:
            print("--- DEBUG: STAKEHOLDER FEEDBACK PROMPT ---")
            print(feedback_prompt)
            print("------------------------------------------")
            start_time = time.time()

        feedback_response = await self.openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": feedback_prompt}]
        )

        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: STAKEHOLDER FEEDBACK API CALL TOOK {elapsed:.2f} seconds ---")

        feedback_data = json.loads(feedback_response.choices[0].message.content)
        audit_data["reviews"] = {
            "Founder": feedback_data.get("founder", "This draft misses our core ethos. Let's apply these fixes."),
            "CBO": feedback_data.get("cbo", "The guidelines are clear. We need to correct these structural issues immediately."),
            "Brand Critic": feedback_data.get("brand_critic", "Too generic. Hopefully the improvements will give it some actual character.")
        }

        # Recursively replace any leaked "McKINLEY" text with custom brand name
        audit_data = replace_mckinley_brand(audit_data, brand_name)

        if settings.debug:
            print(f"--- DEBUG: FULL AUDIT RESULT: {json.dumps(audit_data, indent=2)} ---")

        yield f"data: {json.dumps({'status': 'Complete', 'result': audit_data})}\n\n"

    async def apply_image_improvements(self, image_base64: str, description: str, improvements: list, rejections: list, brand_name: str = "McKINLEY"):
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        image_bytes = base64.b64decode(image_base64)
        
        # Clean input text variables
        description = replace_mckinley_brand(description, brand_name)
        improvements = replace_mckinley_brand(improvements, brand_name)
        rejections = replace_mckinley_brand(rejections, brand_name)

        rejections_str = "\n".join(f"- {r}" for r in rejections) if rejections else "None"
        improvements_str = "\n".join(f"- {i}" for i in improvements) if improvements else "None"

        yield f"data: {json.dumps({'status': 'Consulting Loci Agent for brand rules...'})}\n\n"
        # Build queries from actual violations and improvements found in the audit
        # so the synthesis step retrieves relevant rules, not generic buckets.
        violation_queries = []
        for item in (rejections + improvements)[:6]:
            # Strip source citations from the text before using as a query
            clean = re.sub(r'\[Source:[^\]]+\]', '', item).strip()
            if clean:
                violation_queries.append(clean[:200])
        fallback_queries = [
            "brand trademark logo color typography rules",
            "brand aesthetic philosophy visual style photography",
            "brand refusals prohibitions constraints",
        ]
        queries = violation_queries if violation_queries else fallback_queries
        brand_knowledge = await self._get_targeted_brand_knowledge(queries)
        # Clean brand_knowledge retrieved from database
        brand_knowledge = replace_mckinley_brand(brand_knowledge, brand_name)

        yield f"data: {json.dumps({'status': 'Synthesizing visual edit instructions...'})}\n\n"

        synthesis_prompt = f"""<role_definition>
You are an expert prompt engineer for an image editing AI. Your job is to translate complex brand guidelines, required improvements, and rejections into a single, concise paragraph of edit instructions.
</role_definition>

<original_image_description>
{description}
</original_image_description>

<brand_knowledge>
{brand_knowledge}
</brand_knowledge>

<improvements_to_apply>
{improvements_str}
</improvements_to_apply>

<violations_to_remove>
{rejections_str}
</violations_to_remove>

<instructions>
Write a cohesive prompt for the image editor (maximum 800 characters) that tells it exactly what to change to fix the violations, while heavily injecting the brand's unique aesthetics, lighting, mood, and visual style.
- Include concrete, specific, and direct instructions for the mechanical edits (e.g., "Change the red logo in the top-left corner to a clean solid white logo").
- Explicitly integrate the brand's required aesthetics, mood, and photography style from the <brand_knowledge> to ensure the final image maintains the premium brand nuance.
- Do NOT include any citations or source names.
- Do not write anything outside of the final prompt itself.
</instructions>
"""

        synthesis_response = await self.openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.0
        )
        
        final_edit_prompt = synthesis_response.choices[0].message.content.strip()
        # Ensure the final edit instruction to DALL-E uses the custom brand name
        final_edit_prompt = replace_mckinley_brand(final_edit_prompt, brand_name)

        yield f"data: {json.dumps({'status': 'Generating surgically improved draft...'})}\n\n"

        if settings.debug:
            print("--- DEBUG: SYNTHESIZED IMAGE EDIT PROMPT ---")
            print(final_edit_prompt)
            print("--------------------------------------------")
            start_time = time.time()

        result = await self.openai_client.images.edit(
            model=settings.openai_image_model,
            image=[("image.png", image_bytes)],
            prompt=final_edit_prompt
        )
        
        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: IMAGE EDIT API CALL TOOK {elapsed:.2f} seconds ---")
            
        new_image_base64 = result.data[0].b64_json

        yield f"data: {json.dumps({'status': 'Complete', 'result': {'image_base64': f'data:image/png;base64,{new_image_base64}'}})}\n\n"

_validator_service_instance = None

def get_validator_service() -> ValidatorService:
    global _validator_service_instance
    if _validator_service_instance is None:
        _validator_service_instance = ValidatorService()
    return _validator_service_instance
