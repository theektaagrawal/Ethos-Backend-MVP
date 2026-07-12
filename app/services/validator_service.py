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


def _normalize_source(name: str) -> str:
    """Normalize a filename for tolerant matching: lowercase, drop extension,
    drop parenthetical numbers like '(1)', keep alphanumerics only."""
    name = name.lower().strip()
    name = re.sub(r'\.\w+$', '', name)       # trailing extension
    name = re.sub(r'\(\s*\d+\s*\)', '', name)  # (1), (2)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name


def verify_citations(findings, valid_sources):
    """Keep only findings whose [Source: X] matches a filename actually returned
    by retrieval. Findings with no citation, or a citation that matches no
    retrieved source, are treated as ungrounded and dropped. Returns
    (kept_findings, dropped_findings)."""
    norm_valid = {_normalize_source(s) for s in valid_sources if s}
    kept, dropped = [], []
    for f in findings:
        m = re.search(r'\[Source:\s*([^\]]+)\]', f)
        cited = _normalize_source(m.group(1)) if m else ""
        matched = bool(cited) and any(
            cited == v or cited in v or v in cited for v in norm_valid
        )
        (kept if matched else dropped).append(f)
    return kept, dropped


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

    async def _get_targeted_brand_knowledge(self, queries: list[str], max_characters: int = 7000) -> str:
        """Gathers raw document chunks from OpenRAG based on image analysis queries."""
        async def search_query(query):
            try:
                response = await self.openrag_client.client.post(
                    "/v1/search",
                    json={"query": query, "limit": 4, "score_threshold": 0.3},
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
        total_characters = 0
        for query, snippets in query_results:
            if snippets:
                header = f"--- Context for: {query} ---"
                if total_characters + len(header) >= max_characters:
                    break
                context_parts.append(header)
                total_characters += len(header)
                for text, filename in snippets:
                    key = text[:120]
                    if key not in seen:
                        seen.add(key)
                        source = f"\n[Source: {filename}]" if filename else ""
                        snippet = f"{text[:450]}{source}"
                        if total_characters + len(snippet) > max_characters:
                            return "\n".join(context_parts)
                        context_parts.append(snippet)
                        total_characters += len(snippet)

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
        # Keep the evidence request bounded; additional broad searches create
        # duplicate chunks and inflate the audit prompt without improving grounding.
        all_queries = (queries[:4] + [q for q in primitive_queries if q not in queries])[:7]

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
                    return response.json().get("response", "").strip()[:3500]
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

        # No-context guard: findings must cite [Source:] from retrieved factual
        # chunks. With no citable brand rules, any "audit" would be the model
        # free-wheeling on generic design opinion — exactly what we must not do.
        # Refuse to produce findings rather than emit ungrounded ones.
        if not factual_chunks.strip():
            yield f"data: {json.dumps({'status': 'Complete', 'result': {'improvements': [], 'rejections': [], 'reviews': None, 'grounding_warning': 'No brand rules were retrieved from the knowledge base, so no grounded audit could be produced. Confirm OpenRAG has indexed the brand documents.'}})}\n\n"
            return

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
   - Observe and name the exact color of the logo as it actually appears in the image (e.g., red, green, blue, white, black — report only what you see).
   - Compare that observed color ONLY against the approved color versions explicitly stated in <brand_knowledge_context>. Do not assume any color is approved or forbidden unless the context says so.
   - Flag a rejection ONLY if the context explicitly states the observed treatment is not permitted; the corresponding improvement must change it to a color the context explicitly approves.
2. **Inspect Typography and Text overlays**:
   - Observe the fonts, capitalization, textures (e.g., distressed, clean), and placement as they appear.
   - Verify against the typographic rules stated in the context; do not apply typographic preferences that are not in the context.
3. **Inspect Layout, Sizing, and Safe Area**:
   - Observe the size and position of the logo, promotional badges, and other graphical overlays, and check them against any layout/safe-area rules stated in the context.
4. **Identify Violations & Formulate Improvements**:
   - Identify specific elements that violate a refusal, voice rule, color rule, or layout guideline that is explicitly present in the context.
   - Formulate highly specific, literal visual instructions for how the image MUST be edited to fix these violations.
   - Do NOT use generic creative-director feedback (like "Simplify the composition" or "Adjust color grade to be more natural"). Dictate exactly what must change, literally (e.g., "Change the top-left logo to an approved color stated in the context", "Remove the '2025' arrival badge from the top-right", "Remove the distressed/grunge effect from the text").
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

        # Citation verification: drop any finding whose [Source:] does not match a
        # filename actually retrieved from the knowledge base. This blocks
        # fabricated citations from passing as grounded brand findings.
        valid_sources = set(re.findall(r'\[Source:\s*([^\]]+)\]', factual_chunks))
        kept_imp, dropped_imp = verify_citations(audit_data.get('improvements', []), valid_sources)
        kept_rej, dropped_rej = verify_citations(audit_data.get('rejections', []), valid_sources)
        if settings.debug and (dropped_imp or dropped_rej):
            print(f"--- DEBUG: DROPPED UNGROUNDED FINDINGS (citation not in retrieved sources) ---")
            for f in dropped_imp + dropped_rej:
                print(f"  DROPPED: {f}")
            print(f"  valid_sources: {valid_sources}")
        audit_data['improvements'] = kept_imp
        audit_data['rejections'] = kept_rej

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

    async def apply_image_improvements(self, image_base64: str, description: str, improvements: list, rejections: list, brand_name: str = "McKINLEY", previous_response_id: str = None):
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

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
You are an expert prompt engineer for an image editing AI. Your job is to translate brand guidelines, required improvements, and rejections into an explicit, prioritized edit instruction for an image editor.
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
Write an edit instruction for the image editor that makes EVERY violation and improvement land. Keep it as concise as possible while including every mandatory edit; only the closing aesthetic paragraph should be trimmed for length. Structure it exactly like this, in this order:

1. Open with one sentence: "Apply every numbered edit below; all are mandatory. Change ONLY what these edits name and preserve every other element of the image exactly."
2. A numbered list of MECHANICAL edits, most consequential first. Each item must be concrete and spatial, naming what to change and where (e.g. "1. Change the red mountain logo in the top-left corner to the approved solid blue logo lockup."). Prioritize in this order: (a) removals/deletions of prohibited text, badges, or slogans; (b) text edits such as replacing or restyling a headline; (c) logo/color/trademark corrections; (d) layout and product-in-context fixes. State removals bluntly (e.g. "Completely remove the 'NEW ARRIVAL 2025' badge and the 'NO LIMITS' brush lettering; leave that area as clean background.").
3. A final PRESERVATION sentence phrased as "Change only what the numbered edits require and keep everything else identical." Explicitly name the important elements that must stay UNCHANGED — every element visible in <original_image_description> that no numbered edit touches (e.g. the product and the model, the price block, the feature bullet list, the logo). Then add these invariants, each scoped with "unless a numbered edit above requires it": do not alter the overall layout, composition, camera angle, cropping, color grade, or lighting; and never add new text, captions, watermarks, badges, or logos that no edit called for. The editor both drops unmentioned elements and invents new ones, so this clause is mandatory and specific.
4. Apply the brand's aesthetics from <brand_knowledge> IN SERVICE OF THE NUMBERED EDITS — no more, no less. If a numbered edit is itself aesthetic and global (e.g. a finding that the scene violates the brand's photography style — too staged, wrong lighting or mood), apply it fully, including relighting, re-staging, or re-grading the whole scene as that finding requires. But do NOT invent aesthetic changes that no finding calls for: never restyle, re-grade, or relight elements that no numbered edit touches.

Rules:
- Deletions and text edits are the highest priority — never omit or soften them to save space; drop aesthetic detail before dropping a mechanical edit.
- Only remove an element if a violation explicitly calls for its removal. Never remove price, product, feature bullets, logo, or supporting copy unless a numbered edit names it — list them in the preservation sentence instead.
- RESOLVE EVERY ABSTRACTION TO A CONCRETE VALUE. The image editor cannot interpret vague direction like "a product-specific headline" or "on-brand copy" — it can only render literal text. When an improvement is abstract, replace it with the exact literal string to render, inferred from the product and brand facts in <original_image_description> and <brand_knowledge> (e.g. turn "change to a product-specific headline" into: change the headline text to read "X-PRO 3L JACKET"). Never pass an abstract instruction through to the editor.
- Always wrap the exact in-image text to render in double quotes so the editor renders it verbatim and adds no extra characters (e.g. render the headline exactly as "X-PRO 3L JACKET").
- When brand rules specify a typeface, DO name it (e.g. "Work Sans Medium") AND add a short visible-style descriptor alongside it as a fallback (e.g. 'set the headline in Work Sans Medium — uppercase, clean geometric sans-serif, medium weight'). The name anchors the look for the editor; the descriptor covers the approximation.
- Do NOT include any citations or source names.
- Do not write anything outside of the final edit instruction itself.
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
            print(f"--- previous_response_id: {previous_response_id} ---")
            start_time = time.time()

        # Multi-turn, context-preserving edit via the Responses API image tool.
        # First turn sends the original image inline; subsequent turns omit it and
        # reference the model's own prior image through previous_response_id, which
        # preserves far more fidelity than re-uploading a flattened frame.
        if previous_response_id:
            user_content = [{"type": "input_text", "text": final_edit_prompt}]
        else:
            user_content = [
                {"type": "input_text", "text": final_edit_prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
            ]

        response = await self.openai_client.responses.create(
            model=settings.openai_responses_model,
            previous_response_id=previous_response_id,
            input=[{"role": "user", "content": user_content}],
            tools=[{"type": "image_generation"}],
        )

        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: RESPONSES EDIT CALL TOOK {elapsed:.2f} seconds ---")

        image_calls = [o for o in response.output if getattr(o, "type", None) == "image_generation_call"]
        new_image_base64 = image_calls[0].result if image_calls else None

        if not new_image_base64:
            yield f"data: {json.dumps({'status': 'Error', 'error': 'The image tool did not return an edited image.'})}\n\n"
            return

        yield f"data: {json.dumps({'status': 'Complete', 'result': {'image_base64': f'data:image/png;base64,{new_image_base64}', 'response_id': response.id}})}\n\n"

_validator_service_instance = None

def get_validator_service() -> ValidatorService:
    global _validator_service_instance
    if _validator_service_instance is None:
        _validator_service_instance = ValidatorService()
    return _validator_service_instance
