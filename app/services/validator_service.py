import json
import asyncio
import time
import re
from openai import AsyncOpenAI
from app.services.openrag_client import get_openrag_client

from app.config import settings

def _vlog(title: str, content=""):
    """Unconditional step logging for the validator pipeline. Everything the audit
    retrieves, decides, and sends to the editor lands in stdout (docker logs folio),
    so a bad result can be diagnosed from the exact intermediate artifacts."""
    print(f"\n===== [VALIDATOR] {title} =====", flush=True)
    if content:
        print(content, flush=True)


def replace_mckinley_brand(content, brand_name: str):
    if not brand_name:
        return content
    if isinstance(content, str):
        # Lambda replacement so brand names containing regex escapes (\, $1...) stay literal.
        return re.sub(r'mckinley', lambda _: brand_name, content, flags=re.IGNORECASE)
    elif isinstance(content, list):
        return [replace_mckinley_brand(item, brand_name) for item in content]
    elif isinstance(content, dict):
        return {k: replace_mckinley_brand(v, brand_name) for k, v in content.items()}
    return content


def _split_data_url(image_base64: str) -> tuple[str, str]:
    """Return (mime, payload). Accepts a full data URL or raw base64 (assumed JPEG)."""
    if "," in image_base64:
        header, payload = image_base64.split(",", 1)
        m = re.match(r"data:(image/\w+);base64", header)
        return (m.group(1) if m else "image/jpeg"), payload
    return "image/jpeg", image_base64


def _normalize_source(name: str) -> str:
    """Normalize a filename for tolerant matching: lowercase, drop extension,
    drop parenthetical numbers like '(1)', keep alphanumerics only."""
    name = name.lower().strip()
    name = re.sub(r'\.\w+$', '', name)       # trailing extension
    name = re.sub(r'\(\s*\d+\s*\)', '', name)  # (1), (2)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name


def _extract_cited_sources(text: str) -> set[str]:
    """Collect document names cited as [Source: X] or (Source: X)."""
    return set(re.findall(r'[\[\(]Source:\s*([^\]\)]+)[\]\)]', text))


def _strip_citation(text: str) -> str:
    text = re.sub(r'\s*\[Source:[^\]]*\]', '', text)
    text = re.sub(r'\s*\(Source:[^)]*\)', '', text)
    return text.strip()


def _clip_at_sentence(text: str, limit: int) -> str:
    """Clip to `limit` chars, preferring a sentence boundary so a rule's operative
    clause isn't cut mid-sentence."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > limit * 0.5 else cut


def verify_findings(findings, valid_sources):
    """Deterministic validation of structured audit findings. Keeps a finding only if:
    - element/violation/fix are non-empty,
    - verdict is exactly 'restyle' or 'remove',
    - the element wasn't already given a verdict (first one wins — no contradictions),
    - its cited source matches a document actually returned by retrieval.
    Returns (kept, dropped)."""
    norm_valid = {_normalize_source(s) for s in valid_sources if s}
    kept, dropped = [], []
    seen_elements: set[str] = set()
    for f in findings if isinstance(findings, list) else []:
        if not isinstance(f, dict):
            dropped.append(f)
            continue
        element = str(f.get("element", "")).strip()
        verdict = str(f.get("verdict", "")).strip().lower()
        violation = str(f.get("violation", "")).strip()
        fix = str(f.get("fix", "")).strip()
        cited = _normalize_source(str(f.get("source", "")))
        ok = (
            element and violation and fix
            and verdict in ("restyle", "remove")
            and element.lower() not in seen_elements
            and bool(cited)
            and any(cited == v or cited in v or v in cited for v in norm_valid)
        )
        if ok:
            seen_elements.add(element.lower())
            kept.append({
                "element": element,
                "verdict": verdict,
                "violation": violation,
                "fix": fix,
                "source": str(f.get("source", "")).strip(),
            })
        else:
            dropped.append(f)
    return kept, dropped


def build_edit_prompt(fixes: list[str], preserve: list[str]) -> str:
    """Assemble the image-editor instruction deterministically. Every line traces to a
    verified finding; the preserve list is the audit's actual element inventory. The
    structure and lock language follow OpenAI's image-gen prompting guide: state the
    deliverable, skimmable numbered edits, explicit invariants, verbatim-text and
    person-identity locks, repeated on every iteration."""
    lines = [
        "You are editing a professional brand marketing advertisement. Apply every numbered edit below; all are mandatory. Change ONLY what these edits name and preserve every other element of the image exactly.",
        "",
    ]
    for i, fix in enumerate(fixes, 1):
        lines.append(f"{i}. {fix}")
    lines.append("")
    if preserve:
        lines.append(
            "Keep these elements exactly as they are, completely unchanged: "
            + "; ".join(preserve) + "."
        )
    lines.extend([
        "If people appear, keep each person's face, body shape, pose, hair, and expression exactly unchanged unless a numbered edit targets them.",
        "Render every quoted string verbatim, character for character — no extra characters, no omissions, no respelling.",
        "When an edit changes text or its styling, keep that text block's position, size, and alignment unchanged unless the edit states otherwise.",
        "Do not alter the overall layout, composition, camera angle, cropping, color grade, "
        "saturation, contrast, or lighting unless a numbered edit above explicitly requires it. "
        "Never add new text, captions, badges, watermarks, or logos that no numbered edit calls for.",
    ])
    return "\n".join(lines)


# Fixed retrieval queries covering every rule domain the audit inspects. A fixed set is
# deliberately chosen over LLM-generated queries: retrieval becomes repeatable across
# runs of the same knowledge base, costs one less vision call, and no rule domain
# depends on what a query-writing model happened to notice in the image.
AUDIT_RULE_QUERIES = [
    "logo trademark approved color versions usage rules",
    "brand color palette approved primary colors",
    "typography typeface font capitalization hierarchy rules",
    "photography imagery style staging lighting composition rules",
    "brand refusals prohibitions never do constraints",
    "layout safe area overlay badge placement rules",
    "brand voice tone copywriting language rules",
]

# One fixed question for the agentic OpenRAG flow (which routes between chunk search and
# the LightRAG relational graph). Fixed for the same repeatability reason, and phrased to
# elicit rule statements with citations so its content is groundable.
RELATIONAL_CONTEXT_QUERY = (
    "List the brand's explicit visual identity rules: approved logo colors and usage, "
    "approved color palette, typography rules, photography and imagery rules, layout and "
    "overlay rules, and hard refusals (things the brand must never do). Quote each rule "
    "and cite its source document."
)


class ValidatorService:
    def __init__(self):
        self.openrag_client = get_openrag_client()

        api_key = settings.openai_api_key
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def _get_targeted_brand_knowledge(self, queries: list[str], max_characters: int = 9000) -> str:
        """Gathers raw document chunks from OpenRAG for the fixed rule-domain queries.
        Snippets are interleaved round-robin across queries so no rule domain is starved
        by an earlier verbose one, and clipped at sentence boundaries."""
        async def search_query(query):
            try:
                response = await self.openrag_client.client.post(
                    "/v1/search",
                    json={"query": query, "limit": 4, "score_threshold": 0},
                )
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    snippets = []
                    for r in results:
                        text = r.get("text", "").strip()
                        filename = r.get("filename", "")
                        if text:
                            snippets.append((text, filename))
                    return snippets
            except Exception:
                pass
            return []

        if settings.debug:
            start_time = time.time()

        query_results = await asyncio.gather(*(search_query(q) for q in queries))

        if settings.debug:
            elapsed = time.time() - start_time
            print(f"--- DEBUG: OPENRAG QUERIES TOOK {elapsed:.2f} seconds ---")

        seen: set[str] = set()
        parts: list[str] = []
        total = 0
        max_rank = max((len(s) for s in query_results), default=0)
        for rank in range(max_rank):
            for snippets in query_results:
                if rank >= len(snippets):
                    continue
                text, filename = snippets[rank]
                key = text[:160]
                if key in seen:
                    continue
                seen.add(key)
                source = f"\n[Source: {filename}]" if filename else ""
                entry = f"{_clip_at_sentence(text, 700)}{source}"
                if total + len(entry) > max_characters:
                    return "\n\n".join(parts)
                parts.append(entry)
                total += len(entry)
        return "\n\n".join(parts)

    async def _get_relational_context(self) -> str:
        """One call into the agentic OpenRAG flow, which decides between chunk search and
        the LightRAG relational graph. Its citations are parsed and count as valid
        grounding sources for findings."""
        try:
            response = await self.openrag_client.client.post(
                "/v1/chat",
                json={"message": RELATIONAL_CONTEXT_QUERY},
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()[:3500]
        except Exception:
            pass
        return ""

    async def audit_image_draft(self, image_base64: str, description: str, brand_name: str = "McKINLEY", previous_fixes: list[str] | None = None):
        mime, image_payload = _split_data_url(image_base64)

        # Clean description
        description = replace_mckinley_brand(description, brand_name)

        _vlog("AUDIT START", f"brand={brand_name} mime={mime} description={description!r} previous_fixes={len(previous_fixes or [])}")
        if previous_fixes:
            _vlog("PREVIOUS FIXES (iteration memory)", "\n".join(f"- {f}" for f in previous_fixes))

        yield f"data: {json.dumps({'status': 'Consulting brand knowledge base...'})}\n\n"

        retrieval_start = time.time()
        relational_ctx, factual_chunks = await asyncio.gather(
            self._get_relational_context(),
            self._get_targeted_brand_knowledge(AUDIT_RULE_QUERIES),
        )
        _vlog(f"RETRIEVAL DONE in {time.time() - retrieval_start:.2f}s",
              f"relational_ctx chars={len(relational_ctx)} factual_chunks chars={len(factual_chunks)}")
        _vlog("RELATIONAL CONTEXT (/v1/chat -> LightRAG-aware agent)", relational_ctx or "(empty)")
        _vlog("FACTUAL CHUNKS (/v1/search)", factual_chunks or "(empty)")

        sections = []
        if relational_ctx:
            sections.append(f"=== Brand Rules (relational graph synthesis) ===\n{relational_ctx}")
        if factual_chunks:
            sections.append(f"=== Brand Rules (document excerpts) ===\n{factual_chunks}")
        brand_knowledge = replace_mckinley_brand("\n\n".join(sections), brand_name)

        # No-context guard: findings must cite retrieved sources. With no rules retrieved,
        # any "audit" would be generic design opinion — refuse rather than free-wheel.
        if not factual_chunks.strip() and not relational_ctx.strip():
            yield f"data: {json.dumps({'status': 'Complete', 'result': {'improvements': [], 'rejections': [], 'compliant': None, 'findings': [], 'preserve': [], 'reviews': None, 'grounding_warning': 'No brand rules were retrieved from the knowledge base, so no grounded audit could be produced. Confirm OpenRAG has indexed the brand documents.'}})}\n\n"
            return

        # Iteration memory: elements already fixed in a prior round must not be re-flagged
        # for the same rule, or the audit->apply loop oscillates forever.
        previous_block = ""
        if previous_fixes:
            fixes_lines = "\n".join(f"- {_strip_citation(f)}" for f in previous_fixes[:10])
            previous_block = f"""
<previous_round_edits>
The following edits were already applied to produce the draft you are inspecting:
{fixes_lines}
For any element one of these edits already addressed, re-flag it ONLY if you can name the specific visible evidence that it still violates the rule (e.g. "the headline still shows a distressed grunge texture"). If compliance is ambiguous at the pixel level — such as a dark logo that plausibly is the approved dark blue, or clean sans-serif text that plausibly is the documented typeface — treat the element as COMPLIANT and preserve it. An element is never a violation merely because it was edited in a previous round.
</previous_round_edits>
"""

        yield f"data: {json.dumps({'status': 'Auditing against brand guidelines...'})}\n\n"

        prompt = f"""<role_definition>
You are a brand compliance auditor. You inspect a draft marketing image against the brand's documented rules and produce a structured, machine-actionable audit.
</role_definition>

<grounding_rules>
- Every finding MUST be grounded in a specific rule stated in <brand_knowledge_context>. Never apply generic design opinion or taste.
- If the context contains no rule covering an element, that element is COMPLIANT by default — do not flag it.
- If the draft violates nothing, return an empty findings list. A clean audit is a valid and expected outcome; do not manufacture findings to appear thorough.
</grounding_rules>

<verifiability_rules>
Only flag what you can actually SEE in the image. Never flag an element because a property cannot be positively confirmed from a rendered image.
- Typeface identity is NOT verifiable from pixels: never flag text merely because you "cannot confirm" it is the documented typeface. Flag typography only on a VISIBLE deviation — wrong capitalization, or a decorative, script, serif, distressed, or hand-drawn treatment that clearly is not the brand's clean sans-serif.
- Colors must be judged with perceptual tolerance: approved brand colors can look close to other colors in a photo (a very dark approved blue reads as near-black). If an element's color plausibly matches an approved value, it is compliant; flag color only on a clear mismatch (e.g. red where only blue, black, or white are approved).
- Each violation must name the visible evidence ("the headline letterforms have a distressed grunge texture"), not an unverifiable assertion ("the text is not in the documented typeface system").
</verifiability_rules>

<brand_knowledge_context>
{brand_knowledge}
</brand_knowledge_context>

<campaign_context>
{description}
</campaign_context>
{previous_block}
<instructions>
Work in two passes.

PASS 1 — INVENTORY. List every distinct visual element in the image: each logo/trademark, each text block (headline, supporting copy, price, spec/feature lists), each badge or overlay, the product(s), any people, and the scene/background. Give each a short spatial name (e.g. "top-left logo", "price block bottom-left", "headline").

PASS 2 — AUDIT. Check each inventoried element against the explicit rules in <brand_knowledge_context>:
- An element receives AT MOST ONE finding, with exactly one verdict:
  - "restyle": the element stays but must change (color, typeface, wording, treatment). The fix must state the exact target using literal values from the context — the exact approved color name, the exact typeface, the exact replacement text in double quotes. When the target is a color, give a plain-language description alongside any code, because the image editor acts on descriptions, not hex values (e.g. 'the approved brand blue #002539 — a very dark navy, almost black').
  - "remove": the element is prohibited by a rule in the context and must be deleted entirely.
- Never issue two findings for the same element, and never restyle an element you also consider prohibited — decide which single verdict the rules actually support.
- When a fix changes photographic style or scene staging, describe the target in concrete photographic language — lighting direction and quality, candid/unposed framing, natural textures — not abstract adjectives (write "relight with soft natural daylight, candid unposed moment, no dramatic sunset grading", not "make it more authentic").
- Every element with no finding goes into "preserve" — this list is the editor's manifest of what must not change, so it must be complete.
- Each fix must be a self-contained, concrete, spatial instruction an image editor can execute with no other context.
</instructions>

<structured_output_contract>
Return ONLY a JSON object:
{{
  "compliant": true or false,
  "findings": [
    {{
      "element": "short spatial name from your inventory",
      "verdict": "restyle" or "remove",
      "violation": "which documented rule is broken and how",
      "fix": "literal, self-contained edit instruction",
      "source": "exact source document name from the context"
    }}
  ],
  "preserve": ["every inventoried element that has no finding"]
}}
Do not add any prose or markdown outside the JSON object.
</structured_output_contract>
"""
        _vlog("AUDIT PROMPT (sent to vision model)", prompt)
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
                                "url": f"data:{mime};base64,{image_payload}"
                            }
                        }
                    ]
                }
            ]
        )

        _vlog(f"AUDIT MODEL CALL took {time.time() - start_time:.2f}s")
        _vlog("RAW AUDIT MODEL OUTPUT", response.choices[0].message.content)

        try:
            audit_data = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            _vlog("AUDIT PARSE FAILED — malformed JSON from model")
            yield f"data: {json.dumps({'status': 'Error', 'error': 'The audit model returned malformed output. Please retry.'})}\n\n"
            return

        # Deterministic validation: schema, single-verdict-per-element, and citation
        # verification against sources actually returned by retrieval (both the chunk
        # search and the relational graph synthesis count).
        valid_sources = _extract_cited_sources(factual_chunks) | _extract_cited_sources(relational_ctx)
        kept, dropped = verify_findings(audit_data.get("findings"), valid_sources)
        _vlog("VALID SOURCES (retrieved documents findings may cite)", "\n".join(sorted(valid_sources)) or "(none)")
        _vlog(f"FINDINGS VERIFIED: kept={len(kept)} dropped={len(dropped)}")
        if dropped:
            _vlog("DROPPED FINDINGS (schema violation / duplicate element / bad citation)",
                  "\n".join(json.dumps(f) for f in dropped))

        preserve = [str(p).strip() for p in audit_data.get("preserve", []) if str(p).strip()][:25]

        result = {
            "compliant": len(kept) == 0,
            "findings": kept,
            "preserve": preserve,
            # Legacy flat lists consumed by the existing UI cards.
            "improvements": [f"{f['fix']} [Source: {f['source']}]" for f in kept],
            "rejections": [f"{f['violation']} [Source: {f['source']}]" for f in kept],
            "reviews": None,
        }
        result = replace_mckinley_brand(result, brand_name)
        _vlog("FINAL AUDIT RESULT", json.dumps(result, indent=2))

        # Yield findings immediately — the stakeholder reviews below are presentational
        # and must not delay the actual audit result.
        yield f"data: {json.dumps({'status': 'Complete', 'result': result})}\n\n"

        if not kept:
            return

        yield f"data: {json.dumps({'status': 'Gathering stakeholder feedback...'})}\n\n"

        feedback_prompt = f"""<role_definition>
You are simulating the reactions of three key stakeholders to a drafted brand image that has just been audited.
</role_definition>

<context>
Original Description: {description}
Violations found: {result['rejections']}
Suggested Improvements: {result['improvements']}
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
        try:
            feedback_response = await self.openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": feedback_prompt}]
            )
            feedback_data = json.loads(feedback_response.choices[0].message.content)
            result["reviews"] = replace_mckinley_brand({
                "Founder": feedback_data.get("founder", "This draft misses our core ethos. Let's apply these fixes."),
                "CBO": feedback_data.get("cbo", "The guidelines are clear. We need to correct these structural issues immediately."),
                "Brand Critic": feedback_data.get("brand_critic", "Too generic. Hopefully the improvements will give it some actual character.")
            }, brand_name)
            # Re-yield the full result with reviews merged; the client's last-write wins.
            yield f"data: {json.dumps({'status': 'Complete', 'result': result})}\n\n"
        except Exception:
            # Reviews are theater; their failure must never sink a finished audit.
            pass

    async def apply_image_improvements(self, image_base64: str, description: str, improvements: list, rejections: list, brand_name: str = "McKINLEY", previous_response_id: str = None, findings: list | None = None, preserve: list | None = None):
        mime, image_payload = _split_data_url(image_base64)

        yield f"data: {json.dumps({'status': 'Assembling edit instructions...'})}\n\n"

        # The edit prompt is assembled deterministically in code from the audit's
        # verified findings — no synthesis LLM between the audit and the editor.
        # Removals first: deletions are the edits the image model drops most readily.
        if findings:
            order = {"remove": 0, "restyle": 1}
            ordered = sorted(
                (f for f in findings if isinstance(f, dict) and str(f.get("fix", "")).strip()),
                key=lambda f: order.get(str(f.get("verdict", "restyle")).lower(), 1),
            )
            fixes = [_strip_citation(str(f["fix"])) for f in ordered]
        else:
            # Legacy fallback: flat improvement strings from an older client.
            fixes = [_strip_citation(str(i)) for i in (improvements or []) if _strip_citation(str(i))]

        if not fixes:
            yield f"data: {json.dumps({'status': 'Error', 'error': 'No edits to apply — the audit found nothing actionable.'})}\n\n"
            return

        fixes = replace_mckinley_brand(fixes, brand_name)
        preserve_list = replace_mckinley_brand([str(p).strip() for p in (preserve or []) if str(p).strip()], brand_name)

        final_edit_prompt = build_edit_prompt(fixes, preserve_list)

        _vlog("APPLY START",
              f"structured_findings={bool(findings)} fixes={len(fixes)} preserve={len(preserve_list)} "
              f"previous_response_id={previous_response_id} quality={settings.openai_image_quality}")
        _vlog("DETERMINISTIC IMAGE EDIT PROMPT (sent to image editor)", final_edit_prompt)

        yield f"data: {json.dumps({'status': 'Generating improved draft...'})}\n\n"

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
                {"type": "input_image", "image_url": f"data:{mime};base64,{image_payload}"},
            ]

        response = await self.openai_client.responses.create(
            model=settings.openai_responses_model,
            previous_response_id=previous_response_id,
            input=[{"role": "user", "content": user_content}],
            tools=[{"type": "image_generation", "quality": settings.openai_image_quality}],
        )

        _vlog(f"IMAGE EDIT CALL took {time.time() - start_time:.2f}s",
              f"response_id={response.id} output_types={[getattr(o, 'type', None) for o in response.output]}")

        image_calls = [o for o in response.output if getattr(o, "type", None) == "image_generation_call"]
        new_image_base64 = image_calls[0].result if image_calls else None

        if not new_image_base64:
            _vlog("IMAGE EDIT RETURNED NO IMAGE")
            yield f"data: {json.dumps({'status': 'Error', 'error': 'The image tool did not return an edited image.'})}\n\n"
            return

        yield f"data: {json.dumps({'status': 'Complete', 'result': {'image_base64': f'data:image/png;base64,{new_image_base64}', 'response_id': response.id}})}\n\n"

_validator_service_instance = None

def get_validator_service() -> ValidatorService:
    global _validator_service_instance
    if _validator_service_instance is None:
        _validator_service_instance = ValidatorService()
    return _validator_service_instance
