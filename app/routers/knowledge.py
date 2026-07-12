from collections import defaultdict
from datetime import datetime
import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from app.services.openrag_client import get_openrag_client, OpenRAGClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

BRAND_PRIMITIVES = [
    {
        "id": "heritage",
        "label": "Heritage",
        "query": "brand heritage origin history founding legacy archival references",
    },
    {
        "id": "refusals",
        "label": "Refusals",
        "query": "brand refusals never do constraints prohibitions no-go rules boundaries",
    },
    {
        "id": "voice",
        "label": "Voice",
        "query": "brand voice tone language copywriting phrasing syntax declarative style",
    },
    {
        "id": "taste",
        "label": "Taste",
        "query": "brand taste aesthetics design product sensibility visual style material preferences",
    },
    {
        "id": "lived_position",
        "label": "Lived Position",
        "query": "brand position worldview stance market posture customer promise operating belief",
    },
    {
        "id": "cultural",
        "label": "Cultural",
        "query": "brand culture cultural context rituals community symbols references audience",
    },
    {
        "id": "formative",
        "label": "Formative",
        "query": "formative brand moments origin decisions milestones defining events",
    },
    {
        "id": "contradictions",
        "label": "Contradictions",
        "query": "brand contradictions tensions tradeoffs paradoxes conflicts exceptions",
    },
]

BROAD_DOCUMENT_DISCOVERY_QUERIES = [
    "brand guidelines strategy voice refusals heritage taste",
    "document memo manifesto playbook cultural position product",
]


async def discover_openrag_documents(client: OpenRAGClient):
    discovered = {}

    async def search(query: str):
      try:
          response = await client.client.post(
              "/v1/search",
              json={"query": query, "limit": 50, "score_threshold": 0},
          )
          if response.status_code != 200:
              return []
          return response.json().get("results", [])
      except Exception:
          return []

    results = await asyncio.gather(
        *(search(query) for query in BROAD_DOCUMENT_DISCOVERY_QUERIES)
    )

    for result_set in results:
        for item in result_set:
            filename = item.get("filename")
            if not filename or filename in discovered:
                continue
            discovered[filename] = {
                "id": f"openrag_{filename}",
                "filename": filename,
                "status": "indexed",
                "source": "openrag",
                "score": item.get("score"),
                "size": len(item.get("text") or ""),
            }

    return list(discovered.values())

@router.get("")
@router.get("/")
async def list_documents(client: OpenRAGClient = Depends(get_openrag_client)):
    try:
        from app.services.document_store import load_documents
        local_docs = load_documents()
        openrag_docs = await discover_openrag_documents(client)
        
        doc_map = {d.get("filename"): d for d in local_docs if d.get("filename")}
        for od in openrag_docs:
            fn = od.get("filename")
            if fn:
                if fn in doc_map:
                    doc_map[fn].update(od)
                else:
                    doc_map[fn] = od
        return {"documents": list(doc_map.values())}
    except Exception:
        return {"documents": []}

@router.get("/stats")
async def get_stats(client: OpenRAGClient = Depends(get_openrag_client)):
    try:
        from app.services.document_store import load_documents
        local_docs = load_documents()
        
        openrag_docs = await discover_openrag_documents(client)
        total_docs = len(openrag_docs)
        
        # Vectors: Deterministic calculation based on local docs size
        vectors = sum([max(1, d.get("size", 1000) // 1000) for d in local_docs])
        if vectors == 0 and total_docs > 0:
             vectors = total_docs * 10
        
        # Last Sync: Max created_at from local docs
        last_sync = "Unknown"
        valid_dates = []
        for d in local_docs:
            created_at = d.get("created_at")
            if created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at)
                    valid_dates.append(dt)
                except Exception:
                    pass
        if valid_dates:
            last_sync = max(valid_dates).strftime("%Y-%m-%d %H:%M")

        # Integrity
        openrag_count = len(openrag_docs)
        local_count = len(local_docs)
        if local_count == 0:
            integrity = 100 if openrag_count == 0 else 0
        else:
            integrity = min(100, int((openrag_count / local_count) * 100))

        return {
            "total_documents": total_docs,
            "vectors": vectors,
            "last_sync": last_sync,
            "integrity": integrity
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"total_documents": 0, "vectors": 0, "last_sync": "Unknown", "integrity": 100}

@router.get("/brand-graph")
async def get_brand_graph(client: OpenRAGClient = Depends(get_openrag_client)):
    from app.services.document_store import load_documents

    local_docs = load_documents()
    local_doc_map = {
        (doc.get("filename") or doc.get("name") or ""): doc
        for doc in local_docs
        if doc.get("filename") or doc.get("name")
    }
    primitive_docs = defaultdict(dict)

    async def search_primitive(primitive):
        try:
            response = await client.client.post(
                "/v1/search",
                json={
                    "query": primitive["query"],
                    "limit": 8,
                    "score_threshold": 0,
                },
            )
            if response.status_code != 200:
                return primitive["id"], []
            return primitive["id"], response.json().get("results", [])
        except Exception:
            return primitive["id"], []

    primitive_results = await asyncio.gather(
        *(search_primitive(primitive) for primitive in BRAND_PRIMITIVES)
    )

    for primitive_id, results in primitive_results:
        for item in results:
            filename = item.get("filename") or "Unknown source"
            current = primitive_docs[primitive_id].setdefault(
                filename,
                {
                    "id": local_doc_map.get(filename, {}).get("id", filename),
                    "filename": filename,
                    "tags": set(),
                    "snippets": [],
                    "score": 0,
                },
            )
            current["tags"].add(primitive_id)
            current["score"] = max(current["score"], item.get("score") or 0)
            text = (item.get("text") or "").strip()
            if text and len(current["snippets"]) < 3:
                current["snippets"].append(
                    {
                        "text": text[:420],
                        "page": item.get("page"),
                        "score": item.get("score"),
                    }
                )

    nodes = []
    for primitive in BRAND_PRIMITIVES:
        docs = []
        for doc in primitive_docs[primitive["id"]].values():
            docs.append(
                {
                    **doc,
                    "tags": sorted(doc["tags"]),
                }
            )
        docs.sort(key=lambda doc: doc.get("score") or 0, reverse=True)
        nodes.append(
            {
                "id": primitive["id"],
                "label": primitive["label"],
                "count": len(docs),
                "documents": docs,
            }
        )

    edge_weights = defaultdict(lambda: {"weight": 0, "documents": set()})
    filename_to_primitives = defaultdict(set)
    for primitive_id, docs in primitive_docs.items():
        for filename in docs.keys():
            filename_to_primitives[filename].add(primitive_id)

    for filename, primitive_ids in filename_to_primitives.items():
        ordered_ids = sorted(primitive_ids)
        for index, source in enumerate(ordered_ids):
            for target in ordered_ids[index + 1:]:
                key = (source, target)
                edge_weights[key]["weight"] += 1
                edge_weights[key]["documents"].add(filename)

    edges = [
        {
            "source": source,
            "target": target,
            "weight": payload["weight"],
            "documents": sorted(payload["documents"]),
        }
        for (source, target), payload in edge_weights.items()
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "generated_at": datetime.utcnow().isoformat(),
    }

def _as_string_list(value):
    """Normalise LightRAG source metadata without losing the original payload."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


@router.get("/3d-graph")
async def get_3d_graph(
    label: str = Query("*", min_length=1),
    max_depth: int = Query(3, ge=1, le=8),
    max_nodes: int = Query(300, ge=1, le=1000),
):
    """Return a faithful, display-ready LightRAG subgraph.

    The previous adapter replaced graph properties and invented hub edges when
    LightRAG had no relationships.  This endpoint deliberately preserves the
    source payload so the UI can distinguish extracted evidence from an empty
    or truncated graph.
    """
    import httpx
    from app.config import settings

    nodes = []
    edges = []
    is_truncated = False

    # 1. Try fetching real graph structure from LightRAG /graphs endpoint (same as WebUI)
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            res = await http.get(
                f"{settings.lightrag_url.rstrip('/')}/graphs",
                params={"label": label, "max_depth": max_depth, "max_nodes": max_nodes}
            )
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict):
                    raw_nodes = data.get("nodes", [])
                    raw_edges = data.get("edges", [])

                    for n in raw_nodes:
                        if isinstance(n, dict):
                            props = n.get("properties", {})
                            node_id = str(n.get("id", props.get("entity_id", "")))
                            if node_id:
                                labels = n.get("labels", [node_id])
                                label_str = str(labels[0] if isinstance(labels, list) and labels else node_id)
                                nodes.append({
                                    "id": node_id,
                                    "label": label_str,
                                    "type": str(props.get("entity_type", "concept")).lower(),
                                    "sub": str(props.get("description", "Extracted via LightRAG"))[:500],
                                    "weight": float(props.get("weight", 1) or 1),
                                    "labels": labels if isinstance(labels, list) else [label_str],
                                    "source_ids": _as_string_list(props.get("source_ids", props.get("source_id"))),
                                    "file_path": props.get("file_path"),
                                    "properties": props,
                                })

                    for e in raw_edges:
                        if isinstance(e, dict):
                            src = str(e.get("source", ""))
                            tgt = str(e.get("target", ""))
                            props = e.get("properties", {})
                            if src and tgt and src != tgt:
                                kw = str(props.get("keywords", props.get("description", "relates to")))
                                edges.append({
                                    "source": src,
                                    "target": tgt,
                                    "label": kw if kw else "relates to",
                                    "strength": float(props.get("weight", 1) or 1),
                                    "source_ids": _as_string_list(props.get("source_ids", props.get("source_id"))),
                                    "properties": props,
                                })
                    is_truncated = bool(data.get("is_truncated", False))
                else:
                    is_truncated = False
            else:
                is_truncated = False
    except Exception as e:
        logger.warning(f"LightRAG fetch failed or not ready yet: {e}")

    return {
        "meta": {"query_label": label, "max_depth": max_depth, "max_nodes": max_nodes},
        "nodes": nodes,
        "edges": edges,
        "is_truncated": is_truncated,
        "generated_at": datetime.utcnow().isoformat(),
    }


class ScoreDiagnosticRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=20, ge=1, le=50)


@router.post("/search-diagnostics")
async def search_diagnostics(
    request: ScoreDiagnosticRequest,
    client: OpenRAGClient = Depends(get_openrag_client),
):
    """Expose score distribution without pretending raw hybrid scores are similarity."""
    response = await client.client.post(
        "/v1/search",
        json={"query": request.query, "limit": request.limit, "score_threshold": 0},
    )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="OpenRAG search diagnostics request failed")
    results = response.json().get("results", [])
    scores = sorted(float(item["score"]) for item in results if isinstance(item.get("score"), (int, float)))
    def percentile(fraction: float):
        if not scores:
            return None
        return scores[min(len(scores) - 1, round((len(scores) - 1) * fraction))]
    return {
        "query": request.query,
        "score_kind": "OpenSearch raw hybrid _score (not cosine similarity or a percentage)",
        "result_count": len(results),
        "distribution": {
            "min": percentile(0), "median": percentile(0.5), "p90": percentile(0.9), "max": percentile(1),
            "distinct_rounded_2dp": len({round(score, 2) for score in scores}),
        },
        "results": [
            {"rank": index + 1, "filename": item.get("filename"), "page": item.get("page"), "raw_score": item.get("score"), "embedding_model": item.get("embedding_model")}
            for index, item in enumerate(results)
        ],
    }

@router.delete("/{doc_id}")
async def delete_document_endpoint(
    doc_id: str,
    filename: str | None = Query(default=None),
    client: OpenRAGClient = Depends(get_openrag_client),
):
    try:
        from app.services.document_store import (
            load_documents,
            delete_document,
            delete_document_by_filename,
        )

        docs = load_documents()
        doc = next((d for d in docs if d.get("id") == doc_id), None)
        target_filename = (
            filename
            or (doc or {}).get("filename")
            or (doc or {}).get("name")
            or (doc_id.removeprefix("openrag_") if doc_id.startswith("openrag_") else doc_id)
        )
        if not doc and target_filename:
            doc = next((d for d in docs if d.get("filename") == target_filename or d.get("name") == target_filename), None)

        if not target_filename:
            raise HTTPException(status_code=400, detail="Document filename is required")

        response = await client.client.request(
            "DELETE",
            "/v1/documents",
            json={"filename": target_filename},
        )
        if response.status_code not in (200, 404):
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenRAG error: {response.text}",
            )

        # Brand context is cached for generation; document deletion must take
        # effect immediately rather than waiting for the TTL.
        from app.services.generator_service import invalidate_brand_context_cache
        invalidate_brand_context_cache()

        # Also delete the document from LightRAG index
        try:
            import httpx as _httpx
            from app.config import settings
            async with _httpx.AsyncClient(timeout=10.0) as lr_client:
                # 1. Use stored lightrag_id if available, otherwise query LightRAG POST /documents/paginated
                lr_doc_id = (doc or {}).get("lightrag_id")
                if not lr_doc_id:
                    try:
                        docs_res = await lr_client.post(
                            f"{settings.lightrag_url.rstrip('/')}/documents/paginated",
                            json={"page": 1, "page_size": 100}
                        )
                        if 200 <= docs_res.status_code < 300:
                            docs_data = docs_res.json()
                            items = []
                            if isinstance(docs_data, list):
                                items = docs_data
                            elif isinstance(docs_data, dict):
                                for key in ["statuses", "documents", "data", "results"]:
                                    if key in docs_data and isinstance(docs_data[key], (list, dict)):
                                        val = docs_data[key]
                                        if isinstance(val, list):
                                            items.extend(val)
                                        elif isinstance(val, dict):
                                            for k, v in val.items():
                                                if isinstance(v, dict):
                                                    items.append({"id": k, **v})
                                if not items:
                                    for k, v in docs_data.items():
                                        if isinstance(v, dict):
                                            items.append({"id": k, **v})
                                        elif isinstance(v, str):
                                            items.append({"id": k, "filename": v})

                            target_clean = target_filename.replace("\\", "/").split("/")[-1]
                            for item in items:
                                if not isinstance(item, dict):
                                    continue
                                cid = item.get("id") or item.get("doc_id") or item.get("document_id")
                                if not cid:
                                    continue
                                cfn = str(item.get("file_path") or item.get("filename") or item.get("doc_name") or item.get("name") or item.get("path") or "").replace("\\", "/").split("/")[-1]
                                if cfn == target_clean or (target_clean and target_clean in cfn) or (cfn and cfn in target_clean) or str(cid) == doc_id:
                                    lr_doc_id = str(cid)
                                    break
                    except Exception as get_err:
                        import logging
                        logging.getLogger(__name__).warning(f"Failed to query LightRAG documents list: {get_err}")

                final_lr_id = lr_doc_id or doc_id

                # 2. Call delete_document endpoint passing correct doc_id and delete_file: true
                await lr_client.request(
                    "DELETE",
                    f"{settings.lightrag_url.rstrip('/')}/documents/delete_document",
                    json={"doc_ids": [final_lr_id], "delete_file": True}
                )
        except Exception as lr_err:
            import logging
            logging.getLogger(__name__).warning(f"Failed to forward delete to LightRAG: {lr_err}")

        delete_document(doc_id)
        delete_document_by_filename(target_filename)
        return {"status": "success", "filename": target_filename}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
