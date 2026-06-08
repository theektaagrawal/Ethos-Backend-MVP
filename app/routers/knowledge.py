from collections import defaultdict
from datetime import datetime
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query
from app.services.openrag_client import get_openrag_client, OpenRAGClient

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
        return {"documents": await discover_openrag_documents(client)}
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

        delete_document(doc_id)
        delete_document_by_filename(target_filename)
        return {"status": "success", "filename": target_filename}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
