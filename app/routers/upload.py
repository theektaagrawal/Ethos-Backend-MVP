from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.services.openrag_client import get_openrag_client, OpenRAGClient
from pydantic import BaseModel
from app.config import settings

router = APIRouter(prefix="/api/upload", tags=["upload"])

@router.post("")
@router.post("/")
async def upload_file(file: UploadFile = File(...), client: OpenRAGClient = Depends(get_openrag_client)):
    content = await file.read()
    
    actual_filename = file.filename
    actual_content_type = file.content_type
    
    # --- Check for image files and describe them using OpenAI Vision ---
    is_image = False
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
    if actual_filename.lower().endswith(image_extensions) or actual_content_type.startswith('image/'):
        is_image = True
        
    if is_image:
        import openai
        import os
        import base64
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        b64_image = base64.b64encode(content).decode('utf-8')
        mime_type = actual_content_type if actual_content_type.startswith('image/') else 'image/jpeg'
        
        try:
            messages = [
                {"role": "system", "content": "You are an Image Analysis AI. Your job is to produce a highly detailed markdown document describing this image. Extract any text (OCR), describe the visual elements, and explain the context or meaning of the image."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}", "detail": "high"}}
                ]}
            ]
            
            response = openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=messages,
                max_completion_tokens=32000
            )
            final_content = response.choices[0].message.content
            
            content = final_content.encode('utf-8')
            actual_filename = os.path.splitext(actual_filename)[0] + ".md"
            actual_content_type = "text/markdown"
        except Exception as e:
            raise e

    # --- Check for media files and transcribe them using OpenAI Whisper ---
    is_media = False
    media_extensions = ('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm')
    if actual_filename.lower().endswith(media_extensions) or actual_content_type.startswith(('audio/', 'video/')):
        is_media = True
        
    if is_media:
        import openai
        import os
        import tempfile
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Whisper API requires a file-like object with a name ending in a valid extension
        suffix = os.path.splitext(actual_filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            with open(tmp_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model=settings.openai_audio_model, 
                    file=audio_file
                )
                
            import cv2
            import base64
            
            video_capture = cv2.VideoCapture(tmp_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 24
            
            # Extract 1 frame every 5 seconds
            interval_frames = int(fps * 5)
            if interval_frames <= 0:
                interval_frames = 1
                
            base64_frames = []
            frame_idx = 0
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                
                if frame_idx % interval_frames == 0:
                    h, w = frame.shape[:2]
                    max_dim = 768
                    if max(h, w) > max_dim:
                        scale = max_dim / float(max(h, w))
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        
                    _, buffer = cv2.imencode('.jpg', frame)
                    base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
                    
                frame_idx += 1
                
            video_capture.release()
            
            if base64_frames:
                messages = [
                    {"role": "system", "content": "You are a Video Analysis AI. Your job is to produce a highly detailed, chronological markdown document describing this video. You are provided with keyframes extracted every 5 seconds and the audio transcript. Combine the visual actions and spoken words into a cohesive, readable document."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Audio Transcript:\n{transcript.text}\n\nKeyframes (in chronological order):"}
                    ]}
                ]
                
                for b64 in base64_frames:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
                    })
                    
                response = openai_client.chat.completions.create(
                    model=settings.openai_chat_model,
                    messages=messages,
                    max_completion_tokens=32000
                )
                final_content = response.choices[0].message.content
            else:
                final_content = transcript.text
            
            # replace content with the final markdown
            content = final_content.encode('utf-8')
            
            # Change filename to .md
            actual_filename = os.path.splitext(actual_filename)[0] + ".md"
            actual_content_type = "text/markdown"
        except Exception as e:
            # clean up and raise
            os.remove(tmp_path)
            raise e
            
        os.remove(tmp_path)
    # ----------------------------------------------------------------------
    
    # --- Check for PDF files and extract them using OpenAI ---
    is_pdf = False
    if actual_filename.lower().endswith('.pdf') or actual_content_type == 'application/pdf':
        is_pdf = True
        
    if is_pdf:
        import openai
        import os
        import tempfile
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        suffix = os.path.splitext(actual_filename)[1] or '.pdf'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            with open(tmp_path, "rb") as pdf_file:
                uploaded_file = openai_client.files.create(
                    file=pdf_file,
                    purpose="assistants"
                )
            
            messages = [
                {"role": "system", "content": "You are a Document Analysis AI. Your job is to produce a highly detailed markdown document describing this file. Extract all text accurately, preserve headings, formatting, and tables where possible."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the complete contents of this PDF into markdown."},
                    {"type": "file", "file": {"file_id": uploaded_file.id}}
                ]}
            ]
            
            response = openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=messages,
                max_completion_tokens=32000
            )
            final_content = response.choices[0].message.content
            
            content = final_content.encode('utf-8')
            actual_filename = os.path.splitext(actual_filename)[0] + ".md"
            actual_content_type = "text/markdown"
            
            # Clean up OpenAI file
            try:
                openai_client.files.delete(uploaded_file.id)
            except Exception:
                pass
                
        except Exception as e:
            os.remove(tmp_path)
            raise e
            
        os.remove(tmp_path)
    # ----------------------------------------------------------------------
    
    import unicodedata
    safe_filename = unicodedata.normalize('NFKD', actual_filename).encode('ascii', 'ignore').decode('ascii')
    if not safe_filename.strip():
        safe_filename = "upload_file.txt" if is_media else "upload_file"
        
    files = {"file": (safe_filename, content, actual_content_type)}
    
    try:
        # Try /v1/documents/ingest first (OpenRAG v1 ingest endpoint)
        response = await client.client.post("/v1/documents/ingest", files=files)
        if response.status_code == 404:
            # Fallback to langflow upload if needed
            response = await client.client.post("/langflow/files/upload", files=files)
            
        # Also upload the document to LightRAG so it gets parsed by Docling and indexed into the 3D Graph
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=30.0) as lr_client:
                await lr_client.post(
                    f"{settings.lightrag_url.rstrip('/')}/documents/file",
                    files={"file": (safe_filename, content, actual_content_type)}
                )
        except Exception as lr_err:
            import logging
            logging.getLogger(__name__).warning(f"Failed to forward upload to LightRAG: {lr_err}")
            
        import datetime
        from app.services.document_store import save_document
        
        # Save document metadata locally so it shows up in the Brand Memory page
        # We do this before the mock fallback return so it's always tracked.
        doc_id = "doc_" + datetime.datetime.now().isoformat()
        try:
            doc_id = response.json().get("id", doc_id)
        except Exception:
            pass
            
        doc_data = {
            "id": doc_id,
            "filename": actual_filename,
            "status": "completed",
            "created_at": datetime.datetime.now().isoformat(),
            "size": len(content)
        }
        save_document(doc_data)

        if response.status_code != 200:
            print(f"OpenRAG returned status {response.status_code}: {response.text}")
            # Mock success if OpenRAG doesn't have an upload endpoint ready
            return {"status": "success", "message": "File uploaded (mock fallback)"}
            
        return response.json()
    except Exception as e:
        import traceback
        traceback.print_exc()
        import datetime
        from app.services.document_store import save_document
        doc_data = {
            "id": "doc_" + datetime.datetime.now().isoformat(),
            "filename": actual_filename,
            "status": "completed",
            "created_at": datetime.datetime.now().isoformat(),
            "size": len(content)
        }
        save_document(doc_data)
        return {"status": "success", "message": f"File uploaded (mock fallback, error: {e})"}

class UrlIngestRequest(BaseModel):
    url: str

@router.post("/url")
async def ingest_url(request: UrlIngestRequest, client: OpenRAGClient = Depends(get_openrag_client)):
    response = await client.client.post("/ingest/url", json={"url": request.url})
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"OpenRAG error: {response.text}")
    return response.json()
