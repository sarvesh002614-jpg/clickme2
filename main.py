from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import shutil
import os
import chromadb
from face_utils import get_face_embeddings

app = FastAPI(title="ClickMe API")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Frontend se calls allow karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = chromadb.PersistentClient(path="./clickme_db")
collection = client.get_or_create_collection(name="event_faces")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-event-photos")
async def upload_event_photos(event_id: str = Form(...), files: list[UploadFile] = File(...)):
    """Photographer event ki saari photos yahan upload karega"""
    event_id = event_id.strip()
    processed = 0
    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        faces = get_face_embeddings(save_path)
        for i, face in enumerate(faces):
            doc_id = f"{event_id}_{file.filename}_{i}"
            collection.add(
                ids=[doc_id],
                embeddings=[face["embedding"].tolist()],
                metadatas=[{"event_id": event_id, "filename": file.filename}]
            )
        processed += len(faces)

    return {"status": "success", "faces_processed": processed, "files_count": len(files)}


@app.post("/find-my-photos")
async def find_my_photos(event_id: str = Form(...), selfie: UploadFile = File(...), threshold: float = Form(0.85)):
    """Guest selfie upload karega aur matching photos milengi"""
    event_id = event_id.strip()
    selfie_path = os.path.join(UPLOAD_DIR, f"selfie_{selfie.filename}")
    with open(selfie_path, "wb") as f:
        shutil.copyfileobj(selfie.file, f)

    guest_faces = get_face_embeddings(selfie_path)
    if not guest_faces:
        return {"status": "error", "message": "Selfie mein face detect nahi hua"}

    guest_embedding = guest_faces[0]["embedding"].tolist()
    effective_threshold = 0.85 if threshold > 10.0 else threshold

    results = collection.query(
        query_embeddings=[guest_embedding],
        n_results=50,
        where={"event_id": event_id}
    )

    matched = set()
    if results and "metadatas" in results and results["metadatas"]:
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            if distance < effective_threshold:
                matched.add(metadata["filename"])

    return {"status": "success", "matched_photos": list(matched)}


@app.get("/", response_class=HTMLResponse)
def root():
    frontend_path = os.path.join(os.path.dirname(__file__), "clickme_frontend.html")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())