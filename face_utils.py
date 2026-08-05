import insightface
import cv2
import numpy as np

# Model pehli baar run hone par download hoga (thoda time lagega)
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 = CPU, GPU hai to 0 karo

def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image load nahi hui: {image_path}")
        return []
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    faces = app.get(img)
    results = []
    for face in faces:
        emb = face.embedding
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        results.append({
            "bbox": face.bbox,
            "embedding": emb
        })
    return results

# Test karne ke liye
if __name__ == "__main__":
    test_image = "test.jpg"   # apni koi photo isi folder mein test.jpg naam se rakho
    faces = get_face_embeddings(test_image)
    print(f"{len(faces)} face(s) detect hue")
    for i, f in enumerate(faces):
        print(f"Face {i+1}: embedding shape = {f['embedding'].shape}")