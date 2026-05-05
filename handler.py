import runpod
import base64
import io
import pickle
import numpy as np
import insightface
from PIL import Image

print("[RunPod] Loading ArcFace model...")

# Initialize InsightFace
model = insightface.model_zoo.get_model(
    "/app/models/w600k_r50.onnx",
    providers=["CPUExecutionProvider"]
)
model.prepare(ctx_id=-1)

# Load your pre-computed biometric database
with open("/app/models/arcface_db.pkl", "rb") as f:
    db_data = pickle.load(f)

db_embeddings = np.array(db_data["embeddings"]).astype("float32").reshape(-1, 512)
db_labels     = np.array(db_data["labels"])

print(f"[RunPod] Ready. {len(db_labels)} faces loaded.")

def handler(job):
    inp = job["input"]
    faces_list = inp.get("faces", [])  # Expecting a list from localserver
    cam_id     = inp.get("cam_id", "Unknown")
    
    batch_results = []

    for item in faces_list:
        image_b64  = item["face_b64"]
        process_id = item["pid"]

        # 1. Decode & Pre-process
        img_bytes = base64.b64decode(image_b64)
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize to ArcFace standard input size
        face_bgr  = np.array(img_pil.resize((112, 112), Image.BILINEAR))[:, :, ::-1]

        # 2. Extract Embedding
        embedding = model.get_feat(face_bgr)
        final_name = "Unknown"
        top_score = 0.0

        if embedding is not None:
            embedding = embedding.flatten()
            # Cosine Similarity Calculation
            norm_db = np.linalg.norm(db_embeddings, axis=1)
            norm_emb = np.linalg.norm(embedding)
            sims = np.dot(db_embeddings, embedding) / (norm_db * norm_emb + 1e-9)
            
            idx = np.argmax(sims)
            top_score = float(sims[idx])
            
            # Recognition Threshold (Adjust as needed)
            if top_score > 0.50:
                final_name = str(db_labels[idx])

        batch_results.append({
            "name": final_name,
            "score": f"{top_score:.2f}",
            "pid": process_id,
            "cam_id": cam_id
        })

    print(f"[RunPod] Processed batch of {len(batch_results)} faces.")
    
    return {"batch_results": batch_results}

runpod.serverless.start({"handler": handler})