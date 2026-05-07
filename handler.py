import runpod
import base64
import io
import pickle
import numpy as np
import insightface
from PIL import Image

print("[RunPod] Loading ArcFace model...")

model = insightface.model_zoo.get_model(
    "/app/models/w600k_r50.onnx",
    providers=["CPUExecutionProvider"]
)
model.prepare(ctx_id=-1)

with open("/app/models/arcface_db.pkl", "rb") as f:
    db_data = pickle.load(f)

db_embeddings = np.array(db_data["embeddings"]).astype("float32").reshape(-1, 512)
db_labels     = np.array(db_data["labels"])

print(f"[RunPod] Ready. {len(db_labels)} faces loaded.")

def handler(job):
    inp        = job["input"]
    faces_list = inp.get("faces", [])
    cam_id     = inp.get("cam_id", "Unknown")
    device_id  = inp.get("device_id", "Unknown")

    print(f"")
    print(f"{'='*55}")
    print(f"[JOB START] job_id={job.get('id', 'N/A')}")
    print(f"  cam_id    : {cam_id}")
    print(f"  device_id : {device_id}")
    print(f"  faces     : {len(faces_list)}")
    print(f"{'='*55}")

    batch_results = []

    for idx, item in enumerate(faces_list, start=1):
        face_b64   = item["face_b64"]
        process_id = item["pid"]
        time_str   = item.get("time", "N/A")

        print(f"  [{idx}/{len(faces_list)}] Processing ...")
        print(f"    PID       : {process_id}")
        print(f"    cam_id    : {cam_id}")
        print(f"    device_id : {device_id}")
        print(f"    time      : {time_str}")

        # 1. Decode & Pre-process
        img_bytes = base64.b64decode(face_b64)
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        face_bgr  = np.array(img_pil.resize((112, 112), Image.BILINEAR))[:, :, ::-1]

        # 2. Extract Embedding
        embedding  = model.get_feat(face_bgr)
        final_name = "Unknown"
        top_score  = 0.0

        if embedding is not None:
            embedding = embedding.flatten()

            norm_db  = np.linalg.norm(db_embeddings, axis=1)
            norm_emb = np.linalg.norm(embedding)
            sims     = np.dot(db_embeddings, embedding) / (norm_db * norm_emb + 1e-9)

            best_idx  = np.argmax(sims)
            top_score = float(sims[best_idx])
            top_label = str(db_labels[best_idx])

            if top_score > 0.50:
                final_name = top_label
                print(f"    RESULT    : ✅ RECOGNIZED | {final_name} (score={top_score:.4f})")
            else:
                print(f"    RESULT    : ❌ UNKNOWN    (score={top_score:.4f})")
        else:
            print(f"    RESULT    : ⚠️  embedding=None (bad face crop?)")

        batch_results.append({
            "name":      final_name,
            "score":     f"{top_score:.2f}",
            "pid":       process_id,
            "cam_id":    cam_id,
            "device_id": device_id,
            "time":      time_str,
        })

        print(f"    {'─'*45}")

    # Job summary
    recognized    = [r for r in batch_results if r["name"] != "Unknown"]
    unknown       = [r for r in batch_results if r["name"] == "Unknown"]
    names_scores  = [f"{r['name']} ({r['score']})" for r in recognized]

    print(f"")
    print(f"[JOB DONE] job_id={job.get('id', 'N/A')}")
    print(f"  Total     : {len(batch_results)}")
    print(f"  Recognized: {len(recognized)} → {names_scores}")
    print(f"  Unknown   : {len(unknown)}")
    print(f"{'='*55}")
    print(f"")

    return {"batch_results": batch_results}

runpod.serverless.start({"handler": handler})