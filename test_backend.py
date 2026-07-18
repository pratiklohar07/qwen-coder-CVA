# test_backend.py
# Run this FIRST: python test_backend.py

from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import os
import json
from datetime import datetime

app = FastAPI(title="Event Engine Backend")

# Where the backend stores received events
RECEIVED_DIR = "received_events"
os.makedirs(RECEIVED_DIR, exist_ok=True)

@app.post("/api/v1/events/upload")
async def upload_event(
    video: UploadFile = File(...),
    snapshot: UploadFile = File(...),
    metadata: str = Form(...)  # JSON string sent as a form field
):
    """
    Receives the event video, snapshot, and metadata from the Event Engine.
    """
    try:
        # 1. Parse the metadata JSON string
        meta_dict = json.loads(metadata)
        event_id = meta_dict.get("event_id", "unknown")
        
        # Create a subfolder for this specific event
        event_folder = os.path.join(RECEIVED_DIR, f"event_{event_id}")
        os.makedirs(event_folder, exist_ok=True)

        # 2. Save the Video
        video_path = os.path.join(event_folder, video.filename)
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        # 3. Save the Snapshot
        snapshot_path = os.path.join(event_folder, snapshot.filename)
        with open(snapshot_path, "wb") as f:
            content = await snapshot.read()
            f.write(content)

        # 4. Save the Metadata
        meta_path = os.path.join(event_folder, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=4)

        print(f"\n📥 BACKEND RECEIVED EVENT {event_id}!")
        print(f"   Object: {meta_dict.get('object_detected')}")
        print(f"   Time:   {meta_dict.get('timestamp_human')}")
        print(f"   Files saved to: {event_folder}/")

        return {
            "status": "success",
            "message": f"Event {event_id} received and stored.",
            "event_id": event_id
        }

    except Exception as e:
        print(f"❌ Error processing upload: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/v1/health")
async def health_check():
    return {"status": "online", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    print("🚀 Starting Backend Server on http://localhost:8000")
    print("   API Docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
