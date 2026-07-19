# backend.py
import os
import json
import shutil
import bcrypt
from datetime import datetime, timedelta
from typing import List
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session 
from pydantic import BaseModel
from jose import JWTError, jwt
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import httpx
from fastapi.responses import StreamingResponse
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage



# --- EMAIL CONFIGURATION ---
# Use a Gmail account. You MUST generate an "App Password" in your Google Account Security settings.
EMAIL_SENDER = "iampratiknil@gmail.com"
EMAIL_PASSWORD = "mrysnhhsybfezgxn" 
EMAIL_RECEIVER = "pratiklohar37178@gmail.com"

# ==========================================
# CONFIGURATION & DATABASE SETUP
# ==========================================
DATABASE_URL = "sqlite:///./surveillance.db"
SECRET_KEY = "your-super-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440 # 24 hours
EDGE_API_KEY = "super_secret_edge_key_123"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# DATABASE MODELS
# ==========================================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, unique=True, index=True)
    object_detected = Column(String)
    timestamp_human = Column(String)
    timestamp_unix = Column(Float)
    video_path = Column(String)
    snapshot_path = Column(String)
    metadata_json = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    # --- NEW FOR PAGER APK ---
    is_acknowledged = Column(Boolean, default=False) 

Base.metadata.create_all(bind=engine)

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class EventResponse(BaseModel):
    id: int
    event_id: int
    object_detected: str
    timestamp_human: str
    snapshot_url: str
    video_url: str

# ==========================================
# SECURITY (Passwords & JWT) - FIXED FOR SWAGGER UI
# ==========================================
oauth2_scheme = HTTPBearer() # <--- This creates the simple "Paste Token" box in Swagger

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# <--- FIXED: Extracts token from the HTTPBearer credentials
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
):
    token = credentials.credentials # <--- Grabs the raw JWT string
    credentials_exception = HTTPException(status_code=401, detail="Invalid authentication credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.username == username).first()
    if user is None: raise credentials_exception
    return user

def send_email_alert(snapshot_path, metadata):
    """Sends an HTML email with the snapshot attached."""
    try:
        msg = MIMEMultipart()
        msg['Subject'] = f"🚨 URGENT: {metadata['object_detected'].upper()} Detected (Event #{metadata['event_id']})"
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color:red;">🚨 Security Alert: {metadata['object_detected'].upper()}</h2>
                <p><strong>Event ID:</strong> {metadata['event_id']}</p>
                <p><strong>Time:</strong> {metadata['timestamp_human']}</p>
                <hr>
                <p>Please open the <b>AI Surveillance Pager APK</b> or log in to the Command Center Dashboard to review the incident and acknowledge the alarm.</p>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        with open(snapshot_path, 'rb') as img:
            mime_img = MIMEImage(img.read(), name="snapshot.jpg")
            msg.attach(mime_img)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("[Alert] 📧 Email Alert Sent!")
    except Exception as e:
        print(f"[Alert] ❌ Email Failed: {e}")

def trigger_background_alerts(snapshot_path, metadata):
    """Runs in a thread so it doesn't block the Edge Engine upload."""
    send_email_alert(snapshot_path, metadata)



# ==========================================
# FASTAPI APP INITIALIZATION
# ==========================================
app = FastAPI(title="AI Surveillance Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/api/v1/auth/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/api/v1/auth/login", response_model=Token)
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/v1/events/upload")
async def upload_event(
    video: UploadFile = File(...),
    snapshot: UploadFile = File(...),
    metadata: str = Form(...),
    x_api_key: str = Header(None),
    db: Session = Depends(get_db)
):
    if x_api_key != EDGE_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid Edge API Key")

    meta_dict = json.loads(metadata)
    event_id = meta_dict.get("event_id")
    
    video_path = os.path.join(STORAGE_DIR, f"event_{event_id}_video.mp4")
    snapshot_path = os.path.join(STORAGE_DIR, f"event_{event_id}_snapshot.jpg")
    
    with open(video_path, "wb") as buffer: shutil.copyfileobj(video.file, buffer)
    with open(snapshot_path, "wb") as buffer: shutil.copyfileobj(snapshot.file, buffer)
    
    db_event = Event(
        event_id=event_id,
        object_detected=meta_dict.get("object_detected"),
        timestamp_human=meta_dict.get("timestamp_human"),
        timestamp_unix=meta_dict.get("timestamp_unix"),
        video_path=video_path,
        snapshot_path=snapshot_path,
        metadata_json=metadata
    )
    db.add(db_event)
    db.commit()
    
    threading.Thread(target=trigger_background_alerts, args=(snapshot_path, meta_dict), daemon=True).start()

    
    return {"status": "success", "event_id": event_id}

@app.get("/api/v1/events", response_model=List[EventResponse])
def get_events(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    events = db.query(Event).order_by(Event.event_id.desc()).limit(100).all()
    
    response = []
    for e in events:
        response.append({
            "id": e.id,
            "event_id": e.event_id,
            "object_detected": e.object_detected,
            "timestamp_human": e.timestamp_human,
            "snapshot_url": f"/api/v1/files/{os.path.basename(e.snapshot_path)}",
            "video_url": f"/api/v1/files/{os.path.basename(e.video_path)}"
        })
    return response

@app.get("/api/v1/files/{file_name}")
async def get_file(file_name: str):
    file_path = os.path.join(STORAGE_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# ==========================================
# SERVE DASHBOARD & START SERVER
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    try:
        # This serves your HTML file directly from the backend!
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>dashboard.html not found. Place it in the same folder as backend.py.</h1>", status_code=404)
    

# Add this endpoint to backend.py
@app.get("/api/v1/live_feed")
async def proxy_live_feed():
    """
    Proxies the live MJPEG stream from the Edge Engine (Port 8080) 
    through the Backend (Port 8000) to bypass Firewall and CORS issues.
    """
    async def stream_generator():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", "http://localhost:8080/video_feed", timeout=None) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except Exception:
            yield b"" # Stream offline

    return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# ==========================================
# PAGER APK ENDPOINTS
# ==========================================

@app.get("/api/v1/pager/latest")
def get_latest_pager_alert(db: Session = Depends(get_db)):
    """The APK polls this every 2 seconds."""
    # Get the absolute latest event
    latest_event = db.query(Event).order_by(Event.id.desc()).first()
    
    if not latest_event:
        return {"alert_active": False}
        
    return {
        "alert_active": True,
        "event_id": latest_event.event_id,
        "object_detected": latest_event.object_detected,
        "timestamp_human": latest_event.timestamp_human,
        "is_acknowledged": latest_event.is_acknowledged,
        "snapshot_url": f"/api/v1/files/{os.path.basename(latest_event.snapshot_path)}"
    }

@app.post("/api/v1/pager/acknowledge/{event_id}")
def acknowledge_alert(event_id: int, db: Session = Depends(get_db)):
    """The APK calls this when the user presses the 'Silence' button."""
    db_event = db.query(Event).filter(Event.event_id == event_id).first()
    if db_event:
        db_event.is_acknowledged = True
        db.commit()
        print(f"[Pager] ✅ Event {event_id} acknowledged by APK user.")
        return {"status": "success", "message": "Alert silenced."}
    return {"status": "error", "message": "Event not found."}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Secure Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

