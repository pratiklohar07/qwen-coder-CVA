# backend.py
import os
import json
import shutil
import bcrypt
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
# <--- FIXED: Importing HTTPBearer instead of OAuth2PasswordBearer
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session 

from pydantic import BaseModel
from jose import JWTError, jwt

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

@app.get("/")
def root():
    return {"message": "AI Surveillance Backend Online"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Secure Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
