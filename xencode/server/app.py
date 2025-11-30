from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict
import uuid
import logging

from .database import get_db, init_db, Session as SessionModel, User
from .socket_manager import manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Xencode Collaboration Server")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Xencode Collaboration Server"}

@app.post("/sessions/create")
def create_session(username: str, db: Session = Depends(get_db)):
    """Create a new collaboration session"""
    # Create host user if not exists
    host = db.query(User).filter(User.username == username).first()
    if not host:
        host = User(username=username)
        db.add(host)
        db.commit()
        db.refresh(host)
    
    # Generate invite code
    invite_code = str(uuid.uuid4())[:8]
    
    # Create session
    session = SessionModel(host_id=host.id, invite_code=invite_code)
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return {"session_id": session.id, "invite_code": invite_code}

@app.get("/sessions/{invite_code}")
def get_session(invite_code: str, db: Session = Depends(get_db)):
    """Get session details"""
    session = db.query(SessionModel).filter(SessionModel.invite_code == invite_code).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session.id, "host": session.host.username, "is_active": session.is_active}

@app.websocket("/ws/{session_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, username: str):
    await manager.connect(websocket, session_id)
    try:
        # Broadcast join message
        await manager.broadcast(
            {"type": "system", "content": f"{username} joined the session"},
            session_id
        )
        
        while True:
            data = await websocket.receive_json()
            # Add sender info
            data["sender"] = username
            
            # Broadcast to others
            await manager.broadcast(data, session_id, exclude=websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        await manager.broadcast(
            {"type": "system", "content": f"{username} left the session"},
            session_id
        )
