# app.py
import os
import uuid
import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from apscheduler.schedulers.background import BackgroundScheduler
from PIL import Image
import io

# Optional heavy libs - if not available, code falls back to stubs
try:
    import face_recognition
except Exception:
    face_recognition = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ---------- Config ----------
DATABASE_URL = "sqlite:///./seniorcare.db"
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- DB setup ----------
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class UserModel(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    preferred_language = Column(String, default="hi")  # e.g., 'hi', 'en', 'bn', 'te' etc.
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    emergency_contacts = Column(Text, default="[]")  # JSON list of contact dicts


class ReminderModel(Base):
    __tablename__ = "reminders"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    title = Column(String)
    details = Column(Text, nullable=True)
    remind_at = Column(DateTime, index=True)
    repeat = Column(String, nullable=True)  # e.g., "daily", "weekly", cron-like
    active = Column(Boolean, default=True)


class FaceProfileModel(Base):
    __tablename__ = "faces"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    name = Column(String)
    image_path = Column(String)
    encoding = Column(Text, nullable=True)  # serialized encoding if used


Base.metadata.create_all(bind=engine)

# ---------- App & Scheduler ----------
app = FastAPI(title="SeniorCare AI - Backend (Prototype)")
scheduler = BackgroundScheduler()
scheduler.start()

# ---------- Pydantic Schemas ----------
class Contact(BaseModel):
    name: str
    phone: Optional[str]
    email: Optional[str]
    relation: Optional[str]


class CreateUser(BaseModel):
    name: str
    preferred_language: str = "hi"
    phone: Optional[str]
    email: Optional[str]
    emergency_contacts: List[Contact] = Field(default_factory=list)


class ReminderCreate(BaseModel):
    user_id: str
    title: str
    details: Optional[str]
    remind_at: datetime.datetime
    repeat: Optional[str] = None


class SOSPayload(BaseModel):
    user_id: str
    location: Optional[Dict[str, float]] = None  # {lat, lon}
    reason: Optional[str] = "SOS - needs assistance"


# ---------- Utilities: DB helpers ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_user(user: CreateUser):
    db = next(get_db())
    uid = str(uuid.uuid4())
    u = UserModel(
        id=uid,
        name=user.name,
        preferred_language=user.preferred_language,
        phone=user.phone,
        email=user.email,
        emergency_contacts=user.emergency_contacts.json() if hasattr(user, "emergency_contacts") else "[]"
    )
    # emergency_contacts will be stored as text JSON for simplicity
    import json
    u.emergency_contacts = json.dumps([c.dict() for c in user.emergency_contacts])
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def find_user(user_id: str):
    db = next(get_db())
    return db.query(UserModel).filter(UserModel.id == user_id).first()


# ---------- AI / Lang interfaces (provider-agnostic) ----------
def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text -> target_lang.
    Replace the internals with an actual provider (Google Translate, Azure, or local LLM + translator).
    """
    # Placeholder: echo with marker to indicate translation step
    return f"[{target_lang}] {text}"


def synthesize_speech(text: str, lang: str, out_path: str) -> str:
    """
    Create TTS audio file. Use pyttsx3 or cloud TTS in production.
    Returns path to audio file.
    """
    if pyttsx3:
        engine = pyttsx3.init()
        # For simplicity, save to a file is platform dependent; we'll do a stub that writes text bytes
        with open(out_path, "wb") as f:
            f.write(text.encode("utf-8"))
        return out_path
    else:
        # stub
        with open(out_path, "wb") as f:
            f.write(f"[TTS {lang}] {text}".encode("utf-8"))
        return out_path


def speech_to_text_from_audiofile(path: str, lang_code: str = "hi-IN") -> str:
    """
    Convert uploaded audio file to text (STT).
    Replace with real provider call.
    """
    if sr:
        r = sr.Recognizer()
        with sr.AudioFile(path) as source:
            audio = r.record(source)
            try:
                text = r.recognize_google(audio, language=lang_code)
                return text
            except Exception:
                return "[stt-failed] Could not parse audio"
    else:
        return "[stt-stub] Transcribed voice (simulated)"


def ai_interpret_user_text(text: str, user_lang: str) -> Dict[str, Any]:
    """
    Interpret the user's request using an LLM or intent classifier.
    Returns a dict with action, message, suggested_next_steps, severity etc.
    """
    # Stubbed logic for demo
    lower = text.lower()
    if "fall" in lower or "help" in lower or "dizzy" in lower:
        return {"action": "emergency", "message": "Possible fall or emergency detected. Trigger SOS.", "severity": "high"}
    if "medicine" in lower or "remind" in lower:
        return {"action": "set_reminder", "message": "I can set a reminder for your medication.", "severity": "medium"}
    return {"action": "info", "message": "Here's some guidance based on your question.", "severity": "low"}


# ---------- Notification placeholders ----------
def notify_contacts(emergency_payload: SOSPayload):
    """
    Send notifications to pre-defined contacts and emergency services.
    Replace with Twilio (SMS), email, push (FCM), or integration with local emergency services.
    """
    user = find_user(emergency_payload.user_id)
    if not user:
        print("notify_contacts: user not found")
        return

    import json
    contacts = json.loads(user.emergency_contacts or "[]")
    print("NOTIFY: Sending SOS notifications to contacts:", contacts)
    # For demo we print; replace with real SMS/email/push calls.


def notify_family_and_services(user_id: str, message: str, location: Optional[Dict[str, float]] = None):
    print(f"NOTIFY FAMILY: user={user_id}, msg={message}, loc={location}")


# ---------- Reminder scheduling ----------
def schedule_reminder(reminder_id: str, remind_at: datetime.datetime):
    def job():
        db = next(get_db())
        r = db.query(ReminderModel).filter(ReminderModel.id == reminder_id).first()
        if not r or not r.active:
            return
        # Send reminder via TTS, push, or call
        print(f"[REMINDER] {r.user_id} - {r.title} at {datetime.datetime.utcnow()} - details: {r.details}")
        # In prod: call notify endpoint / push notification / phone call

    # APScheduler expects timezone-aware or naive datetimes consistent with scheduler config
    scheduler.add_job(job, 'date', run_date=remind_at, id=reminder_id)


# ---------- Face recognition helpers ----------
def register_face(user_id: str, name: str, image_bytes: bytes) -> Dict[str, Any]:
    # Save image
    fid = str(uuid.uuid4())
    filename = os.path.join(UPLOAD_FOLDER, f"{fid}.jpg")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(filename)

    encoding_serialized = None
    if face_recognition:
        img_arr = face_recognition.load_image_file(filename)
        encs = face_recognition.face_encodings(img_arr)
        if encs:
            import json
            encoding_serialized = json.dumps(encs[0].tolist())
    # Save to DB
    db = next(get_db())
    f = FaceProfileModel(id=fid, user_id=user_id, name=name, image_path=filename, encoding=encoding_serialized)
    db.add(f)
    db.commit()
    return {"face_id": fid, "name": name, "registered": True}


def match_face(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Try to find a matching known face.
    """
    if not face_recognition:
        return {"match": False, "reason": "face_recognition lib not installed; use external API"}

    # load candidate enc
    temp = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid.uuid4().hex}.jpg")
    with open(temp, "wb") as f:
        f.write(image_bytes)
    unknown_img = face_recognition.load_image_file(temp)
    unknown_encs = face_recognition.face_encodings(unknown_img)
    if not unknown_encs:
        return {"match": False}
    unknown = unknown_encs[0]

    db = next(get_db())
    faces = db.query(FaceProfileModel).all()
    for f in faces:
        if not f.encoding:
            continue
        import json, numpy as np
        known = np.array(json.loads(f.encoding))
        matches = face_recognition.compare_faces([known], unknown, tolerance=0.5)
        if matches[0]:
            return {"match": True, "face_id": f.id, "name": f.name}
    return {"match": False}


# ---------- Endpoints ----------

@app.post("/users", response_model=dict)
def create_user(user: CreateUser):
    u = save_user(user)
    return {"user_id": u.id, "name": u.name, "preferred_language": u.preferred_language}


@app.post("/voice-input/{user_id}")
async def voice_input(user_id: str, file: UploadFile = File(...)):
    """
    Voice-first endpoint:
    - Upload an audio file (wav/mp3).
    - Server performs STT in user's language, interprets intent, and returns an answer (text + optional TTS path).
    """
    user = find_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Save file
    content = await file.read()
    fname = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{file.filename}")
    with open(fname, "wb") as f:
        f.write(content)

    # STT (language mapping could be improved)
    lang_code = "hi-IN" if user.preferred_language.startswith("hi") else "en-US"
    transcript = speech_to_text_from_audiofile(fname, lang_code=lang_code)

    # Interpret with AI
    interpretation = ai_interpret_user_text(transcript, user.preferred_language)

    response_text = interpretation.get("message", "I heard you.")
    # translate response back to user preferred language
    translated = translate_text(response_text, user.preferred_language)

    # create TTS audio file
    tts_path = os.path.join(UPLOAD_FOLDER, f"tts_{uuid.uuid4().hex}.bin")
    synthesize_speech(translated, user.preferred_language, tts_path)

    # If action is emergency - trigger SOS flow (non-blocking)
    if interpretation.get("action") == "emergency":
        # simulate SOS
        payload = SOSPayload(user_id=user_id, location=None, reason=transcript)
        BackgroundTasks().add_task(notify_contacts, payload)

    return {"transcript": transcript, "interpretation": interpretation, "response_text": translated, "tts_path": tts_path}


@app.post("/reminders", response_model=dict)
def create_reminder(rem: ReminderCreate):
    db = next(get_db())
    rid = str(uuid.uuid4())
    r = ReminderModel(
        id=rid,
        user_id=rem.user_id,
        title=rem.title,
        details=rem.details,
        remind_at=rem.remind_at,
        repeat=rem.repeat,
        active=True
    )
    db.add(r)
    db.commit()
    # schedule
    schedule_reminder(rid, rem.remind_at)
    return {"reminder_id": rid, "scheduled_for": rem.remind_at.isoformat()}


@app.post("/emergency/sos")
def sos(payload: SOSPayload):
    """
    One-tap SOS endpoint. Should be callable by a physical button or when fall sensor triggers.
    """
    # Immediately notify contacts and emergency services
    notify_contacts(payload)
    # Return simple acknowledgement
    return {"status": "sent", "user_id": payload.user_id}


@app.post("/emergency/fall-detected")
def fall_detected(payload: SOSPayload):
    """
    Called by device sensors or wearables when fall is detected.
    This endpoint simulates immediate notification and optional automatic call.
    """
    # Interpret severity via simple heuristics (could be ML in production)
    notify_contacts(payload)
    return {"status": "fall_received", "action": "contacts_notified"}


@app.post("/memory/face/register")
async def face_register(user_id: str, name: str, file: UploadFile = File(...)):
    content = await file.read()
    result = register_face(user_id, name, content)
    return result


@app.post("/memory/face/match")
async def face_match(file: UploadFile = File(...)):
    content = await file.read()
    result = match_face(content)
    return result


@app.get("/memory/cognitive-game/{user_id}")
def cognitive_game(user_id: str):
    """
    Simple voice-oriented cognitive games (server returns a game to be TTS-d).
    A real app would provide interactive back-and-forth; here we return a task.
    """
    # Example: memory sequence task
    game = {
        "type": "memory_sequence",
        "instructions": "I will say 4 words. After I finish, please repeat them in the same order.",
        "items": ["apple", "train", "river", "lamp"],  # in prod: localized words
        "language": "auto"
    }
    return game


@app.post("/cancer/nearby-centers")
def cancer_nearby(user_id: str, location: Optional[Dict[str, float]] = None, query: Optional[str] = None):
    """
    Placeholder: should query an external healthcare directory or Google Places to return nearby centers.
    For privacy and freshness, call real APIs from production server with proper API keys.
    """
    # Example stubbed response:
    centers = [
        {"name": "Community Cancer Care Center", "distance_km": 2.4, "services": ["Diagnosis", "Chemotherapy"], "address": "MG Road"},
        {"name": "Free Diagnostics Clinic", "distance_km": 4.1, "services": ["Dialysis", "Screening"], "address": "Sector 7"}
    ]
    return {"user_id": user_id, "location": location, "centers": centers}


@app.post("/cancer/personalized-guidance")
def cancer_guidance(user_id: str, condition_summary: Optional[str] = None):
    """
    Provide tailored educational information based on a short summary.
    In production this can call an LLM + trusted medical sources and include citations.
    """
    # stubbed guidance
    guidance = {
        "summary": "Follow-up screening schedule, local support groups, nutrition advice.",
        "notes": "This is educational guidance. For medical decisions, consult your doctor."
    }
    # Translate result to user's language if needed
    user = find_user(user_id)
    if user:
        guidance["summary_local"] = translate_text(guidance["summary"], user.preferred_language)
    return guidance


@app.get("/health/overview/{user_id}")
def health_overview(user_id: str):
    """
    Shows impact across health, mobility, safety and social well-being.
    This is a summary endpoint that could feed the UI (graphs, status).
    """
    # For demo, we return a synthesized health status
    overview = {
        "health": {"medication_adherence": "good", "recent_issues": ["mild fatigue"]},
        "mobility": {"steps_today": 1200, "assistive_devices": ["walker"]},
        "safety": {"last_sos": None, "fall_events_last_month": 0},
        "social": {"contacts_interacted_today": 2, "recommended_activities": ["group walk", "phone call reminder"]}
    }
    return overview


# ---------- Simulated sensor webhook (for wearables) ----------
@app.post("/sensor/webhook")
async def sensor_webhook(data: Dict[str, Any]):
    """
    Sensors (wearables, smart watches) call this webhook when they detect events.
    e.g. { "device_id": "...", "event": "impact", "accel": [0.1, 0.2, 9.8], "user_id": "..."}
    """
    event = data.get("event")
    user_id = data.get("user_id")
    if event == "impact" or event == "fall":
        # decide whether to call fall-detected endpoint
        payload = SOSPayload(user_id=user_id, location=data.get("location"), reason="sensor_fall")
        notify_contacts(payload)
        return {"status": "fall-handled"}
    return {"status": "ignored"}


# ---------- Graceful shutdown ----------
@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown(wait=False)

# ---------- Example root ----------
@app.get("/")
def root():
    return {"message": "SeniorCare AI Backend Prototype - endpoints: /voice-input, /reminders, /emergency/sos, /memory/*, /cancer/*"}

