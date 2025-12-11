from datetime import datetime
from typing import Optional, List, Dict, Any

import math
import json
import base64
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

# --- Deep learning imports ---
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw

import numpy as np

# ------------------------------------------------------------------------------------
# PATHS & CONSTANTS
# ------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28
NUM_LETTERS = 26
SHAPE_LABELS = ["CIRCLE", "SQUARE", "TRIANGLE", "STAR"]
NUM_SHAPES = len(SHAPE_LABELS)
NUM_CLASSES = NUM_LETTERS + NUM_SHAPES

QUALITY_MODEL_PATH = MODELS_DIR / "quality_cnn.pth"

# ------------------------------------------------------------------------------------
# DB CONFIG
# ------------------------------------------------------------------------------------

DATABASE_URL = "mysql+mysqlconnector://root:@localhost:3306/kidcoach"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------------
# SQLAlchemy MODELS
# ------------------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    attempts = relationship("Attempt", back_populates="user")


class PracticeItem(Base):
    __tablename__ = "practice_items"

    id = Column(Integer, primary_key=True, index=True)
    item_type = Column(String(20), nullable=False)  # LETTER | SHAPE
    label = Column(String(50), nullable=False)
    difficulty = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    attempts = relationship("Attempt", back_populates="practice_item")


class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    practice_item_id = Column(Integer, ForeignKey("practice_items.id"), nullable=False)

    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=False)
    duration_seconds = Column(Float, nullable=False)

    score = Column(Float, nullable=False)
    success = Column(Boolean, default=False)

    num_corrections = Column(Integer, default=0)
    num_undos = Column(Integer, default=0)
    shakiness = Column(Float, default=0.0)

    predicted_mood = Column(String(30), nullable=True)
    stress_level = Column(Float, default=0.0)  # 0 to 1

    coach_comment = Column(Text, nullable=True)
    extra_json = Column(Text, nullable=True)  # store JSON string

    user = relationship("User", back_populates="attempts")
    practice_item = relationship("PracticeItem", back_populates="attempts")


# Create tables (dev-only)
Base.metadata.create_all(bind=engine)

# ------------------------------------------------------------------------------------
# Pydantic SCHEMAS
# ------------------------------------------------------------------------------------

class TrajectoryPoint(BaseModel):
    x: float
    y: float
    t: float  # timestamp in seconds relative to start


class AttemptCreate(BaseModel):
    user_name: str
    user_age: Optional[int] = None

    item_type: str  # LETTER or SHAPE
    item_label: str  # e.g. "A", "Circle"

    started_at: datetime
    ended_at: datetime

    num_corrections: int = 0
    num_undos: int = 0

    # Optional raw mood label from front
    face_mood: Optional[str] = None  # HAPPY / NEUTRAL / SAD / etc.

    # Optional final drawing snapshot
    drawing_base64: Optional[str] = None

    # Webcam face snapshots (data:image/png;base64,...)
    face_snapshots: List[str] = Field(default_factory=list)

    trajectory: List[TrajectoryPoint] = Field(default_factory=list)


class AttemptResponse(BaseModel):
    id: int
    score: float
    success: bool
    predicted_mood: Optional[str]
    stress_level: float
    coach_comment: Optional[str]


class SummaryStats(BaseModel):
    total_attempts: int
    total_time_seconds: float
    avg_score: float
    success_rate: float
    mood_counts: Dict[str, int]


class ProgressPoint(BaseModel):
    date: str
    avg_score: float
    total_attempts: int
    total_time_seconds: float


class ProgressStats(BaseModel):
    points: List[ProgressPoint]


class ItemStats(BaseModel):
    item_type: str
    label: str
    attempts: int
    avg_score: float
    success_rate: float
    avg_duration_seconds: float
    avg_stress_level: float


class ItemsStatsResponse(BaseModel):
    items: List[ItemStats]


# ------------------------------------------------------------------------------------
# Drawing / Shakiness / Stress helpers
# ------------------------------------------------------------------------------------

def compute_shakiness(trajectory: List[TrajectoryPoint]) -> float:
    if len(trajectory) < 3:
        return 0.0

    angles = []
    for i in range(1, len(trajectory) - 1):
        x1 = trajectory[i].x - trajectory[i - 1].x
        y1 = trajectory[i].y - trajectory[i - 1].y
        x2 = trajectory[i + 1].x - trajectory[i].x
        y2 = trajectory[i + 1].y - trajectory[i].y

        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue

        dot = x1 * x2 + y1 * y2
        mag1 = math.hypot(x1, y1)
        mag2 = math.hypot(x2, y2)
        cos_theta = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        angle = math.acos(cos_theta)
        angles.append(angle)

    if not angles:
        return 0.0

    mean_angle = sum(angles) / len(angles)
    var = sum((a - mean_angle) ** 2 for a in angles) / len(angles)
    shakiness = math.sqrt(var)

    return min(shakiness / math.pi, 1.0)


def simple_emotion_to_stress(face_mood: Optional[str]) -> float:
    if face_mood is None:
        return 0.3

    mood = face_mood.upper()
    if mood in ("HAPPY", "CALM", "RELAXED"):
        return 0.1
    if mood in ("NEUTRAL", "SURPRISED"):
        return 0.3
    if mood in ("SAD", "CONFUSED"):
        return 0.6
    if mood in ("ANGRY", "FEAR", "DISGUST", "FRUSTRATED"):
        return 0.8
    return 0.3


def score_quality_placeholder(item_type: str, item_label: str, trajectory: List[TrajectoryPoint]) -> float:
    """
    Simple readability score (0..100) based on path length.
    """
    if not trajectory:
        return 0.0

    length = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i].x - trajectory[i - 1].x
        dy = trajectory[i].y - trajectory[i - 1].y
        length += math.hypot(dx, dy)

    base_score = length * 180.0  # scale up a bit

    if item_type.upper() == "LETTER":
        if length < 0.3:
            base_score *= 0.6
        elif length > 1.5:
            base_score *= 0.8

    if length > 0.1:
        base_score = max(base_score, 20.0)

    return float(max(0.0, min(100.0, base_score)))


# ------------------------------------------------------------------------------------
# DEEP LEARNING: Quality CNN for letters/shapes
# ------------------------------------------------------------------------------------

class QualityCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


_quality_model: Optional[QualityCNN] = None
_quality_device = torch.device("cpu")
_quality_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # (H,W)->(1,H,W)
    ]
)


def _load_quality_model():
    global _quality_model
    if not QUALITY_MODEL_PATH.exists():
        print(f"[WARN] Quality model not found at {QUALITY_MODEL_PATH}. Using heuristic scoring only.")
        _quality_model = None
        return

    model = QualityCNN(num_classes=NUM_CLASSES)
    state_dict = torch.load(QUALITY_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _quality_model = model.to(_quality_device)
    print(f"[INFO] Loaded quality model from {QUALITY_MODEL_PATH}")


def letter_shape_to_class_index(item_type: str, item_label: str) -> Optional[int]:
    t = item_type.strip().upper()

    if t == "LETTER":
        if not item_label:
            return None
        ch = item_label.strip().upper()[0]
        if not ("A" <= ch <= "Z"):
            return None
        return ord(ch) - ord("A")

    if t == "SHAPE":
        name = item_label.strip().upper()
        if "CIR" in name:
            name = "CIRCLE"
        elif "SQU" in name or "CARRE" in name:
            name = "SQUARE"
        elif "TRI" in name:
            name = "TRIANGLE"
        elif "STAR" in name or "ETOILE" in name:
            name = "STAR"

        if name not in SHAPE_LABELS:
            return None
        return NUM_LETTERS + SHAPE_LABELS.index(name)

    return None


def render_trajectory_to_image(trajectory: List[TrajectoryPoint]) -> Image.Image:
    """
    Convert normalized trajectory points (x,y in [0,1]) to a 28x28 grayscale image.
    We CENTER and SCALE the stroke so it looks more like EMNIST letters.
    """
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    if len(trajectory) < 2:
        return img

    xs = [p.x for p in trajectory]
    ys = [p.y for p in trajectory]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return img

    # Use ~80% of the canvas for the bounding box
    margin = 2
    target_size = IMG_SIZE - 2 * margin
    scale = target_size / max(width, height)

    bbox_width_px = width * scale
    bbox_height_px = height * scale

    offset_x = (IMG_SIZE - bbox_width_px) / 2.0
    offset_y = (IMG_SIZE - bbox_height_px) / 2.0

    # If you want to flip horizontally, swap (1 - p.x) here
    coords = [
        (
            (p.x - min_x) * scale + offset_x,
            (p.y - min_y) * scale + offset_y,
        )
        for p in trajectory
    ]

    draw = ImageDraw.Draw(img)
    draw.line(coords, fill=255, width=2)
    return img


def compute_quality_score_and_match(
    item_type: str,
    item_label: str,
    trajectory: List[TrajectoryPoint],
):
    """
    Returns:
      score (0..100),
      match_conf (raw CNN prob for target),
      match_pct (0..100 after smoothing),
      pred_label (A..Z or shape),
      pred_conf (model's best guess probability)
    """
    heur = score_quality_placeholder(item_type, item_label, trajectory)

    if _quality_model is None or not trajectory:
        print(
            f"[QUALITY] type={item_type} label={item_label} len={len(trajectory)} "
            f"heur={heur:.2f} match_conf=None pred_label=None pred_conf=None"
        )
        return heur, None, None, None, None

    class_idx = letter_shape_to_class_index(item_type, item_label)
    img = render_trajectory_to_image(trajectory)
    x = _quality_transform(img).unsqueeze(0).to(_quality_device)

    with torch.no_grad():
        logits = _quality_model(x)
        probs = torch.softmax(logits, dim=1)
        best_idx = int(probs.argmax(dim=1).item())
        best_prob = float(probs[0, best_idx].item())

        if class_idx is not None:
            target_prob = float(probs[0, class_idx].item())
        else:
            target_prob = None

    # Decode predicted label name
    if best_idx < NUM_LETTERS:
        pred_label = chr(ord("A") + best_idx)
    else:
        s_idx = best_idx - NUM_LETTERS
        if 0 <= s_idx < len(SHAPE_LABELS):
            pred_label = SHAPE_LABELS[s_idx]
        else:
            pred_label = f"IDX_{best_idx}"

    # === NEW: smoothed match factor ===
    if target_prob is None:
        # No known mapping; just use readability × how "letter-like" it is
        alignment = best_prob
        match_conf = None
        match_pct = None
    else:
        # target_prob is typically tiny with air-letters → boost it nonlinearly
        # base comes from target_prob, bonus from best_prob
        eps = 1e-4
        base = (target_prob + eps) ** 0.35      # 0.0008 → ~0.08 instead of 0
        bonus = 0.15 * best_prob                # add a bit of "you drew *some* letter"
        alignment = min(1.0, base + bonus)      # clip to [0,1]

        match_conf = target_prob
        match_pct = alignment * 100.0

    score = float(max(0.0, min(100.0, heur * alignment)))

    try:
        mc = match_conf if match_conf is not None else 0.0
        print(
            f"[QUALITY] type={item_type} label={item_label} len={len(trajectory)} "
            f"heur={heur:.2f} target_prob={mc:.6f} best_label={pred_label} best_prob={best_prob:.6f} "
            f"alignment={alignment:.3f} score={score:.2f}"
        )
    except Exception:
        pass

    return score, match_conf, match_pct, pred_label, best_prob


# load model at import
_load_quality_model()

# ------------------------------------------------------------------------------------
# DEEP LEARNING: Face emotion from webcam snapshots (FER)
# ------------------------------------------------------------------------------------

try:
    import fer as fer_module

    FER = getattr(fer_module, "FER", None)
    if FER is None:
        print("[WARN] `fer` package imported but has no FER class. Face-based mood will be disabled.")
except Exception as e:
    FER = None
    print(f"[WARN] Could not import `fer` package: {e}")

_fer_detector = None
if FER is not None:
    try:
        _fer_detector = FER(mtcnn=False)
        print("[INFO] FER emotion model initialized.")
    except Exception as e:
        print(f"[WARN] Could not initialize FER: {e}")
        _fer_detector = None
else:
    print("[WARN] `fer` library not installed or unusable. Face-based mood will not be inferred.")


def _decode_base64_image(b64: str) -> Optional[np.ndarray]:
    try:
        if b64.startswith("data:image"):
            _, b64 = b64.split(",", 1)
        data = base64.b64decode(b64)
        img = Image.open(BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"[WARN] Could not decode face snapshot: {e}")
        return None


def predict_mood_from_snapshots(face_snapshots: List[str]) -> Optional[str]:
    if not face_snapshots or _fer_detector is None:
        return None

    emotions_sum: Dict[str, float] = {}
    count = 0

    for b64 in face_snapshots[:5]:
        arr = _decode_base64_image(b64)
        if arr is None:
            continue

        try:
            res = _fer_detector.detect_emotions(arr)
        except Exception as e:
            print(f"[WARN] FER detection error: {e}")
            continue

        if not res:
            continue

        emo = res[0].get("emotions", {})
        if not emo:
            continue

        count += 1
        for k, v in emo.items():
            emotions_sum[k] = emotions_sum.get(k, 0.0) + float(v)

    if not emotions_sum or count == 0:
        return None

    for k in list(emotions_sum.keys()):
        emotions_sum[k] /= count

    best_label, best_val = max(emotions_sum.items(), key=lambda kv: kv[1])

    mapping = {
        "happy": "HAPPY",
        "neutral": "NEUTRAL",
        "sad": "SAD",
        "angry": "ANGRY",
        "fear": "FEAR",
        "disgust": "DISGUST",
        "surprise": "SURPRISED",
    }
    mood = mapping.get(best_label.lower(), best_label.upper())
    print(f"[MOOD] FER predicted {mood} (p={best_val:.3f}) over {count} frames")
    return mood


# ------------------------------------------------------------------------------------
# FastAPI APP
# ------------------------------------------------------------------------------------

app = FastAPI(title="Kid Drawing Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_or_create_user(db: Session, name: str, age: Optional[int]) -> User:
    user = db.query(User).filter(User.name == name).first()
    if user:
        return user
    user = User(name=name, age=age)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_or_create_item(db: Session, item_type: str, label: str) -> PracticeItem:
    item = (
        db.query(PracticeItem)
        .filter(PracticeItem.item_type == item_type.upper(), PracticeItem.label == label)
        .first()
    )
    if item:
        return item
    item = PracticeItem(item_type=item_type.upper(), label=label, difficulty=1)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def build_coach_comment(score: float, shakiness: float, stress: float) -> str:
    parts: List[str] = []

    if score >= 80:
        parts.append("Great job! Your drawing looks very good.")
    elif score >= 50:
        parts.append("Nice work! A bit more practice and it will be perfect.")
    else:
        parts.append("Keep trying, you’re learning! Let’s draw a bit slower next time.")

    if shakiness > 0.6:
        parts.append("Try to move your hand more smoothly without quick jumps.")
    elif shakiness > 0.3:
        parts.append("Your lines are okay, but you can make them even smoother.")

    if stress > 0.6:
        parts.append("Take a deep breath and relax, drawing should be fun 😊.")
    elif stress < 0.3:
        parts.append("You seem relaxed, that’s great for learning!")

    return " ".join(parts)


@app.post("/api/attempts", response_model=AttemptResponse)
def create_attempt(payload: AttemptCreate, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.user_name, payload.user_age)
    item = get_or_create_item(db, payload.item_type, payload.item_label)

    duration = (payload.ended_at - payload.started_at).total_seconds()
    shakiness = compute_shakiness(payload.trajectory)

    # 1) Mood: FER on snapshots if any
    mood = predict_mood_from_snapshots(payload.face_snapshots)

    if not payload.face_snapshots:
        print("[MOOD] No face snapshots received; falling back to NEUTRAL.")

    # Fallback: use explicit label from front if FER failed
    if mood is None and payload.face_mood:
        mood = payload.face_mood

    # Final fallback so UI never shows Unknown
    if mood is None:
        mood = "NEUTRAL"

    # 2) Stress from mood + shakiness
    stress_from_face = simple_emotion_to_stress(mood)
    stress_level = min(max(stress_from_face + shakiness * 0.5, 0.0), 1.0)

    # 3) Drawing quality score (readability × match)
    score, match_conf, match_pct, pred_label, pred_conf = compute_quality_score_and_match(
        payload.item_type,
        payload.item_label,
        payload.trajectory,
    )

    success = score >= 40.0
    coach_comment = build_coach_comment(score, shakiness, stress_level)

    extra: Dict[str, Any] = {
        "raw_face_mood_from_front": payload.face_mood,
        "final_mood_used": mood,
        "num_points": len(payload.trajectory),
        "drawing_base64_saved": bool(payload.drawing_base64),
        "num_face_snapshots": len(payload.face_snapshots),
        "match_conf": match_conf,
        "match_pct": match_pct,
        "predicted_label": pred_label,
        "predicted_label_conf": pred_conf,
    }

    attempt = Attempt(
        user_id=user.id,
        practice_item_id=item.id,
        started_at=payload.started_at,
        ended_at=payload.ended_at,
        duration_seconds=duration,
        score=score,
        success=success,
        num_corrections=payload.num_corrections,
        num_undos=payload.num_undos,
        shakiness=shakiness,
        predicted_mood=mood,
        stress_level=stress_level,
        coach_comment=coach_comment,
        extra_json=json.dumps(extra),
    )

    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    try:
        mc = match_conf if match_conf is not None else 0.0
        pc = pred_conf if pred_conf is not None else 0.0
        print(
            f"[ATTEMPT] id={attempt.id}"
            f" score={attempt.score:.2f}"
            f" success={attempt.success}"
            f" mood={attempt.predicted_mood}"
            f" stress={attempt.stress_level:.2f}"
            f" points={len(payload.trajectory)}"
            f" snapshots={len(payload.face_snapshots)}"
            f" target_prob={mc:.6f}"
            f" pred_label={pred_label}"
            f" pred_conf={pc:.6f}"
        )
    except Exception:
        pass

    return AttemptResponse(
        id=attempt.id,
        score=attempt.score,
        success=attempt.success,
        predicted_mood=attempt.predicted_mood,
        stress_level=attempt.stress_level,
        coach_comment=attempt.coach_comment,
    )


@app.get("/api/stats/summary", response_model=SummaryStats)
def get_summary_stats(user_name: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Attempt)
    if user_name:
        user = db.query(User).filter(User.name == user_name).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        query = query.filter(Attempt.user_id == user.id)

    attempts = query.all()
    if not attempts:
        return SummaryStats(
            total_attempts=0,
            total_time_seconds=0.0,
            avg_score=0.0,
            success_rate=0.0,
            mood_counts={},
        )

    total_attempts = len(attempts)
    total_time = sum(a.duration_seconds for a in attempts)
    avg_score = sum(a.score for a in attempts) / total_attempts
    success_rate = sum(1 for a in attempts if a.success) / total_attempts * 100.0

    mood_counts: Dict[str, int] = {}
    for a in attempts:
        mood = a.predicted_mood or "UNKNOWN"
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    return SummaryStats(
        total_attempts=total_attempts,
        total_time_seconds=total_time,
        avg_score=avg_score,
        success_rate=success_rate,
        mood_counts=mood_counts,
    )


@app.get("/api/stats/progress", response_model=ProgressStats)
def get_progress_stats(user_name: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Attempt)
    if user_name:
        user = db.query(User).filter(User.name == user_name).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        query = query.filter(Attempt.user_id == user.id)

    attempts = query.order_by(Attempt.started_at.asc()).all()
    buckets: Dict[str, List[Attempt]] = {}

    for a in attempts:
        key = a.started_at.date().isoformat()
        buckets.setdefault(key, []).append(a)

    points: List[ProgressPoint] = []
    for date_str, bucket in sorted(buckets.items()):
        total_attempts = len(bucket)
        total_time = sum(a.duration_seconds for a in bucket)
        avg_score = sum(a.score for a in bucket) / total_attempts
        points.append(
            ProgressPoint(
                date=date_str,
                avg_score=avg_score,
                total_attempts=total_attempts,
                total_time_seconds=total_time,
            )
        )

    return ProgressStats(points=points)


@app.get("/api/stats/by-item", response_model=ItemsStatsResponse)
def get_stats_by_item(user_name: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Attempt).join(PracticeItem)
    if user_name:
        user = db.query(User).filter(User.name == user_name).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        query = query.filter(Attempt.user_id == user.id)

    attempts = query.all()
    by_item: Dict[int, List[Attempt]] = {}
    item_map: Dict[int, PracticeItem] = {}

    for a in attempts:
        pid = a.practice_item_id
        by_item.setdefault(pid, []).append(a)
        item_map[pid] = a.practice_item

    items_stats: List[ItemStats] = []
    for pid, bucket in by_item.items():
        item = item_map[pid]
        attempts_count = len(bucket)
        avg_score = sum(a.score for a in bucket) / attempts_count
        success_rate = sum(1 for a in bucket if a.success) / attempts_count * 100.0
        avg_duration = sum(a.duration_seconds for a in bucket) / attempts_count
        avg_stress = sum(a.stress_level for a in bucket) / attempts_count

        items_stats.append(
            ItemStats(
                item_type=item.item_type,
                label=item.label,
                attempts=attempts_count,
                avg_score=avg_score,
                success_rate=success_rate,
                avg_duration_seconds=avg_duration,
                avg_stress_level=avg_stress,
            )
        )

    return ItemsStatsResponse(items=items_stats)
