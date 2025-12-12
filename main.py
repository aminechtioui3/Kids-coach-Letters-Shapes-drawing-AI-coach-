from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

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
    avg_stress_level: float


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


class UserSummary(BaseModel):
    name: str
    age: Optional[int]
    created_at: datetime
    total_attempts: int
    avg_score: float
    avg_stress_level: float
    last_attempt_at: Optional[datetime]


class AttemptHistoryItem(BaseModel):
    id: int
    user_name: str
    item_type: str
    item_label: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    score: float
    success: bool
    predicted_mood: Optional[str]
    stress_level: float


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

    base_score = length * 180.0  # scale up

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


def canonical_shape_label(label: str) -> Optional[str]:
    if not label:
        return None
    name = label.strip().upper()
    if "CIR" in name:
        return "CIRCLE"
    if "SQU" in name or "CARRE" in name:
        return "SQUARE"
    if "TRI" in name:
        return "TRIANGLE"
    if "STAR" in name or "ETOILE" in name:
        return "STAR"
    if name in SHAPE_LABELS:
        return name
    return None


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
        name = canonical_shape_label(item_label)
        if name is None:
            return None
        return NUM_LETTERS + SHAPE_LABELS.index(name)

    return None


def render_trajectory_to_image(trajectory: List[TrajectoryPoint]) -> Image.Image:
    """
    Convert normalized trajectory points (x,y in [0,1]) to a 28x28 grayscale image.
    Center + scale to look more like EMNIST strokes.
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

    margin = 2
    target_size = IMG_SIZE - 2 * margin
    scale = target_size / max(width, height)

    bbox_width_px = width * scale
    bbox_height_px = height * scale

    offset_x = (IMG_SIZE - bbox_width_px) / 2.0
    offset_y = (IMG_SIZE - bbox_height_px) / 2.0

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


def generate_saliency_image(
    item_type: str,
    item_label: str,
    trajectory: List[TrajectoryPoint],
) -> Tuple[str, Optional[str], float, float]:
    """
    Build a saliency heatmap for the drawing, showing which pixels
    influenced the CNN prediction most.

    Returns:
      (data_url_png, predicted_label, predicted_conf, match_conf_for_target)
    """
    if _quality_model is None:
        raise RuntimeError("Quality model not loaded")

    if not trajectory:
        raise RuntimeError("Empty trajectory")

    class_idx = letter_shape_to_class_index(item_type, item_label)
    img = render_trajectory_to_image(trajectory)
    x = _quality_transform(img).unsqueeze(0).to(_quality_device)
    x.requires_grad_(True)

    _quality_model.zero_grad()
    logits = _quality_model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())
    match_conf = float(probs[class_idx].item()) if class_idx is not None else 0.0

    # grad of predicted logit w.r.t input pixels
    score_logit = logits[0, pred_idx]
    score_logit.backward()
    grad = x.grad[0, 0]  # (28,28)
    saliency = grad.abs()
    max_val = float(saliency.max().item())
    if max_val > 0:
        saliency = saliency / max_val

    sal_np = saliency.cpu().numpy()

    # upscale to 224x224 and overlay as red heatmap
    size = 224
    base_img = img.resize((size, size)).convert("L")
    base_rgb = Image.merge("RGB", (base_img, base_img, base_img))
    heat = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(heat)

    h_src, w_src = sal_np.shape
    for y in range(size):
        src_y = int(y * h_src / size)
        for x in range(size):
            src_x = int(x * w_src / size)
            val = float(sal_np[src_y, src_x])
            if val <= 0.05:
                continue
            alpha = int(80 + 175 * val)  # 80-255
            red = 255
            green = int(64 + 80 * (1.0 - val))
            blue = 0
            draw.point((x, y), (red, green, blue, alpha))

    combined = Image.alpha_composite(base_rgb.convert("RGBA"), heat)

    buf = BytesIO()
    combined.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    pred_label: Optional[str] = None
    if pred_idx < NUM_LETTERS:
        pred_label = chr(ord("A") + pred_idx)
    else:
        shape_idx = pred_idx - NUM_LETTERS
        if 0 <= shape_idx < len(SHAPE_LABELS):
            pred_label = SHAPE_LABELS[shape_idx]

    data_url = f"data:image/png;base64,{b64}"
    return data_url, pred_label, pred_conf, match_conf


def shape_match_score(shape_name: str, trajectory: List[TrajectoryPoint]) -> float:
    """
    Heuristic [0..1] of how much the trajectory looks like a given SHAPE.
    """
    if not trajectory or shape_name not in SHAPE_LABELS:
        return 0.0
    if len(trajectory) < 5:
        return 0.0

    xs = [p.x for p in trajectory]
    ys = [p.y for p in trajectory]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return 0.0

    aspect = min(width, height) / max(width, height)

    def aspect_score_fn(a: float) -> float:
        if a <= 0.4:
            return 0.0
        if a >= 0.9:
            return 1.0
        return (a - 0.4) / (0.9 - 0.4)

    if shape_name == "CIRCLE":
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        rs = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
        r_mean = sum(rs) / len(rs)
        if r_mean <= 0:
            return 0.0
        r_var = sum((r - r_mean) ** 2 for r in rs) / len(rs)
        r_std = math.sqrt(r_var)
        rel_std = r_std / r_mean
        if rel_std >= 0.5:
            radial_score = 0.0
        else:
            radial_score = 1.0 - (rel_std / 0.5)

        return 0.6 * aspect_score_fn(aspect) + 0.4 * radial_score

    seg_angles: List[float] = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i].x - trajectory[i - 1].x
        dy = trajectory[i].y - trajectory[i - 1].y
        if dx == 0 and dy == 0:
            continue
        seg_angles.append(abs(math.atan2(dy, dx)))

    if not seg_angles:
        return 0.0

    if shape_name == "SQUARE":
        axis_scores = []
        for ang in seg_angles:
            mod = ang % (math.pi / 2)
            dev = min(mod, (math.pi / 2) - mod)
            if dev >= (math.pi / 8):
                axis_scores.append(0.0)
            else:
                axis_scores.append(1.0 - dev / (math.pi / 8))
        axis_align = sum(axis_scores) / len(axis_scores) if axis_scores else 0.0
        return 0.5 * aspect_score_fn(aspect) + 0.5 * axis_align

    if shape_name == "TRIANGLE" or shape_name == "STAR":
        corners = 0
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
            if mag1 == 0 or mag2 == 0:
                continue
            cos_theta = max(min(dot / (mag1 * mag2), 1.0), -1.0)
            angle = math.acos(cos_theta)
            if angle > math.pi / 4:
                corners += 1

        if shape_name == "TRIANGLE":
            return math.exp(-((corners - 3) ** 2) / 4.0)

        if shape_name == "STAR":
            return math.exp(-((corners - 8) ** 2) / 8.0)

    return 0.0


def compute_quality_score_and_match(
    item_type: str,
    item_label: str,
    trajectory: List[TrajectoryPoint],
):
    """
    Returns:
      score (0..100)             -> final score (quality × match to target)
      match_conf (0..1 or None)  -> raw CNN probability for the target class
      match_pct (0..100 or None) -> nicer % version of match_conf
      pred_label (str or None)   -> best letter/shape according to CNN
      pred_conf (0..1 or None)   -> confidence of pred_label
    """

    # ===============================
    # 1) Heuristic "quality" based on path length
    # ===============================
    heur = score_quality_placeholder(item_type, item_label, trajectory)

    # If there is some drawing but heur is tiny, give a small floor so
    # the child still gets a non-zero base score.
    if trajectory and heur < 5.0:
        heur = 20.0

    # If no model or no trajectory: fallback to heuristics only
    if _quality_model is None or not trajectory:
        print(
            f"[QUALITY] type={item_type} label={item_label} len={len(trajectory)} "
            f"heur={heur:.2f} (model_off) match_conf=None pred_label=None pred_conf=None"
        )
        return heur, None, None, None, None

    # ===============================
    # 2) CNN prediction
    # ===============================
    class_idx = letter_shape_to_class_index(item_type, item_label)

    img = render_trajectory_to_image(trajectory)
    x = _quality_transform(img).unsqueeze(0).to(_quality_device)

    with torch.no_grad():
        logits = _quality_model(x)
        probs = torch.softmax(logits, dim=1)[0]

    best_idx = int(probs.argmax().item())
    best_prob = float(probs[best_idx].item())

    # Probability for the *target* class (= how close to chosen letter/shape)
    target_prob: Optional[float]
    if class_idx is not None:
        target_prob = float(probs[class_idx].item())
    else:
        target_prob = None  # e.g. unknown label

    # Decode best label for display
    if best_idx < NUM_LETTERS:
        pred_label = chr(ord("A") + best_idx)
    else:
        s_idx = best_idx - NUM_LETTERS
        if 0 <= s_idx < len(SHAPE_LABELS):
            pred_label = SHAPE_LABELS[s_idx]
        else:
            pred_label = f"IDX_{best_idx}"

    # ===============================
    # 3) Extra heuristic for SHAPES
    # ===============================
    shape_alignment: Optional[float] = None
    if item_type.strip().upper() == "SHAPE":
        canon = canonical_shape_label(item_label)
        if canon is not None:
            # 0..1: how round / square / triangular etc. the trajectory looks
            shape_alignment = shape_match_score(canon, trajectory)

    # ===============================
    # 4) Turn CNN confidence → alignment factor [0.15 .. 1.0]
    #    so score never collapses to 0 but still rewards correct letters most.
    # ===============================
    if target_prob is not None:
        match_raw = max(0.0, min(1.0, target_prob))
    else:
        # If the label is weird, use the best class as a fallback
        match_raw = max(0.0, min(1.0, best_prob))

    # base_alignment grows with sqrt of probability:
    #  - target_prob ~ 0.0  -> ~0.15 (child still gets some points)
    #  - target_prob ~ 0.25 -> ~0.6
    #  - target_prob ~ 0.60 -> ~0.85
    base_alignment = 0.15 + 0.85 * math.sqrt(match_raw)

    if shape_alignment is not None:
        # For shapes we also blend in geometric similarity
        alignment = min(1.0, 0.5 * base_alignment + 0.5 * shape_alignment)
    else:
        alignment = base_alignment

    # ===============================
    # 5) Final score: quality × alignment
    # ===============================
    score = float(max(0.0, min(100.0, heur * alignment)))
    match_conf = target_prob
    match_pct = match_raw * 100.0 if target_prob is not None else None

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


class ExplainRequest(BaseModel):
  item_type: str
  item_label: str
  trajectory: List[TrajectoryPoint]


class ExplainResponseModel(BaseModel):
  saliency_image: str
  predicted_label: Optional[str]
  predicted_confidence: float
  match_confidence: float


@app.post("/api/explain", response_model=ExplainResponseModel)
def explain_attempt(payload: ExplainRequest):
    if not payload.trajectory:
        raise HTTPException(status_code=400, detail="Empty trajectory")
    if _quality_model is None:
        raise HTTPException(
            status_code=503,
            detail="Quality model not loaded on server.",
        )

    try:
        img_b64, pred_label, pred_conf, match_conf = generate_saliency_image(
            payload.item_type,
            payload.item_label,
            payload.trajectory,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ExplainResponseModel(
        saliency_image=img_b64,
        predicted_label=pred_label,
        predicted_confidence=pred_conf,
        match_confidence=match_conf,
    )


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

    # Fallback: explicit label from front if FER failed
    if mood is None and payload.face_mood:
        mood = payload.face_mood

    # Final fallback so UI never shows Unknown
    if mood is None:
        mood = "NEUTRAL"

    # 2) Stress from mood + shakiness
    stress_from_face = simple_emotion_to_stress(mood)
    stress_level = min(max(stress_from_face + shakiness * 0.5, 0.0), 1.0)

    # 3) Drawing quality score (readability × match with selected item)
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
        mood_val = a.predicted_mood or "UNKNOWN"
        mood_counts[mood_val] = mood_counts.get(mood_val, 0) + 1

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
        avg_stress = sum(a.stress_level for a in bucket) / total_attempts
        points.append(
            ProgressPoint(
                date=date_str,
                avg_score=avg_score,
                total_attempts=total_attempts,
                total_time_seconds=total_time,
                avg_stress_level=avg_stress,
            )
        )

    return ProgressStats(points=points)


class FrustratingItem(BaseModel):
    item_type: str
    label: str
    attempts: int
    avg_score: float
    avg_stress: float


class SuggestionModel(BaseModel):
    item_type: str
    label: str
    reason: str


class StudentGamification(BaseModel):
    level: int
    stars: int
    badges: List[str]
    today_attempts: int
    today_goal: int


class StudentInsightsResponse(BaseModel):
    user_name: str
    avg_score: float
    avg_stress: float
    success_rate: float
    frustrating_items: List[FrustratingItem]
    suggestions: List[SuggestionModel]
    gamification: StudentGamification


def _compute_student_insights(
    user: User, db: Session
) -> StudentInsightsResponse:
    attempts = (
        db.query(Attempt)
        .join(PracticeItem, Attempt.practice_item_id == PracticeItem.id)
        .filter(Attempt.user_id == user.id)
        .order_by(Attempt.started_at.asc())
        .all()
    )

    if not attempts:
        gam = StudentGamification(
            level=1,
            stars=0,
            badges=[],
            today_attempts=0,
            today_goal=5,
        )
        return StudentInsightsResponse(
            user_name=user.name,
            avg_score=0.0,
            avg_stress=0.0,
            success_rate=0.0,
            frustrating_items=[],
            suggestions=[],
            gamification=gam,
        )

    total = len(attempts)
    avg_score = sum(a.score for a in attempts) / total
    avg_stress = sum(a.stress_level for a in attempts) / total
    success_rate = sum(1 for a in attempts if a.success) / total * 100.0

    # Count attempts for today
    today = datetime.utcnow().date()
    today_attempts = sum(1 for a in attempts if a.started_at.date() == today)

    # Group by item
    by_item: Dict[int, List[Attempt]] = {}
    item_map: Dict[int, PracticeItem] = {}
    for a in attempts:
        pid = a.practice_item_id
        by_item.setdefault(pid, []).append(a)
        item_map[pid] = a.practice_item

    frustrating: List[FrustratingItem] = []
    letter_best_score: Dict[str, float] = {}
    letter_counts: Dict[str, int] = {}

    for pid, bucket in by_item.items():
        item = item_map[pid]
        n = len(bucket)
        item_avg_score = sum(a.score for a in bucket) / n
        item_avg_stress = sum(a.stress_level for a in bucket) / n

        # frustration: low score + high stress + enough attempts
        if item_avg_score < 50.0 and item_avg_stress > 0.5 and n >= 3:
            frustrating.append(
                FrustratingItem(
                    item_type=item.item_type,
                    label=item.label,
                    attempts=n,
                    avg_score=item_avg_score,
                    avg_stress=item_avg_stress,
                )
            )

        if item.item_type.upper() == "LETTER":
            lbl = (item.label or "").strip().upper()
            if not lbl:
                continue
            letter_counts[lbl] = letter_counts.get(lbl, 0) + n
            best_old = letter_best_score.get(lbl, 0.0)
            if item_avg_score > best_old:
                letter_best_score[lbl] = item_avg_score

    # Sort frustrating by "worst first"
    frustrating.sort(
        key=lambda f: (f.avg_stress, -f.avg_score),
        reverse=True,
    )

    suggestions: List[SuggestionModel] = []
    if frustrating:
        for f in frustrating[:3]:
            suggestions.append(
                SuggestionModel(
                    item_type=f.item_type,
                    label=f.label,
                    reason="High stress and low score. Good candidate for extra practice.",
                )
            )
    else:
        # fallback: suggest most frequently practiced letters
        for lbl, cnt in sorted(
            letter_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:3]:
            suggestions.append(
                SuggestionModel(
                    item_type="LETTER",
                    label=lbl,
                    reason="Frequently practiced letter. Keep consolidating this skill.",
                )
            )

    # Gamification
    level = max(1, min(5, int(avg_score // 20) + 1))
    stars = min(4, int(success_rate // 25))

    badges: List[str] = []
    avg_shakiness = sum(a.shakiness for a in attempts) / total

    if avg_score >= 70.0 and total >= 10:
        badges.append("Consistent Learner")
    if avg_stress <= 0.35 and total >= 5:
        badges.append("Calm Artist")
    if avg_shakiness <= 0.35 and total >= 5:
        badges.append("Smooth Lines")

    mastered_letters = [
        lbl
        for lbl, s in letter_best_score.items()
        if "A" <= lbl <= "F" and s >= 75.0
    ]
    if len(mastered_letters) >= 3:
        badges.append("Letter Master A–F")

    gam = StudentGamification(
        level=level,
        stars=stars,
        badges=badges,
        today_attempts=today_attempts,
        today_goal=5,
    )

    return StudentInsightsResponse(
        user_name=user.name,
        avg_score=avg_score,
        avg_stress=avg_stress,
        success_rate=success_rate,
        frustrating_items=frustrating,
        suggestions=suggestions,
        gamification=gam,
    )


@app.get("/api/insights/student", response_model=StudentInsightsResponse)
def get_student_insights(
    user_name: str, db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.name == user_name).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return _compute_student_insights(user, db)


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


@app.get("/api/users/with-stats", response_model=List[UserSummary])
def get_users_with_stats(db: Session = Depends(get_db)):
    users = db.query(User).all()
    summaries: List[UserSummary] = []

    for u in users:
        atts = db.query(Attempt).filter(Attempt.user_id == u.id).all()
        if atts:
            total = len(atts)
            avg_score = sum(a.score for a in atts) / total
            avg_stress = sum(a.stress_level for a in atts) / total
            last_at = max(a.started_at for a in atts)
        else:
            total = 0
            avg_score = 0.0
            avg_stress = 0.0
            last_at = None

        summaries.append(
            UserSummary(
                name=u.name,
                age=u.age,
                created_at=u.created_at,
                total_attempts=total,
                avg_score=avg_score,
                avg_stress_level=avg_stress,
                last_attempt_at=last_at,
            )
        )

    return summaries


@app.get("/api/attempts/history", response_model=List[AttemptHistoryItem])
def get_attempts_history(
    user_name: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = (
        db.query(Attempt, User, PracticeItem)
        .join(User, Attempt.user_id == User.id)
        .join(PracticeItem, Attempt.practice_item_id == PracticeItem.id)
    )
    if user_name:
        query = query.filter(User.name == user_name)

    rows = query.order_by(Attempt.started_at.desc()).all()
    items: List[AttemptHistoryItem] = []

    for a, u, item in rows:
        items.append(
            AttemptHistoryItem(
                id=a.id,
                user_name=u.name,
                item_type=item.item_type,
                item_label=item.label,
                started_at=a.started_at,
                ended_at=a.ended_at,
                duration_seconds=a.duration_seconds,
                score=a.score,
                success=a.success,
                predicted_mood=a.predicted_mood,
                stress_level=a.stress_level,
            )
        )

    return items
