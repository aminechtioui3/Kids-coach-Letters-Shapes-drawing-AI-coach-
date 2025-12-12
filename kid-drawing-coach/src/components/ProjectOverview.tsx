// src/components/ProjectOverview.tsx
import React, { useState } from 'react';

type ModalState = {
  open: boolean;
  title: string;
  subtitle: string;
  code: string;
};

// Code snippets for modals
const MAIN_SNIPPET = `from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

@app.post("/api/attempts")
def create_attempt(payload: AttemptCreate, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.user_name, payload.user_age)
    item = get_or_create_item(db, payload.item_type, payload.item_label)

    duration = (payload.ended_at - payload.started_at).total_seconds()
    shakiness = compute_shakiness(payload.trajectory)

    mood = predict_mood_from_snapshots(payload.face_snapshots)
    if mood is None and payload.face_mood:
        mood = payload.face_mood
    if mood is None:
        mood = "NEUTRAL"

    stress = simple_emotion_to_stress(mood) + shakiness * 0.5

    score, match_conf, match_pct, pred_label, pred_conf = compute_quality_score_and_match(
        payload.item_type,
        payload.item_label,
        payload.trajectory,
    )

    success = score >= 40.0
    comment = build_coach_comment(score, shakiness, stress)

    attempt = Attempt(
        user_id=user.id,
        practice_item_id=item.id,
        started_at=payload.started_at,
        ended_at=payload.ended_at,
        duration_seconds=duration,
        score=score,
        success=success,
        shakiness=shakiness,
        predicted_mood=mood,
        stress_level=stress,
        coach_comment=comment,
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    return AttemptResponse(
        id=attempt.id,
        score=attempt.score,
        success=attempt.success,
        predicted_mood=attempt.predicted_mood,
        stress_level=attempt.stress_level,
        coach_comment=attempt.coach_comment,
    )`;

const TRAIN_SNIPPET = `from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import EMNIST
from torchvision import transforms

from models.quality_cnn import QualityCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

transform_train = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(degrees=8, fill=0),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1),
        fill=0,
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

transform_eval = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

emnist_train = EmnistLettersDataset(
    root=DATA_ROOT,
    train=True,
    transform=transform_train,
)

shape_dataset = SyntheticShapesDataset(
    num_per_class=2000,
    img_size=28,
    transform=transform_train,
)

train_dataset = ConcatDataset([emnist_train, shape_dataset])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

model = QualityCNN(num_classes=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "models/quality_cnn.pth")`;

// Extra snippet used in the confusion-matrix / evaluation block
const EVAL_SNIPPET = `from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def evaluate_with_confusion_matrix(model, data_loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()`;

// NEW: 5 focused snippets from train_quality_model.py
const CONFIG_SNIPPET = `IMG_SIZE = 28  # 28x28 images
BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 1e-3

NUM_LETTERS = 26  # A..Z
SHAPE_LABELS = ["CIRCLE", "SQUARE", "TRIANGLE", "STAR"]
NUM_CLASSES = NUM_LETTERS + len(SHAPE_LABELS)

QUALITY_MODEL_PATH = MODELS_DIR / "quality_cnn.pth"`;

const LABEL_SNIPPET = `def build_label_mapping():
    label_to_index = {}
    for i in range(NUM_LETTERS):
        ch = chr(ord("A") + i)
        label_to_index[f"LETTER:{ch}"] = i

    for j, shape in enumerate(SHAPE_LABELS):
        label_to_index[f"SHAPE:{shape}"] = NUM_LETTERS + j

    return label_to_index`;

const MODEL_SNIPPET = `class QualityCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )`;

const DATASET_SNIPPET = `class SyntheticShapesDataset(Dataset):
    def __init__(self, num_per_class=2000, img_size=IMG_SIZE, transform=None):
        self.num_per_class = num_per_class
        self.img_size = img_size
        self.transform = transform
        self.shape_labels = SHAPE_LABELS

    def __getitem__(self, idx):
        class_idx = idx // self.num_per_class
        shape_name = self.shape_labels[class_idx]
        img = self._draw_shape_image(shape_name)
        if self.transform is not None:
            img = self.transform(img)
        label = NUM_LETTERS + class_idx
        return img, label`;

const TRAIN_LOOP_SNIPPET = `best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"=== Epoch {epoch}/{EPOCHS} ===")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = eval_epoch(model, test_loader, device)

    print(
        f"Train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}% | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), QUALITY_MODEL_PATH)
        print(f"[INFO] Saved best model to {QUALITY_MODEL_PATH}")`;

const systemBlocks = [
  {
    id: 'browser',
    title: 'Browser + MediaPipe',
    subtitle:
      'Tracks the child’s hand in real-time and creates the (x, y, t) trajectory.',
    snippet: `// Pseudo-code – MediaPipe hand tracking
const points: TrajectoryPoint[] = [];

onHandLandmarks((landmarks, t) => {
  const indexTip = landmarks[8]; // index finger tip
  const xNorm = indexTip.x;      // 0..1
  const yNorm = indexTip.y;      // 0..1
  points.push({ x: xNorm, y: yNorm, t });
});`,
  },
  {
    id: 'react',
    title: 'React Frontend',
    subtitle:
      'Collects drawing + webcam snapshots and sends one attempt to FastAPI.',
    snippet: `const payload: AttemptCreatePayload = {
  user_name: childName,
  user_age: age,
  item_type: mode,       // "LETTER" or "SHAPE"
  item_label: label,     // "A", "B", "circle", ...
  started_at: startedAt.toISOString(),
  ended_at: new Date().toISOString(),
  num_corrections: 0,
  num_undos: 0,
  face_mood: null,
  drawing_base64: null,
  face_snapshots,
  trajectory,
};

await createAttempt(payload);`,
  },
  {
    id: 'fastapi',
    title: 'FastAPI Backend',
    subtitle: 'Receives attempts, runs QualityCNN + FER and stores results in MySQL.',
    snippet: MAIN_SNIPPET,
  },
  {
    id: 'models',
    title: 'QualityCNN + FER',
    subtitle: 'CNN scores letters/shapes, FER estimates mood from face snapshots.',
    snippet: `# render trajectory to 28x28 image
img = render_trajectory_to_image(trajectory)
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = quality_cnn(x)
    probs = torch.softmax(logits, dim=1)
    best_idx = int(probs.argmax(dim=1))
    best_prob = float(probs[0, best_idx])

emotion = fer_detector.detect_emotions(face_image)
mood_label = decode_emotion(emotion)`,
  },
  {
    id: 'db',
    title: 'MySQL Database',
    subtitle: 'Stores users, practice items and attempts for long-term tracking.',
    snippet: `class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    practice_item_id = Column(Integer, ForeignKey("practice_items.id"), nullable=False)

    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=False)
    duration_seconds = Column(Float, nullable=False)

    score = Column(Float, nullable=False)
    success = Column(Boolean, default=False)
    predicted_mood = Column(String(30))
    stress_level = Column(Float, default=0.0)`,
  },
  {
    id: 'dashboard',
    title: 'Teacher Dashboard',
    subtitle: 'Aggregates attempts into per-student charts for score and stress.',
    snippet: `const [summary, setSummary] = useState<SummaryStats | null>(null);

useEffect(() => {
  getSummary(selectedStudentName).then(setSummary);
}, [selectedStudentName]);`,
  },
];

const ProjectOverview: React.FC = () => {
  const [modal, setModal] = useState<ModalState>({
    open: false,
    title: '',
    subtitle: '',
    code: '',
  });

  const openModal = (title: string, subtitle: string, code: string) => {
    setModal({ open: true, title, subtitle, code });
  };

  const closeModal = () => {
    setModal((prev) => ({ ...prev, open: false }));
  };

  return (
    <div>
      {/* Animations + decorations */}
      <style>{`
        @keyframes float {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-6px); }
          100% { transform: translateY(0px); }
        }
        @keyframes pulseSoft {
          0% { transform: scale(1); opacity: 0.8; }
          50% { transform: scale(1.05); opacity: 1; }
          100% { transform: scale(1); opacity: 0.8; }
        }
        @keyframes flowPulse {
          0%, 15% { box-shadow: 0 0 0 0 rgba(79,70,229,0); }
          20% { box-shadow: 0 0 0 3px rgba(79,70,229,0.55); }
          30% { box-shadow: 0 0 0 0 rgba(79,70,229,0); }
          100% { box-shadow: 0 0 0 0 rgba(79,70,229,0); }
        }
        @keyframes arrowPulse {
          0%, 20%, 100% { opacity: 0.25; transform: translateX(0); }
          40% { opacity: 1; transform: translateX(4px); }
          60% { opacity: 0.6; transform: translateX(2px); }
        }
      `}</style>

      {/* Value proposition / who is this for */}
      <div
        style={{
          borderRadius: 16,
          padding: 16,
          background:
            'linear-gradient(135deg,#fef3c7,#fffbeb 40%,#e0f2fe 100%)',
          border: '1px solid #fed7aa',
          marginBottom: 16,
        }}
      >
        <div
          style={{
            fontSize: 18,
            fontWeight: 700,
            marginBottom: 6,
          }}
        >
          Why Kid Drawing Coach? 🌍
        </div>
        <p
          style={{
            fontSize: 13,
            color: '#4b5563',
            marginBottom: 10,
            maxWidth: 640,
          }}
        >
          Many children struggle with handwriting, concentration and confidence.
          Kid Drawing Coach turns simple drawing exercises into measurable
          signals: quality of strokes, stress level and progress over time. It
          gives teachers and parents a clear picture of how the child is doing,
          not just if the letter is “right or wrong”.
        </p>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit,minmax(190px,1fr))',
            gap: 12,
          }}
        >
          <div
            style={{
              borderRadius: 12,
              padding: 10,
              background: '#fff7ed',
              border: '1px solid #fed7aa',
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
              👩‍🏫 For teachers & schools
            </div>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 16,
                margin: 0,
              }}
            >
              <li>Objective scores for letters and shapes.</li>
              <li>Stress trends to detect frustration early.</li>
              <li>Reports per student &amp; per class.</li>
            </ul>
          </div>

          <div
            style={{
              borderRadius: 12,
              padding: 10,
              background: '#ecfeff',
              border: '1px solid #a5f3fc',
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
              👨‍👩‍👧 For parents & therapists
            </div>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 16,
                margin: 0,
              }}
            >
              <li>Fun, gamified practice at home.</li>
              <li>Visual feedback about calm vs. stressful sessions.</li>
              <li>Share simple PDF summaries with specialists.</li>
            </ul>
          </div>

          <div
            style={{
              borderRadius: 12,
              padding: 10,
              background: '#eef2ff',
              border: '1px solid #c7d2fe',
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
              🧪 For researchers & EdTech labs
            </div>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 16,
                margin: 0,
              }}
            >
              <li>Fine-grained trajectory and mood data.</li>
              <li>Open CNN / FastAPI pipeline ready to extend.</li>
              <li>Ideal sandbox for explainable AI in education.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Hero / intro with kids decoration */}
      <div
        style={{
          borderRadius: 16,
          padding: 16,
          background:
            'radial-gradient(circle at top left,#bfdbfe,#eef2ff 35%,#f9fafb 70%)',
          marginBottom: 16,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Soft clouds */}
        <div
          style={{
            position: 'absolute',
            top: -20,
            right: 10,
            width: 70,
            height: 40,
            borderRadius: 999,
            background: '#ffffff88',
            filter: 'blur(1px)',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: -10,
            left: 30,
            width: 80,
            height: 40,
            borderRadius: 999,
            background: '#ffffff66',
            filter: 'blur(1px)',
          }}
        />

        <div
          style={{
            display: 'flex',
            gap: 16,
            alignItems: 'center',
            flexWrap: 'wrap',
            position: 'relative',
            zIndex: 1,
          }}
        >
          <div style={{ flex: 1, minWidth: 220 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                marginBottom: 8,
              }}
            >
              <div
                style={{
                  width: 46,
                  height: 46,
                  borderRadius: 999,
                  background:
                    'radial-gradient(circle at 25% 20%, #fb7185, #f97316)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 0 0 3px #fee2e2',
                  animation: 'pulseSoft 4s ease-in-out infinite',
                }}
              >
                <span style={{ fontSize: 26 }}>👦👧</span>
              </div>
              <div>
                <h2
                  style={{
                    fontSize: 20,
                    marginBottom: 2,
                    marginTop: 0,
                  }}
                >
                  What is Kid Drawing Coach? ✨
                </h2>
                <p
                  style={{
                    fontSize: 12,
                    color: '#4b5563',
                    margin: 0,
                  }}
                >
                  A playful system that mixes computer vision, deep learning and
                  pedagogy to help children practice writing and drawing.
                </p>
              </div>
            </div>

            <p
              style={{
                fontSize: 13,
                color: '#4b5563',
                marginBottom: 6,
                marginTop: 6,
              }}
            >
              Children draw letters and shapes in the air; the camera tracks
              their fingertip and face. The system then:
            </p>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 18,
                margin: 0,
              }}
            >
              <li>Converts air-drawing into a clean 28×28 image 🎯</li>
              <li>Uses a CNN to estimate how close it is to the target ✏️</li>
              <li>Uses a face model to estimate mood and stress 🙂</li>
              <li>Stores everything in a database for teachers 📚</li>
            </ul>
          </div>

          <div
            style={{
              flex: '0 0 230px',
              borderRadius: 18,
              background: '#ffffffdd',
              padding: 12,
              boxShadow: '0 8px 18px rgba(15,23,42,0.12)',
              animation: 'float 4s ease-in-out infinite',
            }}
          >
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                marginBottom: 6,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              🎓 Teacher view
            </div>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              • See average scores and stress levels per student.
            </p>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              • Explore history of attempts over time.
            </p>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              • Use charts to discuss progress with families.
            </p>
          </div>
        </div>
      </div>

      {/* Deep learning sections + buttons to open code modals */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1.3fr 1.3fr',
          gap: 16,
          marginBottom: 16,
        }}
      >
        <div
          style={{
            borderRadius: 14,
            padding: 12,
            border: '1px solid #e5e7eb',
            background: '#f9fafb',
          }}
        >
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              marginBottom: 4,
            }}
          >
            Deep Learning for Letter &amp; Shape Quality 🧠
          </div>
          <p
            style={{
              fontSize: 13,
              color: '#4b5563',
              marginBottom: 6,
            }}
          >
            A custom convolutional neural network <code>QualityCNN</code> is
            trained on:
          </p>
          <ul
            style={{
              fontSize: 12,
              color: '#4b5563',
              margin: 0,
              paddingLeft: 18,
            }}
          >
            <li>EMNIST handwritten letters (A–Z)</li>
            <li>Extra synthetic shapes: circle, square, triangle, star</li>
          </ul>
          <p
            style={{
              fontSize: 12,
              color: '#4b5563',
              marginTop: 6,
            }}
          >
            Each air-drawn stroke is converted to a small grayscale image,
            centered and scaled, then passed to <code>QualityCNN</code>, which
            outputs probabilities for each class.
          </p>

          <pre
            style={{
              fontSize: 11,
              background: '#111827',
              color: '#e5e7eb',
              padding: 8,
              borderRadius: 8,
              marginTop: 8,
              overflowX: 'auto',
            }}
          >
            {`class QualityCNN(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )`}
          </pre>

          <div
            style={{
              display: 'flex',
              gap: 8,
              marginTop: 8,
              flexWrap: 'wrap',
            }}
          >
            <button
              type="button"
              onClick={() =>
                openModal(
                  'Backend FastAPI – main.py',
                  'Core endpoint that scores a drawing and stores an attempt.',
                  MAIN_SNIPPET,
                )
              }
              style={{
                padding: '5px 10px',
                fontSize: 11,
                borderRadius: 999,
                border: 'none',
                background: '#4f46e5',
                color: '#ffffff',
                cursor: 'pointer',
              }}
            >
              View backend snippet (main.py)
            </button>
            <button
              type="button"
              onClick={() =>
                openModal(
                  'QualityCNN Training Script',
                  'How we train the CNN on EMNIST + synthetic shapes with augmentation.',
                  TRAIN_SNIPPET,
                )
              }
              style={{
                padding: '5px 10px',
                fontSize: 11,
                borderRadius: 999,
                border: '1px solid #4f46e5',
                background: '#eef2ff',
                color: '#4f46e5',
                cursor: 'pointer',
              }}
            >
              View training script (train_quality_model.py)
            </button>
          </div>
        </div>

        <div
          style={{
            borderRadius: 14,
            padding: 12,
            border: '1px solid #e5e7eb',
            background: '#f9fafb',
          }}
        >
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              marginBottom: 4,
            }}
          >
            Deep Learning for Face Emotion (Mood) 🙂
          </div>
          <p
            style={{
              fontSize: 13,
              color: '#4b5563',
              marginBottom: 6,
            }}
          >
            We use the <code>fer</code> library, which wraps a pretrained facial
            expression recognition model. While the child draws, we capture a
            few webcam frames and feed them to FER.
          </p>
          <ul
            style={{
              fontSize: 12,
              color: '#4b5563',
              margin: 0,
              paddingLeft: 18,
            }}
          >
            <li>Each frame is analyzed for emotions (happy, neutral, sad...)</li>
            <li>Probabilities are averaged across time.</li>
            <li>The strongest emotion becomes the final mood label.</li>
          </ul>
          <p
            style={{
              fontSize: 12,
              color: '#4b5563',
              marginTop: 6,
            }}
          >
            The mood is converted to a stress level and combined with hand
            shakiness to compute the final stress score used in the dashboard.
          </p>

          <pre
            style={{
              fontSize: 11,
              background: '#111827',
              color: '#e5e7eb',
              padding: 8,
              borderRadius: 8,
              marginTop: 8,
              overflowX: 'auto',
            }}
          >
            {`mood = predict_mood_from_snapshots(face_snapshots)
stress_from_face = simple_emotion_to_stress(mood)
stress_level = min(max(stress_from_face + shakiness * 0.5, 0.0), 1.0)`}
          </pre>
        </div>
      </div>

      {/* Rectangle architecture diagram with animated data flow */}
      <div
        style={{
          borderRadius: 14,
          padding: 12,
          border: '1px solid #e5e7eb',
          background: '#ffffff',
          marginBottom: 16,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 6,
          }}
        >
          Global architecture – animated data flow 📦 ➜ 📦
        </div>
        <p
          style={{
            fontSize: 13,
            color: '#4b5563',
            marginBottom: 10,
          }}
        >
          The rectangles below represent the main components of the system. A
          soft glow moves from left to right to show how data flows through the
          pipeline. Click any block to see a short explanation and a code
          snippet.
        </p>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 10,
            justifyContent: 'center',
            padding: 6,
          }}
        >
          {systemBlocks.map((block, index) => {
            const isEven = index % 2 === 0;
            const cardBg = isEven
              ? 'linear-gradient(135deg,#4f46e5,#6366f1)'
              : 'linear-gradient(135deg,#ec4899,#f97316)';

            return (
              <React.Fragment key={block.id}>
                {/* Rectangle block */}
                <div
                  onClick={() =>
                    openModal(block.title, block.subtitle, block.snippet)
                  }
                  style={{
                    position: 'relative',
                    width: 180,
                    minHeight: 90,
                    borderRadius: 14,
                    padding: 10,
                    background: '#0f172a',
                    border: '1px solid #1f2937',
                    cursor: 'pointer',
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    animation: `flowPulse 6s linear infinite`,
                    animationDelay: `${index * 0.6}s`,
                  }}
                >
                  {/* Animated gradient bar at the top to suggest data entering */}
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      height: 4,
                      background: cardBg,
                      opacity: 0.9,
                    }}
                  />
                  {/* Soft internal gradient */}
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      background:
                        'radial-gradient(circle at top left, rgba(148,163,184,0.35), transparent 55%)',
                      pointerEvents: 'none',
                    }}
                  />
                  <div
                    style={{
                      position: 'relative',
                      zIndex: 1,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: '#f9fafb',
                        marginBottom: 4,
                      }}
                    >
                      {block.title}
                    </div>
                    <p
                      style={{
                        fontSize: 11,
                        color: '#cbd5f5',
                        margin: 0,
                      }}
                    >
                      {block.subtitle}
                    </p>
                  </div>

                  {/* Animated data dot inside block */}
                  <div
                    style={{
                      position: 'absolute',
                      bottom: 6,
                      left: 8,
                      width: 8,
                      height: 8,
                      borderRadius: 999,
                      background: isEven ? '#a5b4fc' : '#fed7e2',
                      boxShadow: '0 0 6px rgba(191,219,254,0.8)',
                      animation: 'float 3.2s ease-in-out infinite',
                      animationDelay: `${index * 0.6}s`,
                    }}
                  />
                </div>

                {/* Arrow between rectangles (not after last) */}
                {index < systemBlocks.length - 1 && (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      minWidth: 28,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 20,
                        color: '#9ca3af',
                        animation: 'arrowPulse 6s linear infinite',
                        animationDelay: `${index * 0.6 + 0.3}s`,
                      }}
                    >
                      ➜
                    </span>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Example small React snippet */}
      <div
        style={{
          borderRadius: 14,
          padding: 12,
          border: '1px solid #e5e7eb',
          background: '#f9fafb',
          marginBottom: 16,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 4,
          }}
        >
          Example: sending an attempt from React to FastAPI 💻 ➜ 🐍
        </div>
        <pre
          style={{
            fontSize: 11,
            background: '#111827',
            color: '#e5e7eb',
            padding: 8,
            borderRadius: 8,
            marginTop: 4,
            overflowX: 'auto',
          }}
        >
          {`const payload = {
  user_name: childName,
  user_age: age,
  item_type: mode,      // "LETTER" or "SHAPE"
  item_label: label,    // "A", "B", "circle", ...
  started_at: startedAt.toISOString(),
  ended_at: new Date().toISOString(),
  num_corrections: 0,
  num_undos: 0,
  face_mood: null,
  drawing_base64: null,
  face_snapshots,
  trajectory,           // [{x,y,t}, ...]
};

const res = await fetch('http://localhost:8000/api/attempts', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
const result = await res.json();`}
        </pre>
        <p
          style={{
            fontSize: 12,
            color: '#4b5563',
            marginTop: 6,
          }}
        >
          This result object is then used to update the &quot;Result&quot; card
          and the teacher dashboards.
        </p>
      </div>

      {/* Confusion-matrix / evaluation block */}
      <div
        style={{
          borderRadius: 14,
          padding: 12,
          border: '1px solid #e5e7eb',
          background: '#ffffff',
          marginBottom: 18,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 6,
          }}
        >
          How do we check if the CNN really understands children&apos;s letters? 📊
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,1.1fr)',
            gap: 16,
            alignItems: 'flex-start',
          }}
        >
          {/* Big-line explanation */}
          <div>
            <p
              style={{
                fontSize: 13,
                color: '#4b5563',
                marginBottom: 6,
              }}
            >
              After training, we don&apos;t just look at the accuracy number. We
              also build a <strong>confusion matrix</strong> to see where the
              model hesitates, for example when an <strong>S</strong> is
              confused with a <strong>U</strong>.
            </p>
            <ol
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 18,
                marginTop: 4,
                marginBottom: 4,
              }}
            >
              <li>
                Run the trained <code>QualityCNN</code> on all test images
                (letters + shapes) and store predicted vs. true labels.
              </li>
              <li>
                Use <code>confusion_matrix</code> and{' '}
                <code>classification_report</code> to see which pairs are
                confused (S vs U, C vs O, etc.).
              </li>
              <li>
                Visualize the matrix as a heatmap – bright cells on the diagonal
                mean good predictions, bright off-diagonal cells show mistakes.
              </li>
              <li>
                These insights guide us to tune data augmentation and
                preprocessing (for example, flipping or rotation) until the
                matrix becomes cleaner.
              </li>
              <li>
                We can even export this plot for the jury to show before/after
                improvements of the model.
              </li>
            </ol>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginTop: 6,
              }}
            >
              This is exactly how we debug the case where a drawing that looks
              like an <strong>S</strong> to humans is predicted as{' '}
              <strong>U</strong>: the confusion matrix exposes the problem and
              justifies why we changed the training pipeline.
            </p>
          </div>

          {/* Code snippet */}
          <div>
            <pre
              style={{
                fontSize: 11,
                background: '#111827',
                color: '#e5e7eb',
                padding: 8,
                borderRadius: 8,
                marginTop: 4,
                overflowX: 'auto',
              }}
            >
              {EVAL_SNIPPET}
            </pre>
            <p
              style={{
                fontSize: 11,
                color: '#9ca3af',
                marginTop: 6,
              }}
            >
              This helper can be called after training with the test loader and
              a list of label names to quickly generate a confusion matrix and a
              textual report of precision/recall per class.
            </p>
          </div>
        </div>
      </div>

      {/* NEW: 5 code snippets from train_quality_model.py with explanations */}
      <div
        style={{
          borderRadius: 14,
          padding: 12,
          border: '1px solid #e5e7eb',
          background: '#f9fafb',
          marginBottom: 18,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 6,
          }}
        >
          Training script – 5 key blocks (code + big lines) 🧩
        </div>

        {/* 1. Config */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,1.1fr)',
            gap: 16,
            marginTop: 10,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                marginBottom: 4,
              }}
            >
              1️⃣ Global configuration
            </div>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              This block defines all global hyper-parameters of the training:
            </p>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 18,
                margin: 0,
              }}
            >
              <li>
                <code>IMG_SIZE</code>, <code>BATCH_SIZE</code>,{' '}
                <code>EPOCHS</code>, <code>LEARNING_RATE</code>.
              </li>
              <li>
                Number of letters and shape labels used to compute{' '}
                <code>NUM_CLASSES</code>.
              </li>
              <li>
                <code>QUALITY_MODEL_PATH</code> defines where the best model is
                saved.
              </li>
              <li>
                In one slide, you can say: &quot;Changing these values lets us
                experiment with faster training vs. better accuracy.&quot;
              </li>
            </ul>
          </div>
          <div>
            <pre
              style={{
                fontSize: 11,
                background: '#111827',
                color: '#e5e7eb',
                padding: 8,
                borderRadius: 8,
                overflowX: 'auto',
              }}
            >
              {CONFIG_SNIPPET}
            </pre>
          </div>
        </div>

        {/* 2. Label mapping */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,1.1fr)',
            gap: 16,
            marginTop: 16,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                marginBottom: 4,
              }}
            >
              2️⃣ Label mapping (letters &amp; shapes)
            </div>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              We map human-friendly labels to numeric indices for the CNN:
            </p>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 18,
                margin: 0,
              }}
            >
              <li>
                <code>LETTER:A..Z</code> are mapped to <code>0..25</code>.
              </li>
              <li>
                <code>SHAPE:CIRCLE/SQUARE/TRIANGLE/STAR</code> become{' '}
                <code>26..29</code>.
              </li>
              <li>
                This mapping is saved to JSON so the backend + dashboard can
                decode predictions back to real labels.
              </li>
              <li>
                In the jury explanation: &quot;The model only sees numbers, but
                we keep a mapping to explain results in natural language.&quot;
              </li>
            </ul>
          </div>
          <div>
            <pre
              style={{
                fontSize: 11,
                background: '#111827',
                color: '#e5e7eb',
                padding: 8,
                borderRadius: 8,
                overflowX: 'auto',
              }}
            >
              {LABEL_SNIPPET}
            </pre>
          </div>
        </div>

        {/* 3. CNN model */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,1.1fr)',
            gap: 16,
            marginTop: 16,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                marginBottom: 4,
              }}
            >
              3️⃣ QualityCNN – the brain of letter recognition
            </div>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              This class defines the convolutional neural network:
            </p>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 18,
                margin: 0,
              }}
            >
              <li>
                Two convolutional layers with pooling shrink the 28×28 image to
                7×7 while extracting edges and shapes.
              </li>
              <li>
                The classifier (fully-connected layers) turns these features
                into class probabilities.
              </li>
              <li>
                <code>Dropout</code> helps avoid overfitting on a single
                writing style.
              </li>
              <li>
                In the slide: &quot;This is the AI that judges how close the
                child&apos;s drawing is to the target letter or shape.&quot;
              </li>
            </ul>
          </div>
          <div>
            <pre
              style={{
                fontSize: 11,
                background: '#111827',
                color: '#e5e7eb',
                padding: 8,
                borderRadius: 8,
                overflowX: 'auto',
              }}
            >
              {MODEL_SNIPPET}
            </pre>
          </div>
        </div>

        {/* 4. SyntheticShapesDataset */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,1.1fr)',
            gap: 16,
            marginTop: 16,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                marginBottom: 4,
              }}
            >
              4️⃣ SyntheticShapesDataset – teaching the model basic geometry
            </div>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              EMNIST has letters, but no shapes. We generate them ourselves:
            </p>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 18,
                margin: 0,
              }}
            >
              <li>
                For each index, we choose a shape (circle, square, triangle,
                star).
              </li>
              <li>
                We draw it with Pillow in a 28×28 grayscale image and optionally
                apply transforms.
              </li>
              <li>
                Its label is placed after letters: <code>26..29</code>.
              </li>
              <li>
                This allows the coach to also train “geometric drawing”
                exercises in the same model as letters.
              </li>
            </ul>
          </div>
          <div>
            <pre
              style={{
                fontSize: 11,
                background: '#111827',
                color: '#e5e7eb',
                padding: 8,
                borderRadius: 8,
                overflowX: 'auto',
              }}
            >
              {DATASET_SNIPPET}
            </pre>
          </div>
        </div>

        {/* 5. Training loop */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,1.1fr)',
            gap: 16,
            marginTop: 16,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                marginBottom: 4,
              }}
            >
              5️⃣ Training loop – how the model learns
            </div>
            <p
              style={{
                fontSize: 12,
                color: '#4b5563',
                marginBottom: 4,
              }}
            >
              The loop coordinates the full learning process:
            </p>
            <ul
              style={{
                fontSize: 12,
                color: '#4b5563',
                paddingLeft: 18,
                margin: 0,
              }}
            >
              <li>
                For each epoch, we call <code>train_epoch</code> then{' '}
                <code>eval_epoch</code> on the test set.
              </li>
              <li>
                We print training and validation accuracy to track progress over
                time.
              </li>
              <li>
                Whenever the validation accuracy improves, we save a new{' '}
                <code>quality_cnn.pth</code>.
              </li>
              <li>
                In one sentence: &quot;This loop is the teacher of the AI –
                showing examples, correcting mistakes, and keeping the best
                student copy.&quot;
              </li>
            </ul>
          </div>
          <div>
            <pre
              style={{
                fontSize: 11,
                background: '#111827',
                color: '#e5e7eb',
                padding: 8,
                borderRadius: 8,
                overflowX: 'auto',
              }}
            >
              {TRAIN_LOOP_SNIPPET}
            </pre>
          </div>
        </div>
      </div>

      {/* Modal for code snippets */}
      {modal.open && (
        <div
          onClick={closeModal}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15,23,42,0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 50,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 'min(840px, 92%)',
              maxHeight: '80vh',
              background: '#0f172a',
              borderRadius: 14,
              padding: 14,
              boxShadow: '0 20px 40px rgba(0,0,0,0.45)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 8,
                alignItems: 'center',
                marginBottom: 10,
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 600,
                    color: '#e5e7eb',
                    marginBottom: 4,
                  }}
                >
                  {modal.title}
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: '#9ca3af',
                  }}
                >
                  {modal.subtitle}
                </div>
              </div>
              <button
                type="button"
                onClick={closeModal}
                style={{
                  border: 'none',
                  borderRadius: 999,
                  padding: '3px 9px',
                  background: '#ef4444',
                  color: '#f9fafb',
                  fontSize: 12,
                  cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>
            <div
              style={{
                flex: 1,
                overflow: 'auto',
                borderRadius: 10,
                background: '#020617',
                padding: 8,
                border: '1px solid #1f2937',
              }}
            >
              <pre
                style={{
                  margin: 0,
                  fontFamily:
                    'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
                  fontSize: 12,
                  color: '#e5e7eb',
                  whiteSpace: 'pre',
                }}
              >
                {modal.code}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProjectOverview;
