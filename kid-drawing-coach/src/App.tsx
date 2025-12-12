// src/App.tsx
import React, { useEffect, useState } from 'react';
import FingerDrawingCanvas from './components/FingerDrawingCanvas';
import WebcamMoodTracker from './components/WebcamMoodTracker';
import StudentsDashboard from './components/StudentDashboard';
import ProjectOverview from './components/ProjectOverview';

type TrajectoryPoint = { x: number; y: number; t: number };

interface AttemptResponse {
  id: number;
  score: number;
  success: boolean;
  predicted_mood: string | null;
  stress_level: number;
  coach_comment: string | null;
}

type StudentGamification = {
  level: number;
  stars: number;
  badges: string[];
  today_attempts: number;
  today_goal: number;
};

type StudentInsightsResponse = {
  user_name: string;
  avg_score: number;
  avg_stress: number;
  success_rate: number;
  gamification: StudentGamification;
};

type ExplainResponse = {
  saliency_image: string;
  predicted_label: string | null;
  predicted_confidence: number;
  match_confidence: number | null;
};

const API_BASE = 'http://localhost:8000';
const TODAY_GOAL = 5;

type TabKey = 'coach' | 'students' | 'project';

type ThemeKey = 'space' | 'underwater' | 'jungle';

type ThemeConfig = {
  label: string;
  icon: string;
  pageBackground: string;
  headerOrbGradient: string;
  accent: string;
  accentSoft: string;
  accentText: string;
};

const THEMES: Record<ThemeKey, ThemeConfig> = {
  space: {
    label: 'Space',
    icon: '🚀',
    pageBackground: 'linear-gradient(180deg,#020617,#0f172a,#020617)',
    headerOrbGradient:
      'radial-gradient(circle at 25% 20%, #f97316, #4f46e5)',
    accent: '#4f46e5',
    accentSoft: '#e0e7ff',
    accentText: '#312e81',
  },
  underwater: {
    label: 'Underwater',
    icon: '🐠',
    pageBackground: 'linear-gradient(180deg,#0ea5e9,#e0f2fe)',
    headerOrbGradient:
      'radial-gradient(circle at 25% 20%, #22d3ee, #0ea5e9)',
    accent: '#0ea5e9',
    accentSoft: '#e0f2fe',
    accentText: '#0f172a',
  },
  jungle: {
    label: 'Jungle',
    icon: '🦁',
    pageBackground: 'linear-gradient(180deg,#064e3b,#bbf7d0)',
    headerOrbGradient:
      'radial-gradient(circle at 25% 20%, #22c55e, #16a34a)',
    accent: '#16a34a',
    accentSoft: '#dcfce7',
    accentText: '#064e3b',
  },
};

// Simple TTS helper for audio coaching messages
function speakTip(text: string) {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
  const synth = window.speechSynthesis;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.pitch = 1.0;
  synth.cancel();
  synth.speak(utter);
}

// Frontend shakiness computation (same idea as backend)
function computeShakiness(points: TrajectoryPoint[]): number {
  if (points.length < 3) return 0;

  const angles: number[] = [];
  for (let i = 1; i < points.length - 1; i++) {
    const p0 = points[i - 1];
    const p1 = points[i];
    const p2 = points[i + 1];

    const x1 = p1.x - p0.x;
    const y1 = p1.y - p0.y;
    const x2 = p2.x - p1.x;
    const y2 = p2.y - p1.y;

    if ((x1 === 0 && y1 === 0) || (x2 === 0 && y2 === 0)) continue;

    const dot = x1 * x2 + y1 * y2;
    const mag1 = Math.hypot(x1, y1);
    const mag2 = Math.hypot(x2, y2);
    if (mag1 === 0 || mag2 === 0) continue;

    const cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
    const angle = Math.acos(cos);
    angles.push(angle);
  }

  if (!angles.length) return 0;

  const mean =
    angles.reduce((acc, a) => acc + a, 0) / angles.length;
  const variance =
    angles.reduce((acc, a) => acc + (a - mean) ** 2, 0) /
    angles.length;
  const shakiness = Math.sqrt(variance);

  return Math.min(shakiness / Math.PI, 1);
}

const App: React.FC = () => {
  const [userName, setUserName] = useState('Amine');
  const [userAge, setUserAge] = useState<number | ''>(8);
  const [itemType, setItemType] = useState<'LETTER' | 'SHAPE'>('LETTER');
  const [itemLabel, setItemLabel] = useState('A');

  const [startedAt, setStartedAt] = useState<number>(Date.now());
  const [resetKey, setResetKey] = useState<number>(0);

  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);
  const [faceSnapshots, setFaceSnapshots] = useState<string[]>([]);
  const [phase, setPhase] = useState<'idle' | 'drawing' | 'done'>('idle');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AttemptResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<TabKey>('coach');

  // Gamification / insights
  const [avgScore, setAvgScore] = useState<number | null>(null);
  const [level, setLevel] = useState<number>(1);
  const [stars, setStars] = useState<number>(0);
  const [badges, setBadges] = useState<string[]>([]);
  const [attemptsToday, setAttemptsToday] = useState<number>(0);

  // Live coaching
  const [liveShakiness, setLiveShakiness] = useState<number>(0);
  const [lastTip, setLastTip] = useState<string | null>(null);
  const [lastTipTime, setLastTipTime] = useState<number | null>(null);

  // Explainable AI – saliency map
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);

  // Kid-friendly theme
  const [theme, setTheme] = useState<ThemeKey>('space');
  const currentTheme = THEMES[theme];

  const progressPct = Math.max(
    0,
    Math.min(100, (attemptsToday / TODAY_GOAL) * 100),
  );

  // Load insights for a given user
  async function refreshInsights(name: string) {
    try {
      const res = await fetch(
        `${API_BASE}/api/insights/student?user_name=${encodeURIComponent(
          name,
        )}`,
      );
      if (!res.ok) {
        return; // optional: insights are not critical
      }
      const data: StudentInsightsResponse = await res.json();
      setAvgScore(data.avg_score);
      setLevel(data.gamification.level);
      setStars(data.gamification.stars);
      setBadges(data.gamification.badges);
      setAttemptsToday(data.gamification.today_attempts);
    } catch (e) {
      console.error('Failed to load insights', e);
    }
  }

  // Reset gamification when user changes so levels don't mix between kids
  useEffect(() => {
    setAvgScore(null);
    setLevel(1);
    setStars(0);
    setBadges([]);
    setAttemptsToday(0);
    refreshInsights(userName);
  }, [userName]);

  const handleStart = () => {
    setStartedAt(Date.now());
    setResetKey((k) => k + 1);
    setPhase('drawing');
    setTrajectory([]);
    setFaceSnapshots([]);
    setResult(null);
    setError(null);
    setExplain(null);
    setExplainError(null);
    setLiveShakiness(0);

    const msg =
      "Let's draw together! Take your time and enjoy your " +
      (theme === 'space'
        ? 'space mission.'
        : theme === 'underwater'
        ? 'underwater adventure.'
        : 'jungle walk.');
    setLastTip(msg);
    setLastTipTime(Date.now());
    speakTip(msg);
  };

  const handleFinish = async () => {
    if (phase !== 'drawing') return;
    setPhase('done');
    setLoading(true);
    setError(null);

    try {
      const payload = {
        user_name: userName,
        user_age: userAge === '' ? null : Number(userAge),
        item_type: itemType,
        item_label: itemLabel,
        started_at: new Date(startedAt).toISOString(),
        ended_at: new Date().toISOString(),
        num_corrections: 0,
        num_undos: 0,
        face_mood: null, // FER decides based on snapshots
        drawing_base64: null,
        face_snapshots: faceSnapshots,
        trajectory,
      };

      const res = await fetch(`${API_BASE}/api/attempts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Request failed');
      }

      const data: AttemptResponse = await res.json();
      setResult(data);

      // Audio tip based on stress and score
      if (data.stress_level >= 0.65) {
        const msg =
          'You worked hard on this one. Take a little break and shake your hand.';
        setLastTip(msg);
        setLastTipTime(Date.now());
        speakTip(msg);
      } else if (data.score >= 70) {
        const msg = 'Great job! Your drawing is really good!';
        setLastTip(msg);
        setLastTipTime(Date.now());
        speakTip(msg);
      } else {
        const msg =
          'Nice try! Let’s practice again and make the lines smoother.';
        setLastTip(msg);
        setLastTipTime(Date.now());
        speakTip(msg);
      }

      // Refresh insights/gamification from backend (level, stars, attempts)
      refreshInsights(userName);
    } catch (e: any) {
      console.error(e);
      setError(e.message || 'Error while sending attempt');
    } finally {
      setLoading(false);
    }
  };

  // Handle trajectory updates from the canvas (no change to FingerDrawingCanvas)
  const handleTrajectoryChange = (points: TrajectoryPoint[]) => {
    setTrajectory(points);
    if (!points.length) {
      setLiveShakiness(0);
      return;
    }

    const shakiness = computeShakiness(points);
    setLiveShakiness(shakiness);

    const now = Date.now();
    if (
      phase === 'drawing' &&
      shakiness > 0.6 &&
      (!lastTipTime || now - lastTipTime > 8000)
    ) {
      const msg =
        'Your hand is moving fast, try slowing down a little bit.';
      setLastTip(msg);
      setLastTipTime(now);
      speakTip(msg);
    }
  };

  // Explainable AI: ask backend for saliency heatmap for current trajectory
  const handleExplain = async () => {
    if (!trajectory.length) return;
    setLoadingExplain(true);
    setExplainError(null);
    try {
      const payload = {
        item_type: itemType,
        item_label: itemLabel,
        trajectory,
      };
      const res = await fetch(`${API_BASE}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Explain API error');
      }
      const data: ExplainResponse = await res.json();
      setExplain(data);
    } catch (e: any) {
      console.error(e);
      setExplainError(
        e.message || 'Error while explaining this drawing',
      );
    } finally {
      setLoadingExplain(false);
    }
  };

  const renderCoachTab = () => (
    <>
      {/* Gamification strip */}
      <div
        style={{
          marginBottom: 20,
          borderRadius: 14,
          padding: 12,
          background: currentTheme.accentSoft,
          border: `1px solid ${currentTheme.accent}`,
          display: 'grid',
          gridTemplateColumns: '2fr 2fr 2fr',
          gap: 12,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 13,
              fontWeight: 600,
              marginBottom: 4,
              color: currentTheme.accentText,
            }}
          >
            Level & Stars
          </div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <span
              style={{
                fontSize: 18,
                fontWeight: 700,
                color: currentTheme.accent,
              }}
            >
              Lv. {level}
            </span>
            <span style={{ fontSize: 18 }}>
              {'★'.repeat(stars)}
              {'☆'.repeat(Math.max(0, 4 - stars))}
            </span>
          </div>
          <div
            style={{
              fontSize: 11,
              color: currentTheme.accentText,
              marginTop: 2,
            }}
          >
            {avgScore !== null
              ? `Average score: ${avgScore.toFixed(1)} / 100`
              : 'Draw a few times to unlock your level.'}
          </div>
        </div>

        <div>
          <div
            style={{
              fontSize: 13,
              fontWeight: 600,
              marginBottom: 4,
              color: currentTheme.accentText,
            }}
          >
            Today&apos;s goal
          </div>
          <div
            style={{
              fontSize: 12,
              color: currentTheme.accentText,
              marginBottom: 4,
            }}
          >
            Attempts today: {attemptsToday} / {TODAY_GOAL}
          </div>
          <div
            style={{
              position: 'relative',
              width: '100%',
              height: 10,
              borderRadius: 999,
              background: '#e5e7eb',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: 0,
                width: `${progressPct}%`,
                borderRadius: 999,
                background:
                  progressPct >= 100 ? '#22c55e' : currentTheme.accent,
                transition: 'width 0.3s ease-out',
              }}
            />
          </div>
        </div>

        <div>
          <div
            style={{
              fontSize: 13,
              fontWeight: 600,
              marginBottom: 4,
              color: currentTheme.accentText,
            }}
          >
            Badges
          </div>
          {badges.length === 0 ? (
            <div
              style={{
                fontSize: 11,
                color: currentTheme.accentText,
              }}
            >
              Practice more to earn badges like{' '}
              <strong>Calm Artist</strong> or{' '}
              <strong>Smooth Lines</strong>.
            </div>
          ) : (
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 6,
              }}
            >
              {badges.map((b) => (
                <span
                  key={b}
                  style={{
                    fontSize: 11,
                    padding: '3px 8px',
                    borderRadius: 999,
                    background: '#fff',
                    border: '1px solid #e5e7eb',
                  }}
                >
                  {b}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      <p
        style={{
          fontSize: 14,
          color: '#4b5563',
          marginBottom: 16,
        }}
      >
        Draw letters or shapes with your fingertip in the air while the
        camera tracks your mood and the quality of your strokes.
      </p>

      {/* Form */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 16,
          marginBottom: 20,
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <label style={{ fontSize: 13 }}>Child name</label>
          <input
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            style={{
              borderRadius: 8,
              border: '1px solid #d1d5db',
              padding: '6px 10px',
            }}
          />
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <label style={{ fontSize: 13 }}>Age</label>
          <input
            type="number"
            value={userAge}
            onChange={(e) =>
              setUserAge(
                e.target.value === '' ? '' : Number(e.target.value),
              )
            }
            style={{
              borderRadius: 8,
              border: '1px solid #d1d5db',
              padding: '6px 10px',
            }}
          />
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <label style={{ fontSize: 13 }}>Mode</label>
          <select
            value={itemType}
            onChange={(e) =>
              setItemType(e.target.value as 'LETTER' | 'SHAPE')
            }
            style={{
              borderRadius: 8,
              border: '1px solid #d1d5db',
              padding: '6px 10px',
            }}
          >
            <option value="LETTER">Letter</option>
            <option value="SHAPE">Shape</option>
          </select>
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <label style={{ fontSize: 13 }}>
            {itemType === 'LETTER'
              ? 'Letter (A–Z)'
              : 'Shape label (circle, star...)'}
          </label>
          <input
            value={itemLabel}
            onChange={(e) => setItemLabel(e.target.value)}
            style={{
              borderRadius: 8,
              border: '1px solid #d1d5db',
              padding: '6px 10px',
            }}
          />
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <button
          type="button"
          onClick={handleStart}
          style={{
            padding: '8px 16px',
            borderRadius: 9999,
            border: 'none',
            background: currentTheme.accent,
            color: '#ffffff',
            fontWeight: 500,
            cursor: 'pointer',
            fontSize: 14,
          }}
        >
          Start drawing
        </button>
        <button
          type="button"
          onClick={handleFinish}
          disabled={phase !== 'drawing' || loading}
          style={{
            padding: '8px 16px',
            borderRadius: 9999,
            border: 'none',
            background:
              phase === 'drawing' && !loading
                ? '#22c55e'
                : '#9ca3af',
            color: '#ffffff',
            fontWeight: 500,
            cursor:
              phase === 'drawing' && !loading
                ? 'pointer'
                : 'not-allowed',
            fontSize: 14,
          }}
        >
          {loading ? 'Sending...' : 'Finish & analyze'}
        </button>
        <button
          type="button"
          onClick={handleExplain}
          disabled={trajectory.length === 0 || loadingExplain}
          style={{
            padding: '8px 16px',
            borderRadius: 9999,
            border: '1px solid #4b5563',
            background: '#ffffff',
            color: '#111827',
            fontWeight: 500,
            cursor:
              trajectory.length === 0 || loadingExplain
                ? 'not-allowed'
                : 'pointer',
            fontSize: 13,
          }}
        >
          {loadingExplain
            ? 'Explaining...'
            : 'Explain this drawing (AI)'}
        </button>
      </div>

      {/* Live coaching banner */}
      {(lastTip || liveShakiness > 0.6) && (
        <div
          style={{
            marginBottom: 16,
            borderRadius: 9999,
            padding: '6px 12px',
            background: currentTheme.accentSoft,
            border: `1px solid ${currentTheme.accent}`,
            fontSize: 12,
            color: currentTheme.accentText,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span role="img" aria-label="coach">
            🎧
          </span>
          <span>
            {lastTip ||
              'Your hand is moving fast, try slowing down a bit.'}
          </span>
        </div>
      )}

      {/* Main area: drawing + webcam */}
      <div
        style={{
          display: 'flex',
          gap: 24,
          alignItems: 'flex-start',
        }}
      >
        <FingerDrawingCanvas
          startedAt={startedAt}
          resetKey={resetKey}
          onTrajectoryChange={handleTrajectoryChange}
        />

        <WebcamMoodTracker
          active={phase === 'drawing'}
          onSnapshotsChange={setFaceSnapshots}
        />
      </div>

      {/* Result */}
      <div style={{ marginTop: 24 }}>
        {error && (
          <div
            style={{
              padding: 12,
              borderRadius: 8,
              background: '#fee2e2',
              color: '#b91c1c',
              fontSize: 14,
              marginBottom: 8,
            }}
          >
            {error}
          </div>
        )}

        {result && (
          <div
            style={{
              marginTop: 8,
              padding: 16,
              borderRadius: 12,
              background: '#f9fafb',
              border: '1px solid #e5e7eb',
            }}
          >
            <h2
              style={{
                fontSize: 18,
                marginBottom: 8,
              }}
            >
              Result
            </h2>
            <p style={{ fontSize: 14, marginBottom: 4 }}>
              <strong>Score:</strong> {result.score.toFixed(1)} / 100
            </p>
            <p style={{ fontSize: 14, marginBottom: 4 }}>
              <strong>Success:</strong>{' '}
              {result.success ? 'Yes 🎉' : 'Not yet'}
            </p>
            <p style={{ fontSize: 14, marginBottom: 4 }}>
              <strong>Predicted mood:</strong>{' '}
              {result.predicted_mood ?? 'Unknown'}
            </p>
            <p style={{ fontSize: 14, marginBottom: 4 }}>
              <strong>Stress level:</strong>{' '}
              {(result.stress_level * 100).toFixed(0)} %
            </p>
            <p style={{ fontSize: 14, marginTop: 8 }}>
              <strong>Coach comment:</strong> {result.coach_comment}
            </p>
          </div>
        )}

        {/* Explainable AI panel */}
        {(explainError || explain) && (
          <div
            style={{
              marginTop: 16,
              padding: 16,
              borderRadius: 12,
              background: '#fefce8',
              border: '1px solid #facc15',
            }}
          >
            <h3
              style={{
                fontSize: 16,
                marginBottom: 6,
              }}
            >
              Explainable AI – where did the model look? 🔍
            </h3>
            {explainError && (
              <p
                style={{
                  fontSize: 13,
                  color: '#b91c1c',
                  marginBottom: 6,
                }}
              >
                {explainError}
              </p>
            )}
            {explain && (
              <>
                <p
                  style={{
                    fontSize: 13,
                    marginBottom: 6,
                  }}
                >
                  <strong>Model prediction:</strong>{' '}
                  {explain.predicted_label ?? 'Unknown'} (
                  {(
                    explain.predicted_confidence * 100
                  ).toFixed(1)}{' '}
                  %)
                </p>
                <p
                  style={{
                    fontSize: 13,
                    marginBottom: 8,
                  }}
                >
                  <strong>
                    Match with your target &quot;{itemLabel}
                    &quot;:
                  </strong>{' '}
                  {(
                    (explain.match_confidence ?? 0) * 100
                  ).toFixed(1)}{' '}
                  %
                </p>
                <div
                  style={{
                    display: 'flex',
                    gap: 16,
                    alignItems: 'flex-start',
                    flexWrap: 'wrap',
                  }}
                >
                  <div>
                    <p
                      style={{
                        fontSize: 12,
                        color: '#4b5563',
                        marginBottom: 4,
                      }}
                    >
                      Red areas show where the CNN paid the most
                      attention when deciding the letter/shape.
                    </p>
                    <img
                      src={explain.saliency_image}
                      alt="Saliency heatmap"
                      style={{
                        width: 220,
                        height: 220,
                        borderRadius: 16,
                        border: '1px solid #e5e7eb',
                        objectFit: 'contain',
                        background: '#ffffff',
                      }}
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );

  return (
    <div
      style={{
        minHeight: '100vh',
        padding: 24,
        background: currentTheme.pageBackground,
        fontFamily:
          'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <div
        style={{
          maxWidth: 1080,
          margin: '0 auto',
          background: '#ffffff',
          borderRadius: 16,
          padding: 24,
          boxShadow: '0 10px 25px rgba(15,23,42,0.15)',
        }}
      >
        {/* Header with kid-friendly theme selector */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 8,
            gap: 12,
            flexWrap: 'wrap',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
            }}
          >
            <div
              style={{
                width: 46,
                height: 46,
                borderRadius: 999,
                background: currentTheme.headerOrbGradient,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 0 3px #fee2e2',
              }}
            >
              <span style={{ fontSize: 26 }}>
                {currentTheme.icon}
              </span>
            </div>
            <div>
              <h1
                style={{
                  fontSize: 24,
                  margin: 0,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                Kid Drawing Coach
                <span style={{ fontSize: 20 }}>🎨</span>
              </h1>
              <p
                style={{
                  fontSize: 12,
                  color: '#6b7280',
                  margin: 0,
                }}
              >
                AI-powered helper to practice letters, shapes and
                confidence.
              </p>
            </div>
          </div>

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-end',
              gap: 8,
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: '#9ca3af',
                padding: '4px 10px',
                borderRadius: 9999,
                border: '1px dashed #e5e7eb',
                background: '#f9fafb',
              }}
            >
              Built with FastAPI · PyTorch · React · FER
            </div>

            {/* Theme selector */}
            <div
              style={{
                display: 'flex',
                gap: 6,
                alignItems: 'center',
              }}
            >
              <span
                style={{
                  fontSize: 11,
                  color: '#6b7280',
                }}
              >
                Theme:
              </span>
              {(Object.keys(THEMES) as ThemeKey[]).map((key) => {
                const config = THEMES[key];
                const active = theme === key;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setTheme(key)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '3px 8px',
                      borderRadius: 999,
                      border: active
                        ? `1px solid ${currentTheme.accent}`
                        : '1px solid #e5e7eb',
                      background: active
                        ? currentTheme.accentSoft
                        : '#ffffff',
                      cursor: 'pointer',
                      fontSize: 11,
                      color: '#111827',
                    }}
                  >
                    <span>{config.icon}</span>
                    <span>{config.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div
          style={{
            display: 'flex',
            gap: 8,
            marginTop: 12,
            marginBottom: 20,
            borderBottom: '1px solid #e5e7eb',
          }}
        >
          {[
            { key: 'coach', label: 'Drawing Coach' },
            { key: 'students', label: 'Students Dashboard' },
            { key: 'project', label: 'Project Presentation' },
          ].map((tab) => {
            const active = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveTab(tab.key as TabKey)}
                style={{
                  padding: '6px 14px',
                  border: 'none',
                  borderBottom: active
                    ? `2px solid ${currentTheme.accent}`
                    : '2px solid transparent',
                  background: 'transparent',
                  cursor: 'pointer',
                  fontSize: 14,
                  fontWeight: active ? 600 : 400,
                  color: active ? '#111827' : '#6b7280',
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        {activeTab === 'coach' && renderCoachTab()}
        {activeTab === 'students' && <StudentsDashboard />}
        {activeTab === 'project' && <ProjectOverview />}
      </div>
    </div>
  );
};

export default App;
