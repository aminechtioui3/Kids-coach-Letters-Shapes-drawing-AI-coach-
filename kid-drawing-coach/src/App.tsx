// src/App.tsx
import React, { useState } from 'react';
import FingerDrawingCanvas from './components/FingerDrawingCanvas';
import WebcamMoodTracker from './components/WebcamMoodTracker';

type TrajectoryPoint = { x: number; y: number; t: number };

interface AttemptResponse {
  id: number;
  score: number;
  success: boolean;
  predicted_mood: string | null;
  stress_level: number;
  coach_comment: string | null;

  match_confidence?: number | null;
  cnn_predicted_label?: string | null;
  cnn_predicted_confidence?: number | null;
}

const API_BASE = 'http://localhost:8000';

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

  const handleStart = () => {
    setStartedAt(Date.now());
    setResetKey((k) => k + 1);
    setPhase('drawing');
    setTrajectory([]);
    setFaceSnapshots([]);
    setResult(null);
    setError(null);
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
        face_mood: null,
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
    } catch (e: any) {
      console.error(e);
      setError(e.message || 'Error while sending attempt');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        padding: 24,
        background: '#f3f4f6',
        fontFamily:
          'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <div
        style={{
          maxWidth: 900,
          margin: '0 auto',
          background: '#ffffff',
          borderRadius: 16,
          padding: 24,
          boxShadow: '0 10px 25px rgba(15,23,42,0.1)',
        }}
      >
        <h1 style={{ fontSize: 24, marginBottom: 16 }}>Kid Drawing Coach 🎨</h1>
        <p style={{ fontSize: 14, color: '#4b5563', marginBottom: 24 }}>
          Draw letters or shapes with your fingertip in the air while the camera
          tracks your mood.
        </p>

        {/* Form */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 16,
            marginBottom: 24,
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
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

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 13 }}>Age</label>
            <input
              type="number"
              value={userAge}
              onChange={(e) =>
                setUserAge(e.target.value === '' ? '' : Number(e.target.value))
              }
              style={{
                borderRadius: 8,
                border: '1px solid #d1d5db',
                padding: '6px 10px',
              }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
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

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
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
        <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
          <button
            type="button"
            onClick={handleStart}
            style={{
              padding: '8px 16px',
              borderRadius: 9999,
              border: 'none',
              background: '#4f46e5',
              color: '#ffffff',
              fontWeight: 500,
              cursor: 'pointer',
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
              background: phase === 'drawing' ? '#22c55e' : '#9ca3af',
              color: '#ffffff',
              fontWeight: 500,
              cursor:
                phase === 'drawing' && !loading ? 'pointer' : 'not-allowed',
            }}
          >
            {loading ? 'Sending...' : 'Finish & analyze'}
          </button>
        </div>

        {/* Main area: drawing + webcam */}
        <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
          <FingerDrawingCanvas
            startedAt={startedAt}
            resetKey={resetKey}
            onTrajectoryChange={setTrajectory}
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
              }}
            >
              {error}
            </div>
          )}

          {result && (
            <div
              style={{
                marginTop: 16,
                padding: 16,
                borderRadius: 12,
                background: '#f9fafb',
                border: '1px solid #e5e7eb',
              }}
            >
              <h2 style={{ fontSize: 18, marginBottom: 8 }}>Result</h2>
              <p style={{ fontSize: 14, marginBottom: 4 }}>
                <strong>Score:</strong> {result.score.toFixed(1)} / 100
              </p>
              <p style={{ fontSize: 14, marginBottom: 4 }}>
                <strong>Success:</strong> {result.success ? 'Yes 🎉' : 'Not yet'}
              </p>
              <p style={{ fontSize: 14, marginBottom: 4 }}>
                <strong>Predicted mood:</strong>{' '}
                {result.predicted_mood ?? 'Unknown'}
              </p>
              <p style={{ fontSize: 14, marginBottom: 4 }}>
                <strong>Stress level:</strong>{' '}
                {(result.stress_level * 100).toFixed(0)} %
              </p>

              {/* Matching info */}
              {typeof result.match_confidence === 'number' && (
                <p style={{ fontSize: 14, marginBottom: 4 }}>
                  <strong>Match with your target "{itemLabel}":</strong>{' '}
                  {(result.match_confidence * 100).toFixed(1)} %
                </p>
              )}

              {result.cnn_predicted_label && (
                <p style={{ fontSize: 14, marginBottom: 4 }}>
                  <strong>Model thinks you drew:</strong>{' '}
                  {result.cnn_predicted_label}
                  {typeof result.cnn_predicted_confidence === 'number' && (
                    <>
                      {' '}
                      (
                      {(result.cnn_predicted_confidence * 100).toFixed(1)} %
                      )
                    </>
                  )}
                </p>
              )}

              <p style={{ fontSize: 14, marginTop: 8 }}>
                <strong>Coach comment:</strong> {result.coach_comment}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
