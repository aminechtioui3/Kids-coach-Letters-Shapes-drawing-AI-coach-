// src/App.tsx
import React, { useState } from 'react';
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

const API_BASE = 'http://localhost:8000';

type TabKey = 'coach' | 'students' | 'project';

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
    } catch (e: any) {
      console.error(e);
      setError(e.message || 'Error while sending attempt');
    } finally {
      setLoading(false);
    }
  };

  const renderCoachTab = () => (
    <>
      <p style={{ fontSize: 14, color: '#4b5563', marginBottom: 24 }}>
        Draw letters or shapes with your fingertip in the air while the camera tracks your mood.
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
            onChange={(e) => setItemType(e.target.value as 'LETTER' | 'SHAPE')}
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
            background:
              phase === 'drawing' && !loading ? '#22c55e' : '#9ca3af',
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
            <p style={{ fontSize: 14, marginTop: 8 }}>
              <strong>Coach comment:</strong> {result.coach_comment}
            </p>
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
        background: 'linear-gradient(180deg,#e0f2fe,#f9fafb)',
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
          boxShadow: '0 10px 25px rgba(15,23,42,0.1)',
        }}
      >
        {/* Header with kids logo */}
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
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: '999px',
                background:
                  'radial-gradient(circle at 30% 20%, #f97316, #facc15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 0 3px #fee2e2',
              }}
            >
              <span style={{ fontSize: 24 }}>👧👦</span>
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
                AI-powered helper to practice letters, shapes and confidence.
              </p>
            </div>
          </div>

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
                    ? '2px solid #4f46e5'
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
