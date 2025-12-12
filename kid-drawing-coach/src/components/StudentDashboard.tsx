// src/components/StudentsDashboard.tsx
import React, { useEffect, useState } from 'react';
import {
  getUsersWithStats,
  getProgress,
  getAttemptsHistory,
} from '../api';
import type {
  UserSummary,
  ProgressStats,
  AttemptHistoryItem,
} from '../types';

const formatDate = (iso: string | null) => {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString();
};

const formatDateTime = (iso: string) => {
  const d = new Date(iso);
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })}`;
};

const ScoreChart: React.FC<{ progress: ProgressStats }> = ({ progress }) => {
  const points = progress.points;
  if (!points.length) {
    return <p style={{ fontSize: 12, color: '#6b7280' }}>No attempts yet.</p>;
  }

  const maxScore = Math.max(...points.map((p) => p.avg_score), 1);

  return (
    <div>
      <div
        style={{
          height: 180,
          display: 'flex',
          alignItems: 'flex-end',
          gap: 6,
          padding: '4px 0',
        }}
      >
        {points.map((p) => {
          const h = (p.avg_score / maxScore) * 150;
          return (
            <div
              key={p.date}
              title={`${p.date}: ${p.avg_score.toFixed(1)} / 100`}
              style={{
                width: 14,
                height: `${h}px`,
                borderRadius: 6,
                background:
                  'linear-gradient(180deg,#4f46e5,#6366f1)',
              }}
            />
          );
        })}
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 10,
          color: '#9ca3af',
          marginTop: 4,
        }}
      >
        {points.map((p) => (
          <span key={p.date}>{p.date.slice(5)}</span>
        ))}
      </div>
    </div>
  );
};

const StressChart: React.FC<{ progress: ProgressStats }> = ({ progress }) => {
  const points = progress.points;
  if (!points.length) {
    return <p style={{ fontSize: 12, color: '#6b7280' }}>No attempts yet.</p>;
  }

  return (
    <div>
      <div
        style={{
          height: 180,
          display: 'flex',
          alignItems: 'flex-end',
          gap: 6,
          padding: '4px 0',
        }}
      >
        {points.map((p) => {
          const stressPct = p.avg_stress_level * 100;
          const h = (stressPct / 100) * 150;
          const color =
            stressPct < 35
              ? 'linear-gradient(180deg,#22c55e,#4ade80)'
              : stressPct < 70
              ? 'linear-gradient(180deg,#eab308,#facc15)'
              : 'linear-gradient(180deg,#ef4444,#f97373)';
          return (
            <div
              key={p.date}
              title={`${p.date}: ${stressPct.toFixed(0)} %`}
              style={{
                width: 14,
                height: `${h}px`,
                borderRadius: 6,
                background: color,
              }}
            />
          );
        })}
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 10,
          color: '#9ca3af',
          marginTop: 4,
        }}
      >
        {points.map((p) => (
          <span key={p.date}>{p.date.slice(5)}</span>
        ))}
      </div>
    </div>
  );
};

const StudentsDashboard: React.FC = () => {
  const [students, setStudents] = useState<UserSummary[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressStats | null>(null);
  const [history, setHistory] = useState<AttemptHistoryItem[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      setLoadingList(true);
      setError(null);
      try {
        const data = await getUsersWithStats();
        setStudents(data);
      } catch (e: any) {
        console.error(e);
        setError(e.message || 'Error loading students');
      } finally {
        setLoadingList(false);
      }
    };
    load();
  }, []);

  useEffect(() => {
    if (!selectedName) {
      setProgress(null);
      setHistory([]);
      return;
    }
    const loadDetail = async () => {
      setLoadingDetail(true);
      setError(null);
      try {
        const [prog, hist] = await Promise.all([
          getProgress(selectedName),
          getAttemptsHistory(selectedName),
        ]);
        setProgress(prog);
        setHistory(hist);
      } catch (e: any) {
        console.error(e);
        setError(e.message || 'Error loading student details');
      } finally {
        setLoadingDetail(false);
      }
    };
    loadDetail();
  }, [selectedName]);

  const selectedSummary = selectedName
    ? students.find((s) => s.name === selectedName)
    : undefined;

  const computeInsights = (): string[] => {
    if (!progress || !progress.points.length) return [];
    const pts = progress.points;

    const totalAttempts = pts.reduce(
      (acc, p) => acc + p.total_attempts,
      0
    );
    const avgScoreAll =
      totalAttempts > 0
        ? pts.reduce(
            (acc, p) => acc + p.avg_score * p.total_attempts,
            0
          ) / totalAttempts
        : 0;
    const avgStressAll =
      totalAttempts > 0
        ? pts.reduce(
            (acc, p) =>
              acc + p.avg_stress_level * p.total_attempts,
            0
          ) / totalAttempts
        : 0;

    const last = pts[pts.length - 1];

    const insights: string[] = [];
    insights.push(
      `Average score so far: ${avgScoreAll.toFixed(1)} / 100.`
    );
    insights.push(
      `Average stress level: ${(avgStressAll * 100).toFixed(0)} %.`
    );
    insights.push(
      `Last practice day (${last.date}): score ${last.avg_score.toFixed(
        1
      )}, stress ${(last.avg_stress_level * 100).toFixed(0)} %.`
    );

    if (last.avg_score > avgScoreAll + 5) {
      insights.push('Trend: improving nicely over time ✅');
    } else if (last.avg_score < avgScoreAll - 5) {
      insights.push('Trend: score dipped recently, more practice can help 💡');
    } else {
      insights.push('Trend: score is stable, keep going 👍');
    }

    if (avgStressAll < 0.3) {
      insights.push('This child usually looks relaxed while drawing 🧘‍♂️');
    } else if (avgStressAll > 0.6) {
      insights.push(
        'Stress is often high, we can suggest more breaks and encouragement 💙'
      );
    }

    return insights;
  };

  const insights = computeInsights();

  return (
    <div>
      <p style={{ fontSize: 14, color: '#4b5563', marginBottom: 16 }}>
        Explore all students, their average scores, stress levels and
        detailed drawing history.
      </p>

      {error && (
        <div
          style={{
            padding: 10,
            borderRadius: 8,
            background: '#fee2e2',
            color: '#b91c1c',
            fontSize: 13,
            marginBottom: 12,
          }}
        >
          {error}
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1.1fr 1.9fr',
          gap: 20,
        }}
      >
        {/* LEFT: students list */}
        <div
          style={{
            borderRadius: 12,
            border: '1px solid #e5e7eb',
            padding: 12,
            background: '#f9fafb',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: 8,
            }}
          >
            <h2
              style={{
                fontSize: 16,
                margin: 0,
              }}
            >
              Students
            </h2>
            {loadingList && (
              <span style={{ fontSize: 11, color: '#9ca3af' }}>
                Loading...
              </span>
            )}
          </div>

          {students.length === 0 && !loadingList && (
            <p style={{ fontSize: 13, color: '#6b7280' }}>
              No students yet. Start drawing in the main tab to create some
              attempts.
            </p>
          )}

          {students.length > 0 && (
            <div
              style={{
                maxHeight: 280,
                overflowY: 'auto',
              }}
            >
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: 12,
                }}
              >
                <thead>
                  <tr
                    style={{
                      background: '#e5e7eb',
                      position: 'sticky',
                      top: 0,
                    }}
                  >
                    <th
                      style={{
                        textAlign: 'left',
                        padding: '6px 8px',
                        fontWeight: 600,
                      }}
                    >
                      Name
                    </th>
                    <th
                      style={{
                        textAlign: 'left',
                        padding: '6px 8px',
                        fontWeight: 600,
                      }}
                    >
                      Attempts
                    </th>
                    <th
                      style={{
                        textAlign: 'left',
                        padding: '6px 8px',
                        fontWeight: 600,
                      }}
                    >
                      Avg score
                    </th>
                    <th
                      style={{
                        textAlign: 'left',
                        padding: '6px 8px',
                        fontWeight: 600,
                      }}
                    >
                      Avg stress
                    </th>
                    <th
                      style={{
                        textAlign: 'left',
                        padding: '6px 8px',
                        fontWeight: 600,
                      }}
                    >
                      Last attempt
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {students.map((s) => {
                    const selected = s.name === selectedName;
                    return (
                      <tr
                        key={s.name + s.created_at}
                        onClick={() => setSelectedName(s.name)}
                        style={{
                          cursor: 'pointer',
                          background: selected
                            ? '#eef2ff'
                            : 'transparent',
                        }}
                      >
                        <td style={{ padding: '6px 8px' }}>{s.name}</td>
                        <td style={{ padding: '6px 8px' }}>
                          {s.total_attempts}
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          {s.avg_score.toFixed(1)}
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          {(s.avg_stress_level * 100).toFixed(0)} %
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          {formatDate(s.last_attempt_at)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* RIGHT: details for selected student */}
        <div
          style={{
            borderRadius: 12,
            border: '1px solid #e5e7eb',
            padding: 14,
            background: '#ffffff',
          }}
        >
          {!selectedName && (
            <p style={{ fontSize: 13, color: '#6b7280' }}>
              Select a student on the left to see their progress and
              history.
            </p>
          )}

          {selectedName && (
            <>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 8,
                }}
              >
                <div>
                  <h2
                    style={{
                      fontSize: 18,
                      margin: 0,
                      marginBottom: 2,
                    }}
                  >
                    {selectedName}
                  </h2>
                  {selectedSummary && (
                    <p
                      style={{
                        fontSize: 12,
                        color: '#6b7280',
                        margin: 0,
                      }}
                    >
                      Attempts: {selectedSummary.total_attempts} · Avg
                      score: {selectedSummary.avg_score.toFixed(1)} / 100 ·
                      Avg stress:{' '}
                      {(
                        selectedSummary.avg_stress_level * 100
                      ).toFixed(0)}{' '}
                      %
                    </p>
                  )}
                </div>
                {loadingDetail && (
                  <span style={{ fontSize: 11, color: '#9ca3af' }}>
                    Loading details...
                  </span>
                )}
              </div>

              {/* Charts */}
              {progress && (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '1.1fr 0.9fr',
                    gap: 16,
                    marginTop: 8,
                    marginBottom: 12,
                  }}
                >
                  <div
                    style={{
                      borderRadius: 10,
                      border: '1px solid #e5e7eb',
                      padding: 10,
                      background: '#f9fafb',
                    }}
                  >
                    <div
                      style={{
                        fontSize: 13,
                        marginBottom: 4,
                        fontWeight: 500,
                      }}
                    >
                      Scores over time
                    </div>
                    <ScoreChart progress={progress} />
                  </div>
                  <div
                    style={{
                      borderRadius: 10,
                      border: '1px solid #e5e7eb',
                      padding: 10,
                      background: '#f9fafb',
                    }}
                  >
                    <div
                      style={{
                        fontSize: 13,
                        marginBottom: 4,
                        fontWeight: 500,
                      }}
                    >
                      Stress over time
                    </div>
                    <StressChart progress={progress} />
                  </div>
                </div>
              )}

              {/* Insights */}
              {insights.length > 0 && (
                <div
                  style={{
                    borderRadius: 10,
                    border: '1px solid #e5e7eb',
                    padding: 10,
                    marginBottom: 12,
                    background: '#fef9c3',
                  }}
                >
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      marginBottom: 4,
                    }}
                  >
                    Insights for this student
                  </div>
                  <ul
                    style={{
                      margin: 0,
                      paddingLeft: 18,
                      fontSize: 12,
                      color: '#4b5563',
                    }}
                  >
                    {insights.map((ins, idx) => (
                      <li key={idx}>{ins}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* History table */}
              <div
                style={{
                  borderRadius: 10,
                  border: '1px solid #e5e7eb',
                  padding: 10,
                  maxHeight: 200,
                  overflowY: 'auto',
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 500,
                    marginBottom: 6,
                  }}
                >
                  Recent attempts
                </div>
                {history.length === 0 ? (
                  <p style={{ fontSize: 12, color: '#6b7280' }}>
                    No attempts yet.
                  </p>
                ) : (
                  <table
                    style={{
                      width: '100%',
                      borderCollapse: 'collapse',
                      fontSize: 11,
                    }}
                  >
                    <thead>
                      <tr
                        style={{
                          background: '#f3f4f6',
                          position: 'sticky',
                          top: 0,
                        }}
                      >
                        <th
                          style={{
                            textAlign: 'left',
                            padding: '4px 6px',
                          }}
                        >
                          Date
                        </th>
                        <th
                          style={{
                            textAlign: 'left',
                            padding: '4px 6px',
                          }}
                        >
                          Item
                        </th>
                        <th
                          style={{
                            textAlign: 'left',
                            padding: '4px 6px',
                          }}
                        >
                          Score
                        </th>
                        <th
                          style={{
                            textAlign: 'left',
                            padding: '4px 6px',
                          }}
                        >
                          Mood
                        </th>
                        <th
                          style={{
                            textAlign: 'left',
                            padding: '4px 6px',
                          }}
                        >
                          Stress
                        </th>
                        <th
                          style={{
                            textAlign: 'left',
                            padding: '4px 6px',
                          }}
                        >
                          OK?
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.slice(0, 40).map((h) => (
                        <tr key={h.id}>
                          <td style={{ padding: '4px 6px' }}>
                            {formatDateTime(h.started_at)}
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {h.item_type}:{' '}
                            <strong>{h.item_label}</strong>
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {h.score.toFixed(1)}
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {h.predicted_mood ?? '—'}
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {(h.stress_level * 100).toFixed(0)} %
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {h.success ? '✅' : '❌'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default StudentsDashboard;
