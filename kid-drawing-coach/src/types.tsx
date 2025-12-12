// src/types.ts

export type TrajectoryPoint = {
  x: number;
  y: number;
  t: number;
};

export interface AttemptCreatePayload {
  user_name: string;
  user_age: number | null;
  item_type: 'LETTER' | 'SHAPE';
  item_label: string;
  started_at: string;
  ended_at: string;
  num_corrections: number;
  num_undos: number;
  face_mood: string | null;
  drawing_base64: string | null;
  face_snapshots: string[];
  trajectory: TrajectoryPoint[];
}

export interface AttemptResponse {
  id: number;
  score: number;
  success: boolean;
  predicted_mood: string | null;
  stress_level: number;
  coach_comment: string | null;
}

export interface SummaryStats {
  total_attempts: number;
  total_time_seconds: number;
  avg_score: number;
  success_rate: number;
  mood_counts: Record<string, number>;
}

export interface ProgressPoint {
  date: string;
  avg_score: number;
  total_attempts: number;
  total_time_seconds: number;
  avg_stress_level: number;
}

export interface ProgressStats {
  points: ProgressPoint[];
}

export interface ItemStats {
  item_type: string;
  label: string;
  attempts: number;
  avg_score: number;
  success_rate: number;
  avg_duration_seconds: number;
  avg_stress_level: number;
}

export interface ItemsStatsResponse {
  items: ItemStats[];
}

export interface UserSummary {
  name: string;
  age: number | null;
  created_at: string;
  total_attempts: number;
  avg_score: number;
  avg_stress_level: number;
  last_attempt_at: string | null;
}

export interface AttemptHistoryItem {
  id: number;
  user_name: string;
  item_type: string;
  item_label: string;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  score: number;
  success: boolean;
  predicted_mood: string | null;
  stress_level: number;
}
