// src/types.ts

export type Mode = 'LETTER' | 'SHAPE';

export interface TrajectoryPoint {
  x: number;
  y: number;
  t: number;
}

export interface AttemptCreatePayload {
  user_name: string;
  user_age?: number;
  item_type: Mode;
  item_label: string;
  started_at: string; // ISO
  ended_at: string;   // ISO
  num_corrections: number;
  num_undos: number;
  face_mood?: string;
  drawing_base64?: string;
  trajectory: TrajectoryPoint[];
}

export interface AttemptResponse {
  id: number;
  score: number;
  success: boolean;
  predicted_mood?: string;
  stress_level: number;
  coach_comment?: string;
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
