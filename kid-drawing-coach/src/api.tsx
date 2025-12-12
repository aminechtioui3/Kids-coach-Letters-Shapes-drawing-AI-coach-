// src/api.ts
import type {
  AttemptCreatePayload,
  AttemptResponse,
  SummaryStats,
  ProgressStats,
  ItemsStatsResponse,
  UserSummary,
  AttemptHistoryItem,
} from './types';

const API_BASE = 'http://localhost:8000';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function createAttempt(
  payload: AttemptCreatePayload
): Promise<AttemptResponse> {
  const res = await fetch(`${API_BASE}/api/attempts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<AttemptResponse>(res);
}

export async function getSummary(userName?: string): Promise<SummaryStats> {
  const url = new URL(`${API_BASE}/api/stats/summary`);
  if (userName) url.searchParams.set('user_name', userName);
  const res = await fetch(url.toString());
  return handleResponse<SummaryStats>(res);
}

export async function getProgress(userName?: string): Promise<ProgressStats> {
  const url = new URL(`${API_BASE}/api/stats/progress`);
  if (userName) url.searchParams.set('user_name', userName);
  const res = await fetch(url.toString());
  return handleResponse<ProgressStats>(res);
}

export async function getItemStats(userName?: string): Promise<ItemsStatsResponse> {
  const url = new URL(`${API_BASE}/api/stats/by-item`);
  if (userName) url.searchParams.set('user_name', userName);
  const res = await fetch(url.toString());
  return handleResponse<ItemsStatsResponse>(res);
}

export async function getUsersWithStats(): Promise<UserSummary[]> {
  const res = await fetch(`${API_BASE}/api/users/with-stats`);
  return handleResponse<UserSummary[]>(res);
}

export async function getAttemptsHistory(
  userName?: string
): Promise<AttemptHistoryItem[]> {
  const url = new URL(`${API_BASE}/api/attempts/history`);
  if (userName) url.searchParams.set('user_name', userName);
  const res = await fetch(url.toString());
  return handleResponse<AttemptHistoryItem[]>(res);
}
