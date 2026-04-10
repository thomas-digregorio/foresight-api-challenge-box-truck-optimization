import type { GameState, Pose, PreviewResponse } from "../types/game";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ message: "Request failed." }));
    throw new Error(payload.message ?? "Request failed.");
  }
  return (await response.json()) as T;
}

export async function startGame(seed?: number): Promise<{ game_id: string }> {
  return request("/local/api/start", {
    method: "POST",
    body: JSON.stringify({
      api_key: "local-dev",
      mode: "dev",
      seed,
    }),
  });
}

export async function fetchStatus(gameId: string): Promise<GameState> {
  return request(`/local/api/status/${gameId}`);
}

export async function previewPlacement(gameId: string, boxId: string, pose: Pose): Promise<PreviewResponse> {
  return request("/local/api/preview", {
    method: "POST",
    body: JSON.stringify({
      game_id: gameId,
      box_id: boxId,
      position: pose.position,
      orientation_wxyz: pose.orientationWxyz,
    }),
  });
}

export async function placeBox(gameId: string, boxId: string, pose: Pose): Promise<GameState> {
  return request("/local/api/place", {
    method: "POST",
    body: JSON.stringify({
      game_id: gameId,
      box_id: boxId,
      position: pose.position,
      orientation_wxyz: pose.orientationWxyz,
    }),
  });
}

export async function stopGame(gameId: string): Promise<GameState> {
  return request("/local/api/stop", {
    method: "POST",
    body: JSON.stringify({
      game_id: gameId,
      api_key: "local-dev",
    }),
  });
}
