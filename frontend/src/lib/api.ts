import type { GameState, Pose, PreviewResponse } from "../types/game";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_CHALLENGE_TRUCK = { depth: 2.0, width: 2.6, height: 2.75 } as const;

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

type SpectatorVariant = "local" | "challenge";

type ChallengeStatusSnapshot = {
  game_id: string;
  game_status: "in_progress" | "completed";
  mode: "dev" | "compete";
  boxes_placed: number;
  boxes_remaining: number;
  density: number;
  placed_boxes: GameState["placed_boxes"];
  current_box: GameState["current_box"];
};

function normalizeChallengeStatus(snapshot: ChallengeStatusSnapshot): GameState {
  return {
    game_id: snapshot.game_id,
    status: snapshot.game_status === "in_progress" ? "ok" : "terminated",
    truck: { ...DEFAULT_CHALLENGE_TRUCK },
    placed_boxes: snapshot.placed_boxes,
    current_box: snapshot.current_box,
    boxes_remaining: snapshot.boxes_remaining,
    density: snapshot.density,
    game_status: snapshot.game_status,
    termination_reason: null,
    mode: snapshot.mode,
    created_at: new Date().toISOString(),
    current_box_started_at: null,
    current_box_deadline: null,
    timeout_seconds: 0,
    loading_guide_x: null,
  };
}

export async function fetchSpectatorStatus(gameId: string, variant?: SpectatorVariant): Promise<{ game: GameState; variant: SpectatorVariant }> {
  if (variant === "local") {
    return { game: await fetchStatus(gameId), variant };
  }
  if (variant === "challenge") {
    return {
      game: normalizeChallengeStatus(await request<ChallengeStatusSnapshot>(`/challenge/api/status/${gameId}`)),
      variant,
    };
  }
  try {
    return { game: await fetchStatus(gameId), variant: "local" };
  } catch (localError) {
    try {
      return {
        game: normalizeChallengeStatus(await request<ChallengeStatusSnapshot>(`/challenge/api/status/${gameId}`)),
        variant: "challenge",
      };
    } catch {
      throw localError;
    }
  }
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
