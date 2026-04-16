import { create } from "zustand";
import { fetchSpectatorStatus, fetchStatus, placeBox, previewPlacement, startGame, stopGame } from "../lib/api";
import { ORIENTATION_PRESETS, eulerDegreesToQuaternionWxyz, normalizeQuaternionWxyz, quaternionWxyzToEulerDegrees } from "../lib/quaternion";
import type { BoxPayload, GameState, Pose, PreviewResponse, QuaternionWxyz, Vec3 } from "../types/game";

type GameMode = "idle" | "playing" | "finished";

type GameStore = {
  mode: GameMode;
  game: GameState | null;
  isSpectating: boolean;
  spectatorGameId: string | null;
  spectatorApiVariant: "local" | "challenge" | null;
  pose: Pose;
  preview: PreviewResponse | null;
  isStarting: boolean;
  isPlacing: boolean;
  isPreviewSyncing: boolean;
  error: string | null;
  showAdvanced: boolean;
  cameraResetToken: number;
  cameraZoomToken: number;
  cameraZoomDirection: "in" | "out" | null;
  previewRequestId: number;
  poseVersion: number;
  previewPoseVersion: number;
  autoProjectToSupport: boolean;
  seed: number | null;
  startNewGame: () => Promise<void>;
  attachToGame: (gameId: string) => Promise<void>;
  exitSpectatorMode: () => void;
  refreshStatus: () => Promise<void>;
  syncPreview: () => Promise<void>;
  confirmPlacement: () => Promise<void>;
  stopCurrentGame: () => Promise<void>;
  setPosition: (partial: Partial<Record<"x" | "y" | "z", number>>) => void;
  setQuaternion: (quaternion: QuaternionWxyz) => void;
  setEuler: (partial: Partial<Record<"roll" | "pitch" | "yaw", number>>) => void;
  nudgePosition: (delta: Vec3) => void;
  applyPreset: (presetId: string) => void;
  toggleAdvanced: () => void;
  resetCamera: () => void;
  zoomCamera: (direction: "in" | "out") => void;
};

function defaultPoseForBox(box: BoxPayload | null, truck: GameState["truck"] | null): Pose {
  if (!box || !truck) {
    return {
      position: [0.5, 1.3, 0.5],
      orientationWxyz: [1, 0, 0, 0],
    };
  }
  return {
    position: [
      Math.max(box.dimensions[0] / 2 + 0.06, 0.18),
      truck.width / 2,
      box.dimensions[2] / 2,
    ],
    orientationWxyz: [1, 0, 0, 0],
  };
}

function withClampedPosition(game: GameState | null, position: Vec3): Vec3 {
  if (!game) {
    return position;
  }
  return [
    Math.min(Math.max(position[0], 0), game.truck.depth),
    Math.min(Math.max(position[1], 0), game.truck.width),
    Math.min(Math.max(position[2], 0), game.truck.height),
  ];
}

function mergedPose(currentPose: Pose, partial: Partial<Record<"x" | "y" | "z", number>>, game: GameState | null): Pose {
  const nextPosition: Vec3 = [
    partial.x ?? currentPose.position[0],
    partial.y ?? currentPose.position[1],
    partial.z ?? currentPose.position[2],
  ];
  return {
    ...currentPose,
    position: withClampedPosition(game, nextPosition),
  };
}

function sameArray(valuesA: number[], valuesB: number[]): boolean {
  return valuesA.length === valuesB.length && valuesA.every((value, index) => Math.abs(value - valuesB[index]) < 1e-6);
}

function samePose(poseA: Pose, poseB: Pose): boolean {
  return sameArray(poseA.position, poseB.position) && sameArray(poseA.orientationWxyz, poseB.orientationWxyz);
}

function actionToPose(
  action:
    | {
        position: Vec3;
        orientation_wxyz: QuaternionWxyz;
      }
    | null
    | undefined,
): Pose | null {
  if (!action) {
    return null;
  }
  return {
    position: action.position,
    orientationWxyz: normalizeQuaternionWxyz(action.orientation_wxyz),
  };
}

export const useGameStore = create<GameStore>((set, get) => ({
  mode: "idle",
  game: null,
  isSpectating: false,
  spectatorGameId: null,
  spectatorApiVariant: null,
  pose: { position: [0.5, 1.3, 0.5], orientationWxyz: [1, 0, 0, 0] },
  preview: null,
  isStarting: false,
  isPlacing: false,
  isPreviewSyncing: false,
  error: null,
  showAdvanced: false,
  cameraResetToken: 0,
  cameraZoomToken: 0,
  cameraZoomDirection: null,
  previewRequestId: 0,
  poseVersion: 0,
  previewPoseVersion: 0,
  autoProjectToSupport: false,
  seed: null,
  startNewGame: async () => {
    set({ isStarting: true, error: null });
    try {
      const seed = Math.floor(Math.random() * 100_000);
      const start = await startGame(seed);
      const game = await fetchStatus(start.game_id);
      const pose = defaultPoseForBox(game.current_box, game.truck);
      set({
        seed,
        isSpectating: false,
        spectatorGameId: null,
        spectatorApiVariant: null,
        mode: game.game_status === "in_progress" ? "playing" : "finished",
        game,
        preview: null,
        pose,
        poseVersion: 1,
        previewPoseVersion: 0,
        isPreviewSyncing: game.game_status === "in_progress" && Boolean(game.current_box),
        autoProjectToSupport: false,
        isStarting: false,
        cameraResetToken: get().cameraResetToken + 1,
      });
      if (game.current_box) {
        await get().syncPreview();
      }
    } catch (error) {
      set({ isStarting: false, error: error instanceof Error ? error.message : "Failed to start a game." });
    }
  },
  attachToGame: async (gameId) => {
    const trimmedGameId = gameId.trim();
    if (!trimmedGameId) {
      set({ error: "Enter a game ID to spectate." });
      return;
    }
    set({ isStarting: true, error: null });
    try {
      const { game: snapshot, variant } = await fetchSpectatorStatus(trimmedGameId);
      const pose = defaultPoseForBox(snapshot.current_box, snapshot.truck);
      set({
        seed: null,
        isSpectating: true,
        spectatorGameId: snapshot.game_id,
        spectatorApiVariant: variant,
        mode: snapshot.game_status === "in_progress" ? "playing" : "finished",
        game: snapshot,
        preview: null,
        pose,
        poseVersion: 1,
        previewPoseVersion: 0,
        isPreviewSyncing: false,
        autoProjectToSupport: false,
        isStarting: false,
        cameraResetToken: get().cameraResetToken + 1,
      });
    } catch (error) {
      set({
        isStarting: false,
        error: error instanceof Error ? error.message : "Failed to attach to the game.",
      });
    }
  },
  exitSpectatorMode: () =>
    set({
      mode: "idle",
      game: null,
      isSpectating: false,
      spectatorGameId: null,
      spectatorApiVariant: null,
      preview: null,
      pose: defaultPoseForBox(null, null),
      poseVersion: 0,
      previewPoseVersion: 0,
      isPreviewSyncing: false,
      autoProjectToSupport: false,
      error: null,
    }),
  refreshStatus: async () => {
    const game = get().game;
    if (!game) {
      return;
    }
    try {
      const { game: snapshot } = get().isSpectating
        ? await fetchSpectatorStatus(game.game_id, get().spectatorApiVariant ?? undefined)
        : { game: await fetchStatus(game.game_id) };
      const previousBoxId = game.current_box?.id;
      const boxChanged = snapshot.current_box?.id !== previousBoxId;
      const nextPose = boxChanged || get().isSpectating ? defaultPoseForBox(snapshot.current_box, snapshot.truck) : get().pose;
      set({
        game: snapshot,
        pose: nextPose,
        preview: get().isSpectating || boxChanged || snapshot.game_status !== "in_progress" ? null : get().preview,
        poseVersion: boxChanged ? get().poseVersion + 1 : get().poseVersion,
        previewPoseVersion: get().isSpectating || boxChanged || snapshot.game_status !== "in_progress" ? 0 : get().previewPoseVersion,
        isPreviewSyncing: !get().isSpectating && boxChanged && snapshot.game_status === "in_progress",
        autoProjectToSupport: false,
        mode: snapshot.game_status === "in_progress" ? "playing" : "finished",
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "Failed to refresh status." });
    }
  },
  syncPreview: async () => {
    const game = get().game;
    const currentBox = game?.current_box;
    if (!game || !currentBox || game.game_status !== "in_progress" || get().isSpectating) {
      return;
    }
    const requestId = get().previewRequestId + 1;
    const poseVersion = get().poseVersion;
    const pose = get().pose;
    set({ previewRequestId: requestId, error: null, isPreviewSyncing: true });
    try {
      const response = await previewPlacement(game.game_id, currentBox.id, pose);
      if (get().previewRequestId !== requestId) {
        return;
      }
      const nextPose =
        response.normalized_position && response.normalized_orientation_wxyz
          ? {
              position: response.normalized_position,
              orientationWxyz: normalizeQuaternionWxyz(response.normalized_orientation_wxyz),
            }
          : get().pose;
      const projectedPose = actionToPose(response.repair_suggestions.support_aligned);
      if (get().autoProjectToSupport && projectedPose && !samePose(nextPose, projectedPose)) {
        set((state) => ({
          pose: projectedPose,
          preview: null,
          poseVersion: state.poseVersion + 1,
          previewPoseVersion: 0,
          isPreviewSyncing: true,
          autoProjectToSupport: false,
        }));
        return;
      }
      if (get().autoProjectToSupport) {
        set({
          preview: response,
          pose: nextPose,
          previewPoseVersion: poseVersion,
          isPreviewSyncing: false,
          autoProjectToSupport: false,
          mode: response.game_status === "in_progress" ? get().mode : "finished",
        });
        if (response.game_status !== "in_progress") {
          await get().refreshStatus();
        }
        return;
      }
      const suggestedPose = actionToPose(response.repair_suggestions.nearby_valid) ?? actionToPose(response.repair_suggestions.any_valid);
      if (!response.is_valid && suggestedPose && !samePose(nextPose, suggestedPose)) {
        set((state) => ({
          pose: suggestedPose,
          preview: null,
          poseVersion: state.poseVersion + 1,
          previewPoseVersion: 0,
          isPreviewSyncing: true,
          autoProjectToSupport: false,
        }));
        return;
      }
      set({
        preview: response,
        pose: nextPose,
        previewPoseVersion: poseVersion,
        isPreviewSyncing: false,
        autoProjectToSupport: false,
        mode: response.game_status === "in_progress" ? get().mode : "finished",
      });
      if (response.game_status !== "in_progress") {
        await get().refreshStatus();
      }
    } catch (error) {
      if (get().previewRequestId !== requestId) {
        return;
      }
      set({
        error: error instanceof Error ? error.message : "Preview sync failed.",
        isPreviewSyncing: false,
        autoProjectToSupport: false,
      });
    }
  },
  confirmPlacement: async () => {
    const game = get().game;
    const currentBox = game?.current_box;
    const preview = get().preview;
    if (
      !game ||
      !currentBox ||
      get().isSpectating ||
      get().isPlacing ||
      get().isPreviewSyncing ||
      !preview?.is_valid ||
      get().previewPoseVersion !== get().poseVersion
    ) {
      return;
    }
    set({ isPlacing: true, error: null });
    try {
      const snapshot = await placeBox(game.game_id, currentBox.id, get().pose);
      const nextPose = defaultPoseForBox(snapshot.current_box, snapshot.truck);
      set({
        game: snapshot,
        preview: null,
        pose: nextPose,
        poseVersion: snapshot.current_box ? get().poseVersion + 1 : get().poseVersion,
        previewPoseVersion: 0,
        isPreviewSyncing: snapshot.game_status === "in_progress" && Boolean(snapshot.current_box),
        autoProjectToSupport: false,
        mode: snapshot.game_status === "in_progress" ? "playing" : "finished",
        isPlacing: false,
        cameraResetToken: get().cameraResetToken + (snapshot.current_box ? 0 : 1),
      });
      if (snapshot.current_box) {
        await get().syncPreview();
      }
    } catch (error) {
      set({
        isPlacing: false,
        error: error instanceof Error ? error.message : "Placement failed.",
      });
    }
  },
  stopCurrentGame: async () => {
    const game = get().game;
    if (!game) {
      return;
    }
    if (get().isSpectating) {
      get().exitSpectatorMode();
      return;
    }
    try {
      const snapshot = await stopGame(game.game_id);
      set({ game: snapshot, mode: "finished", preview: null, isPreviewSyncing: false, autoProjectToSupport: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "Failed to stop the game." });
    }
  },
  setPosition: (partial) => {
    set((state) => {
      if (state.isSpectating) {
        return state;
      }
      const nextPose = mergedPose(state.pose, partial, state.game);
      if (samePose(state.pose, nextPose)) {
        return state;
      }
      const shouldProjectToSupport = partial.z === undefined && (partial.x !== undefined || partial.y !== undefined);
      return {
        pose: nextPose,
        preview: null,
        poseVersion: state.poseVersion + 1,
        previewPoseVersion: 0,
        isPreviewSyncing: true,
        autoProjectToSupport: shouldProjectToSupport,
      };
    });
  },
  setQuaternion: (quaternion) => {
    set((state) => ({
      ...(state.isSpectating
        ? {}
        : samePose(state.pose, {
              ...state.pose,
              orientationWxyz: normalizeQuaternionWxyz(quaternion),
            })
          ? {}
          : {
              pose: {
                ...state.pose,
                orientationWxyz: normalizeQuaternionWxyz(quaternion),
              },
              preview: null,
              poseVersion: state.poseVersion + 1,
              previewPoseVersion: 0,
              isPreviewSyncing: true,
              autoProjectToSupport: false,
            }),
    }));
  },
  setEuler: (partial) => {
    const [roll, pitch, yaw] = quaternionWxyzToEulerDegrees(get().pose.orientationWxyz);
    const nextQuaternion = eulerDegreesToQuaternionWxyz(
      partial.roll ?? roll,
      partial.pitch ?? pitch,
      partial.yaw ?? yaw,
    );
    set((state) => {
      if (state.isSpectating) {
        return state;
      }
      const nextPose = {
        ...state.pose,
        orientationWxyz: nextQuaternion,
      };
      if (samePose(state.pose, nextPose)) {
        return state;
      }
      return {
        pose: nextPose,
        preview: null,
        poseVersion: state.poseVersion + 1,
        previewPoseVersion: 0,
        isPreviewSyncing: true,
        autoProjectToSupport: false,
      };
    });
  },
  nudgePosition: (delta) => {
    set((state) => {
      if (state.isSpectating) {
        return state;
      }
      const [x, y, z] = state.pose.position;
      const nextPose = mergedPose(
        state.pose,
        { x: x + delta[0], y: y + delta[1], z: z + delta[2] },
        state.game,
      );
      if (samePose(state.pose, nextPose)) {
        return state;
      }
      return {
        pose: nextPose,
        preview: null,
        poseVersion: state.poseVersion + 1,
        previewPoseVersion: 0,
        isPreviewSyncing: true,
        autoProjectToSupport: true,
      };
    });
  },
  applyPreset: (presetId) => {
    const preset = ORIENTATION_PRESETS.find((candidate) => candidate.id === presetId);
    if (!preset) {
      return;
    }
    get().setQuaternion(preset.quaternion);
  },
  toggleAdvanced: () => set((state) => ({ showAdvanced: !state.showAdvanced })),
  resetCamera: () =>
    set((state) => ({
      cameraResetToken: state.cameraResetToken + 1,
    })),
  zoomCamera: (direction) =>
    set((state) => ({
      cameraZoomDirection: direction,
      cameraZoomToken: state.cameraZoomToken + 1,
    })),
}));
