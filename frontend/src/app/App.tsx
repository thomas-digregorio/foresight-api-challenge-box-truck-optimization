import { Suspense, lazy, useEffect, useRef } from "react";
import { GameHUD } from "../features/game/GameHUD";
import { ORIENTATION_PRESETS } from "../lib/quaternion";
import { useGameStore } from "../store/gameStore";

const SceneCanvas = lazy(async () => {
  const module = await import("../features/game/SceneCanvas");
  return { default: module.SceneCanvas };
});

function SceneFallback() {
  return (
    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.42),transparent_32%),linear-gradient(180deg,rgba(255,247,234,0.7),rgba(198,184,164,0.84))]">
      <div className="absolute inset-x-0 top-16 flex justify-center">
        <div className="rounded-full border border-slate-900/10 bg-white/55 px-5 py-2 text-sm font-medium tracking-[0.24em] text-slate-800 shadow-xl backdrop-blur">
          Loading 3D Truck View
        </div>
      </div>
    </div>
  );
}

function isInputTarget(target: EventTarget | null): target is HTMLInputElement {
  return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
}

export function App() {
  const mode = useGameStore((state) => state.mode);
  const game = useGameStore((state) => state.game);
  const preview = useGameStore((state) => state.preview);
  const isSpectating = useGameStore((state) => state.isSpectating);
  const poseVersion = useGameStore((state) => state.poseVersion);
  const attachToGame = useGameStore((state) => state.attachToGame);
  const syncPreview = useGameStore((state) => state.syncPreview);
  const refreshStatus = useGameStore((state) => state.refreshStatus);
  const confirmPlacement = useGameStore((state) => state.confirmPlacement);
  const nudgePosition = useGameStore((state) => state.nudgePosition);
  const applyPreset = useGameStore((state) => state.applyPreset);
  const resetCamera = useGameStore((state) => state.resetCamera);
  const zoomCamera = useGameStore((state) => state.zoomCamera);

  const attemptedSpectateRef = useRef<string | null>(null);

  useEffect(() => {
    void import("../features/game/SceneCanvas");
    void import("../components/MiniBoxPreview");
  }, []);

  useEffect(() => {
    const spectatorGameId = new URLSearchParams(window.location.search).get("spectate");
    if (!spectatorGameId || attemptedSpectateRef.current === spectatorGameId) {
      return;
    }
    attemptedSpectateRef.current = spectatorGameId;
    void attachToGame(spectatorGameId);
  }, [attachToGame]);

  useEffect(() => {
    if (mode !== "playing" || !game?.current_box || isSpectating) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      void syncPreview();
    }, 80);
    return () => window.clearTimeout(timeoutId);
  }, [game?.current_box?.id, isSpectating, mode, poseVersion, syncPreview]);

  useEffect(() => {
    if (mode !== "playing" || !game) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void refreshStatus();
    }, isSpectating ? 250 : 700);
    return () => window.clearInterval(intervalId);
  }, [game, isSpectating, mode, refreshStatus]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isInputTarget(event.target)) {
        return;
      }
      if (mode !== "playing") {
        return;
      }
      if (isSpectating) {
        switch (event.key.toLowerCase()) {
          case "i":
            zoomCamera("out");
            break;
          case "o":
            zoomCamera("in");
            break;
          case "r":
            resetCamera();
            break;
          default:
            break;
        }
        return;
      }
      const presetIndex = Number(event.key) - 1;
      if (Number.isInteger(presetIndex) && presetIndex >= 0 && presetIndex < ORIENTATION_PRESETS.length) {
        applyPreset(ORIENTATION_PRESETS[presetIndex].id);
        return;
      }
      const step = event.shiftKey ? 0.08 : 0.04;
      switch (event.key.toLowerCase()) {
        case "w":
          nudgePosition([-step, 0, 0]);
          break;
        case "s":
          nudgePosition([step, 0, 0]);
          break;
        case "a":
          nudgePosition([0, -step, 0]);
          break;
        case "d":
          nudgePosition([0, step, 0]);
          break;
        case "i":
          zoomCamera("out");
          break;
        case "o":
          zoomCamera("in");
          break;
        case "r":
          resetCamera();
          break;
        case " ":
          event.preventDefault();
          if (preview?.is_valid) {
            void confirmPlacement();
          }
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [applyPreset, confirmPlacement, isSpectating, mode, nudgePosition, preview?.is_valid, resetCamera, zoomCamera]);

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-stone-200">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(252,211,130,0.28),transparent_28%),linear-gradient(180deg,rgba(249,244,236,0.22),rgba(149,139,125,0.3))]" />
      <Suspense fallback={<SceneFallback />}>
        <SceneCanvas />
      </Suspense>
      <div className="absolute inset-0 bg-gradient-to-b from-black/12 via-transparent to-black/18" />
      <GameHUD />
    </main>
  );
}
