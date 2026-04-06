import { Suspense, lazy, useEffect, useState } from "react";
import { ORIENTATION_PRESETS, quaternionWxyzToEulerDegrees } from "../../lib/quaternion";
import { useGameStore } from "../../store/gameStore";

const MiniBoxPreview = lazy(async () => {
  const module = await import("../../components/MiniBoxPreview");
  return { default: module.MiniBoxPreview };
});

function formatDimensions(dimensions?: [number, number, number]) {
  if (!dimensions) {
    return "0.0 x 0.0 x 0.0 m";
  }
  return dimensions.map((value) => value.toFixed(2)).join(" x ") + " m";
}

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-full border border-white/8 bg-white/6 px-3 py-1 text-xs uppercase tracking-[0.2em] text-white/70">
      <span className="mr-2 text-white/45">{label}</span>
      <span className="text-white">{value}</span>
    </div>
  );
}

function PreviewFallback() {
  return <div className="h-40 w-full rounded-[1.4rem] bg-slate-900/70" />;
}

function presetHotkeyLabel(index: number) {
  return `${index + 1}`;
}

export function GameHUD() {
  const game = useGameStore((state) => state.game);
  const preview = useGameStore((state) => state.preview);
  const pose = useGameStore((state) => state.pose);
  const mode = useGameStore((state) => state.mode);
  const error = useGameStore((state) => state.error);
  const showAdvanced = useGameStore((state) => state.showAdvanced);
  const isStarting = useGameStore((state) => state.isStarting);
  const isPlacing = useGameStore((state) => state.isPlacing);
  const isPreviewSyncing = useGameStore((state) => state.isPreviewSyncing);
  const poseVersion = useGameStore((state) => state.poseVersion);
  const previewPoseVersion = useGameStore((state) => state.previewPoseVersion);
  const startNewGame = useGameStore((state) => state.startNewGame);
  const confirmPlacement = useGameStore((state) => state.confirmPlacement);
  const stopCurrentGame = useGameStore((state) => state.stopCurrentGame);
  const setPosition = useGameStore((state) => state.setPosition);
  const setQuaternion = useGameStore((state) => state.setQuaternion);
  const setEuler = useGameStore((state) => state.setEuler);
  const applyPreset = useGameStore((state) => state.applyPreset);
  const toggleAdvanced = useGameStore((state) => state.toggleAdvanced);
  const resetCamera = useGameStore((state) => state.resetCamera);
  const zoomCamera = useGameStore((state) => state.zoomCamera);
  const [nowMs, setNowMs] = useState(Date.now());

  useEffect(() => {
    const intervalId = window.setInterval(() => setNowMs(Date.now()), 100);
    return () => window.clearInterval(intervalId);
  }, []);

  const euler = quaternionWxyzToEulerDegrees(pose.orientationWxyz);
  const deadline = game?.current_box_deadline ? new Date(game.current_box_deadline).getTime() : null;
  const remainingSeconds = deadline ? Math.max(0, (deadline - nowMs) / 1000) : 0;
  const percentage = game ? Math.max(0, Math.min(100, game.density * 100)) : 0;
  const previewIsFresh = previewPoseVersion === poseVersion;
  const canConfirmPlacement = Boolean(preview?.is_valid) && previewIsFresh && !isPreviewSyncing && mode === "playing" && !isPlacing;
  const previewTone = isPreviewSyncing
    ? "bg-slate-100/10 text-white/80"
    : canConfirmPlacement
      ? "bg-emerald-400/14 text-emerald-100"
      : preview
        ? "bg-rose-400/14 text-rose-100"
        : "bg-slate-100/10 text-white/80";
  const previewMessage = isPreviewSyncing
    ? "Syncing preview with the engine..."
    : preview?.message ?? "Preview a pose to start the timeout fallback buffer.";

  return (
    <>
      <div className="pointer-events-none absolute inset-0">
        <div className="pointer-events-auto absolute left-5 top-5 w-[18rem] rounded-[1.7rem] p-5 text-white shadow-panel panel-glass">
          <div className="text-3xl font-semibold tracking-tight">{percentage.toFixed(1)}%</div>
          <div className="mt-1 text-sm uppercase tracking-[0.28em] text-white/60">Density</div>
          <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/10">
            <div className="range-meter h-full rounded-full" style={{ width: `${Math.max(6, percentage)}%` }} />
          </div>
          <div className="mt-4 flex items-center justify-between text-sm text-white/80">
            <span>Boxes placed</span>
            <span className="font-medium">{game?.placed_boxes.length ?? 0}</span>
          </div>
          <div className="mt-2 flex items-center justify-between text-sm text-white/65">
            <span>Queue remaining</span>
            <span>{game?.boxes_remaining ?? 0}</span>
          </div>
        </div>

        <div className="pointer-events-auto absolute right-5 top-5 w-[21rem] rounded-[1.9rem] p-5 text-white shadow-panel panel-glass">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-xs uppercase tracking-[0.3em] text-white/55">Next</div>
              <h2 className="mt-2 text-2xl font-semibold">Current Box</h2>
            </div>
            <StatPill label="Timer" value={`${remainingSeconds.toFixed(1)}s`} />
          </div>
          <div className="mt-4">
            <Suspense fallback={<PreviewFallback />}>
              <MiniBoxPreview dimensions={game?.current_box?.dimensions ?? [0.4, 0.4, 0.4]} quaternion={pose.orientationWxyz} />
            </Suspense>
          </div>
          <div className="mt-4 rounded-[1.4rem] bg-white/6 p-4">
            <div className="text-xs uppercase tracking-[0.28em] text-white/55">Dimensions</div>
            <div className="mt-2 text-lg font-medium">{formatDimensions(game?.current_box?.dimensions as [number, number, number] | undefined)}</div>
            <div className="mt-2 text-sm text-white/65">
              Weight {game?.current_box?.weight?.toFixed(1) ?? "--"} kg
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              className="rounded-full bg-amber-400 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:bg-amber-400/40"
              disabled={!canConfirmPlacement}
              onClick={() => void confirmPlacement()}
            >
              {isPlacing ? "Placing..." : isPreviewSyncing ? "Syncing..." : "Confirm Placement"}
            </button>
            <button className="rounded-full border border-white/15 px-4 py-2 text-sm text-white/75" onClick={() => void stopCurrentGame()}>
              Stop
            </button>
            <button className="rounded-full border border-white/15 px-4 py-2 text-sm text-white/75" onClick={() => zoomCamera("out")}>
              Zoom Out (I)
            </button>
            <button className="rounded-full border border-white/15 px-4 py-2 text-sm text-white/75" onClick={() => zoomCamera("in")}>
              Zoom In (O)
            </button>
            <button className="rounded-full border border-white/15 px-4 py-2 text-sm text-white/75" onClick={resetCamera}>
              Reset Camera
            </button>
          </div>
          {error ? <div className="mt-3 rounded-2xl bg-red-400/12 px-4 py-3 text-sm text-red-100">{error}</div> : null}
        </div>

        <div className="pointer-events-auto absolute bottom-6 left-1/2 w-[min(70rem,calc(100%-2rem))] -translate-x-1/2 rounded-[2rem] px-6 py-5 text-white shadow-panel panel-glass">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap gap-2">
              <StatPill label="Move" value="W/A/S/D" />
              <StatPill label="Rotate" value="1 / 2 / 3 / 4 / 5 / 6" />
              <StatPill label="Drop" value="Space" />
              <StatPill label="Zoom" value="I / O" />
              <StatPill label="View" value="R Reset" />
            </div>
            <div className={`rounded-full px-4 py-2 text-sm ${previewTone}`}>
              {previewMessage}
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <button className="rounded-full border border-white/15 px-4 py-2 text-sm" onClick={toggleAdvanced}>
              {showAdvanced ? "Hide Advanced" : "Advanced"}
            </button>
            {ORIENTATION_PRESETS.map((preset, index) => (
              <button key={preset.id} className="rounded-full border border-white/10 px-3 py-2 text-sm text-white/75" onClick={() => applyPreset(preset.id)}>
                {presetHotkeyLabel(index)} {preset.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {mode === "idle" || mode === "finished" ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-6">
          <div className="pointer-events-auto max-w-3xl rounded-[2.2rem] border border-white/20 bg-[linear-gradient(180deg,rgba(10,16,25,0.78),rgba(10,16,25,0.6))] p-8 text-white shadow-panel backdrop-blur-xl">
            <div className="text-xs uppercase tracking-[0.35em] text-amber-200/80">Foresight Local Challenge</div>
            <h1 className="mt-4 max-w-2xl text-4xl font-semibold tracking-tight text-white">
              Deterministic truck loading sandbox with challenge-shaped actions and observations.
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-white/72">
              Manual play, timeout auto-placement from the latest valid preview, and an engine that stays compatible with later agent and RL work.
            </p>
            {mode === "finished" && game ? (
              <div className="mt-6 flex flex-wrap gap-3">
                <StatPill label="Final Density" value={`${percentage.toFixed(1)}%`} />
                <StatPill label="Placed" value={`${game.placed_boxes.length}`} />
                <StatPill label="End State" value={game.game_status} />
              </div>
            ) : null}
            <div className="mt-8 flex justify-center">
              <button
                className="rounded-full bg-amber-400 px-10 py-5 text-lg font-semibold text-slate-950 shadow-[0_24px_60px_rgba(184,118,20,0.35)] transition hover:-translate-y-0.5 hover:bg-amber-300"
                onClick={() => void startNewGame()}
              >
                {isStarting ? "Starting..." : "Start New Game"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showAdvanced ? (
        <div className="pointer-events-auto absolute bottom-40 left-6 w-[26rem] rounded-[1.9rem] p-5 text-white shadow-panel panel-glass">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Advanced Pose</h3>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3">
            {(["x", "y", "z"] as const).map((axis, index) => (
              <label key={axis} className="text-sm text-white/70">
                {axis.toUpperCase()}
                <input
                  className="mt-1 w-full rounded-2xl border border-white/10 bg-white/6 px-3 py-2 text-white"
                  type="number"
                  step="0.01"
                  value={pose.position[index].toFixed(2)}
                  onChange={(event) => setPosition({ [axis]: Number(event.target.value) })}
                />
              </label>
            ))}
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3">
            {(["roll", "pitch", "yaw"] as const).map((axis, index) => (
              <label key={axis} className="text-sm text-white/70">
                {axis}
                <input
                  className="mt-1 w-full rounded-2xl border border-white/10 bg-white/6 px-3 py-2 text-white"
                  type="number"
                  step="1"
                  value={euler[index]}
                  onChange={(event) => setEuler({ [axis]: Number(event.target.value) })}
                />
              </label>
            ))}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            {pose.orientationWxyz.map((value, index) => (
              <label key={index} className="text-sm text-white/70">
                {["w", "x", "y", "z"][index]}
                <input
                  className="mt-1 w-full rounded-2xl border border-white/10 bg-white/6 px-3 py-2 text-white"
                  type="number"
                  step="0.001"
                  value={value.toFixed(3)}
                  onChange={(event) => {
                    const next = [...pose.orientationWxyz] as [number, number, number, number];
                    next[index] = Number(event.target.value);
                    setQuaternion(next);
                  }}
                />
              </label>
            ))}
          </div>
          <div className="mt-4 rounded-[1.4rem] bg-white/6 p-4 text-sm text-white/70">
            <div className="font-medium text-white">Backend validity</div>
            <div className="mt-2">{preview?.message ?? "No preview synced yet."}</div>
            {preview && preview.support_ratio !== null ? <div className="mt-2">Support ratio {(preview.support_ratio * 100).toFixed(1)}%</div> : null}
            {preview?.category ? <div className="mt-2">Category {preview.category}</div> : null}
            {game?.termination_reason ? <div className="mt-2">Termination {game.termination_reason}</div> : null}
          </div>
        </div>
      ) : null}
    </>
  );
}
