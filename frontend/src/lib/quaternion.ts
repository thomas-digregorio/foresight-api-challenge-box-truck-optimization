import { Euler, Quaternion } from "three";
import type { QuaternionWxyz } from "../types/game";

export function normalizeQuaternionWxyz(quaternion: QuaternionWxyz): QuaternionWxyz {
  const length = Math.hypot(...quaternion);
  if (!length) {
    return [1, 0, 0, 0];
  }
  return quaternion.map((value) => value / length) as QuaternionWxyz;
}

export function eulerDegreesToQuaternionWxyz(roll: number, pitch: number, yaw: number): QuaternionWxyz {
  const radians = [roll, pitch, yaw].map((value) => (value * Math.PI) / 180);
  const euler = new Euler(radians[0], radians[1], radians[2], "XYZ");
  const quaternion = new Quaternion().setFromEuler(euler);
  return normalizeQuaternionWxyz([quaternion.w, quaternion.x, quaternion.y, quaternion.z]);
}

export function quaternionWxyzToEulerDegrees(quaternionWxyz: QuaternionWxyz): [number, number, number] {
  const normalized = normalizeQuaternionWxyz(quaternionWxyz);
  const quaternion = new Quaternion(normalized[1], normalized[2], normalized[3], normalized[0]);
  const euler = new Euler().setFromQuaternion(quaternion, "XYZ");
  return [euler.x, euler.y, euler.z].map((value) => Math.round((value * 180) / Math.PI)) as [number, number, number];
}

export const ORIENTATION_PRESETS: { id: string; label: string; quaternion: QuaternionWxyz }[] = [
  { id: "flat-z", label: "Z Up", quaternion: eulerDegreesToQuaternionWxyz(0, 0, 0) },
  { id: "x-up", label: "X Up", quaternion: eulerDegreesToQuaternionWxyz(0, 90, 0) },
  { id: "y-up", label: "Y Up", quaternion: eulerDegreesToQuaternionWxyz(-90, 0, 0) },
  { id: "z-flip", label: "Z Flip", quaternion: eulerDegreesToQuaternionWxyz(180, 0, 0) },
  { id: "x-flip", label: "X Flip", quaternion: eulerDegreesToQuaternionWxyz(0, -90, 0) },
  { id: "y-flip", label: "Y Flip", quaternion: eulerDegreesToQuaternionWxyz(90, 0, 0) },
];

