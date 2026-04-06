import { describe, expect, it } from "vitest";
import { eulerDegreesToQuaternionWxyz, normalizeQuaternionWxyz } from "./quaternion";

describe("quaternion helpers", () => {
  it("normalizes a wxyz quaternion", () => {
    expect(normalizeQuaternionWxyz([2, 0, 0, 0])).toEqual([1, 0, 0, 0]);
  });

  it("keeps identity euler as identity quaternion", () => {
    expect(eulerDegreesToQuaternionWxyz(0, 0, 0)).toEqual([1, 0, 0, 0]);
  });
});
