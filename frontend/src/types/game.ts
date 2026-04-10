export type QuaternionWxyz = [number, number, number, number];
export type Vec3 = [number, number, number];

export type Truck = {
  depth: number;
  width: number;
  height: number;
};

export type BoxPayload = {
  id: string;
  dimensions: Vec3;
  weight?: number | null;
  position?: Vec3 | null;
  orientation_wxyz?: QuaternionWxyz | null;
};

export type GameState = {
  game_id: string;
  status: "ok" | "terminated";
  truck: Truck;
  placed_boxes: BoxPayload[];
  current_box: BoxPayload | null;
  boxes_remaining: number;
  density: number;
  game_status: "in_progress" | "completed" | "timed_out" | "no_feasible_placement";
  termination_reason: string | null;
  mode: "dev" | "compete";
  created_at: string;
  current_box_started_at: string | null;
  current_box_deadline: string | null;
  timeout_seconds: number;
  loading_guide_x: number | null;
};

export type PreviewResponse = {
  is_valid: boolean;
  message: string;
  category: string | null;
  details: Record<string, unknown>;
  support_ratio: number | null;
  normalized_position: Vec3 | null;
  normalized_orientation_wxyz: QuaternionWxyz | null;
  latest_valid_preview_action: {
    box_id: string;
    position: Vec3;
    orientation_wxyz: QuaternionWxyz;
  } | null;
  snap_suggestions: {
    floor?: {
      box_id: string;
      position: Vec3;
      orientation_wxyz: QuaternionWxyz;
    } | null;
    support_plane?: {
      box_id: string;
      position: Vec3;
      orientation_wxyz: QuaternionWxyz;
    } | null;
  };
  repair_suggestions: {
    support_aligned?: {
      box_id: string;
      position: Vec3;
      orientation_wxyz: QuaternionWxyz;
    } | null;
    nearby_valid?: {
      box_id: string;
      position: Vec3;
      orientation_wxyz: QuaternionWxyz;
    } | null;
    any_valid?: {
      box_id: string;
      position: Vec3;
      orientation_wxyz: QuaternionWxyz;
    } | null;
  };
  game_status: GameState["game_status"];
  termination_reason: string | null;
  current_box_deadline: string | null;
  density: number;
};

export type Pose = {
  position: Vec3;
  orientationWxyz: QuaternionWxyz;
};
