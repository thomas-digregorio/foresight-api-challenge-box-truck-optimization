import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Quaternion } from "three";
import type { QuaternionWxyz, Vec3 } from "../types/game";

type MiniBoxPreviewProps = {
  dimensions: Vec3;
  quaternion?: QuaternionWxyz;
};

export function MiniBoxPreview({ dimensions, quaternion = [1, 0, 0, 0] }: MiniBoxPreviewProps) {
  const q = new Quaternion(quaternion[1], quaternion[2], quaternion[3], quaternion[0]);
  return (
    <div className="h-40 w-full overflow-hidden rounded-[1.4rem] bg-slate-900/85">
      <Canvas
        camera={{ position: [1.9, 1.9, 1.4], fov: 42 }}
        onCreated={({ camera }) => {
          camera.up.set(0, 0, 1);
          camera.lookAt(0, 0, 0);
        }}
      >
        <ambientLight intensity={1.6} />
        <directionalLight position={[2, 1.5, 2.5]} intensity={2.2} />
        <mesh rotation-x={0} position={[0, 0, -0.55]}>
          <circleGeometry args={[1.7, 40]} />
          <meshStandardMaterial color="#121a26" />
        </mesh>
        <mesh quaternion={q}>
          <boxGeometry args={dimensions} />
          <meshStandardMaterial color="#d09331" roughness={0.58} metalness={0.08} />
        </mesh>
        <OrbitControls enablePan={false} enableZoom={false} />
      </Canvas>
    </div>
  );
}

