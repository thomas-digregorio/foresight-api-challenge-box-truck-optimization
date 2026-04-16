import { OrbitControls, Edges, Line } from "@react-three/drei";
import { Canvas, ThreeEvent, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import { CanvasTexture, MeshStandardMaterial, Quaternion, RepeatWrapping } from "three";
import { useGameStore } from "../../store/gameStore";
import type { BoxPayload } from "../../types/game";

const FALLBACK_TRUCK = { depth: 2, width: 2.6, height: 2.75 };
const EMPTY_PLACED_BOXES: BoxPayload[] = [];
const CAMERA_POSITION = [2.62, 1.3, 1.56] as const;
const CAMERA_TARGET = [0.7, 1.3, 0.38] as const;

let woodTextureCache: CanvasTexture | null | undefined;

function makeWoodTexture() {
  if (woodTextureCache !== undefined) {
    return woodTextureCache;
  }
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }
  context.fillStyle = "#c69769";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let index = 0; index < 48; index += 1) {
    const y = index * 12;
    context.fillStyle = index % 2 === 0 ? "rgba(130, 89, 54, 0.16)" : "rgba(255,255,255,0.06)";
    context.fillRect(0, y, canvas.width, 6);
  }
  for (let index = 0; index < 14; index += 1) {
    const x = index * 38 + (index % 2 ? 12 : 0);
    context.fillStyle = "rgba(101, 68, 41, 0.14)";
    context.fillRect(x, 0, 3, canvas.height);
  }
  const texture = new CanvasTexture(canvas);
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.repeat.set(2.4, 4);
  woodTextureCache = texture;
  return woodTextureCache;
}

function ResettableControls() {
  const controlsRef = useRef<any>(null);
  const game = useGameStore((state) => state.game);
  const cameraResetToken = useGameStore((state) => state.cameraResetToken);
  const cameraZoomToken = useGameStore((state) => state.cameraZoomToken);
  const cameraZoomDirection = useGameStore((state) => state.cameraZoomDirection);
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(CAMERA_POSITION[0], game?.truck.width ? game.truck.width / 2 : CAMERA_POSITION[1], CAMERA_POSITION[2]);
    camera.up.set(0, 0, 1);
    camera.lookAt(CAMERA_TARGET[0], game?.truck.width ? game.truck.width / 2 : CAMERA_TARGET[1], CAMERA_TARGET[2]);
    controlsRef.current?.target.set(CAMERA_TARGET[0], game?.truck.width ? game.truck.width / 2 : CAMERA_TARGET[1], CAMERA_TARGET[2]);
    controlsRef.current?.update();
  }, [camera, cameraResetToken, game?.game_id, game?.truck.width]);

  useEffect(() => {
    if (!controlsRef.current || cameraZoomDirection === null) {
      return;
    }
    const zoomFactor = 1.2;
    if (cameraZoomDirection === "in") {
      controlsRef.current.dollyIn?.(zoomFactor);
    } else {
      controlsRef.current.dollyOut?.(zoomFactor);
    }
    controlsRef.current.update();
  }, [cameraZoomDirection, cameraZoomToken]);

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      maxDistance={4.8}
      minDistance={0.8}
      minPolarAngle={0.52}
      maxPolarAngle={1.4}
      target={[CAMERA_TARGET[0], game?.truck.width ? game.truck.width / 2 : CAMERA_TARGET[1], CAMERA_TARGET[2]]}
    />
  );
}

function TruckInterior() {
  const game = useGameStore((state) => state.game);
  const isSpectating = useGameStore((state) => state.isSpectating);
  const truck = game?.truck ?? FALLBACK_TRUCK;
  const loadingGuideX = game?.loading_guide_x ?? null;
  const setPosition = useGameStore((state) => state.setPosition);
  const confirmPlacement = useGameStore((state) => state.confirmPlacement);
  const preview = useGameStore((state) => state.preview);
  const floorTexture = makeWoodTexture();
  const floorMaterial = new MeshStandardMaterial({
    color: "#c89a68",
    roughness: 0.78,
    metalness: 0.02,
    map: floorTexture ?? undefined,
  });
  const wallMaterial = new MeshStandardMaterial({ color: "#f2eee8", roughness: 0.92, metalness: 0.02 });

  const onPointerMove = (event: ThreeEvent<PointerEvent>) => {
    if (isSpectating || !game?.current_box || game.game_status !== "in_progress") {
      return;
    }
    setPosition({ x: event.point.x, y: event.point.y });
  };

  const onDoubleClick = async () => {
    if (!isSpectating && preview?.is_valid) {
      await confirmPlacement();
    }
  };

  return (
    <group>
      <ambientLight intensity={1.3} />
      <hemisphereLight intensity={0.6} color="#ffffff" groundColor="#c1a07a" />
      <directionalLight position={[2.5, 1.2, 3.8]} intensity={2.4} castShadow />
      <mesh position={[truck.depth / 2, truck.width / 2, 0]} receiveShadow onPointerMove={onPointerMove} onDoubleClick={onDoubleClick}>
        <planeGeometry args={[truck.depth, truck.width]} />
        <primitive object={floorMaterial} attach="material" />
      </mesh>
      {loadingGuideX !== null ? (
        <Line
          points={[
            [loadingGuideX, 0.08, 0.012],
            [loadingGuideX, truck.width - 0.08, 0.012],
          ]}
          color="#ffffff"
          transparent
          opacity={0.55}
          lineWidth={2.5}
        />
      ) : null}
      <mesh position={[0, truck.width / 2, truck.height / 2]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[truck.height, truck.width]} />
        <primitive object={wallMaterial} attach="material" />
      </mesh>
      <mesh position={[truck.depth / 2, 0, truck.height / 2]} rotation={[Math.PI / 2, 0, Math.PI / 2]}>
        <planeGeometry args={[truck.depth, truck.height]} />
        <primitive object={wallMaterial} attach="material" />
      </mesh>
      <mesh position={[truck.depth / 2, truck.width, truck.height / 2]} rotation={[-Math.PI / 2, 0, Math.PI / 2]}>
        <planeGeometry args={[truck.depth, truck.height]} />
        <primitive object={wallMaterial} attach="material" />
      </mesh>
      <mesh position={[truck.depth / 2, truck.width / 2, truck.height]} rotation={[Math.PI, 0, 0]}>
        <planeGeometry args={[truck.depth, truck.width]} />
        <meshStandardMaterial color="#f7f4ee" roughness={0.85} />
      </mesh>
      {Array.from({ length: 5 }).map((_, index) => (
        <mesh key={`left-rail-${index}`} position={[0.25 + index * 0.36, 0.02, truck.height / 2]}>
          <boxGeometry args={[0.02, 0.04, truck.height - 0.3]} />
          <meshStandardMaterial color="#87909c" metalness={0.5} roughness={0.32} />
        </mesh>
      ))}
      {Array.from({ length: 5 }).map((_, index) => (
        <mesh key={`right-rail-${index}`} position={[0.25 + index * 0.36, truck.width - 0.02, truck.height / 2]}>
          <boxGeometry args={[0.02, 0.04, truck.height - 0.3]} />
          <meshStandardMaterial color="#87909c" metalness={0.5} roughness={0.32} />
        </mesh>
      ))}
      {Array.from({ length: 3 }).map((_, index) => (
        <mesh key={`front-guard-${index}`} position={[0.02, truck.width / 2, 0.18 + index * 0.08]}>
          <boxGeometry args={[0.02, truck.width - 0.2, 0.03]} />
          <meshStandardMaterial color="#8d939b" metalness={0.45} roughness={0.38} />
        </mesh>
      ))}
    </group>
  );
}

function PlacedBoxes() {
  const placedBoxes = useGameStore((state) => state.game?.placed_boxes ?? EMPTY_PLACED_BOXES);
  return (
    <>
      {placedBoxes.map((box) => {
        const quaternion = new Quaternion(
          box.orientation_wxyz?.[1] ?? 0,
          box.orientation_wxyz?.[2] ?? 0,
          box.orientation_wxyz?.[3] ?? 0,
          box.orientation_wxyz?.[0] ?? 1,
        );
        return (
          <mesh key={box.id} position={box.position ?? [0, 0, 0]} quaternion={quaternion} castShadow receiveShadow>
            <boxGeometry args={box.dimensions} />
            <meshStandardMaterial color="#c97f25" roughness={0.7} metalness={0.04} />
            <Edges color="#f5ca91" />
          </mesh>
        );
      })}
    </>
  );
}

function PreviewBox() {
  const game = useGameStore((state) => state.game);
  const isSpectating = useGameStore((state) => state.isSpectating);
  const pose = useGameStore((state) => state.pose);
  const preview = useGameStore((state) => state.preview);

  if (isSpectating || !game?.current_box || game.game_status !== "in_progress") {
    return null;
  }
  const quaternion = new Quaternion(pose.orientationWxyz[1], pose.orientationWxyz[2], pose.orientationWxyz[3], pose.orientationWxyz[0]);
  return (
    <group>
      <mesh position={pose.position} quaternion={quaternion}>
        <boxGeometry args={game.current_box.dimensions} />
        <meshStandardMaterial
          color={preview?.is_valid ? "#8bf3df" : "#ff6f61"}
          transparent
          opacity={preview?.is_valid ? 0.34 : 0.28}
          roughness={0.2}
          metalness={0.08}
        />
        <Edges color={preview?.is_valid ? "#b9fff4" : "#ffd0cb"} />
      </mesh>
      {preview?.latest_valid_preview_action ? (
        <mesh
          position={preview.latest_valid_preview_action.position}
          quaternion={
            new Quaternion(
              preview.latest_valid_preview_action.orientation_wxyz[1],
              preview.latest_valid_preview_action.orientation_wxyz[2],
              preview.latest_valid_preview_action.orientation_wxyz[3],
              preview.latest_valid_preview_action.orientation_wxyz[0],
            )
          }
        >
          <boxGeometry args={game.current_box.dimensions} />
          <meshStandardMaterial color="#d5c66c" transparent opacity={0.12} />
        </mesh>
      ) : null}
      <Line
        points={[
          [pose.position[0], 0, 0.01],
          [pose.position[0], game.truck.width, 0.01],
        ]}
        color="#8bf3df"
        transparent
        opacity={0.58}
        lineWidth={1}
      />
    </group>
  );
}

export function SceneCanvas() {
  return (
    <div className="absolute inset-0">
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [CAMERA_POSITION[0], CAMERA_POSITION[1], CAMERA_POSITION[2]], fov: 48, near: 0.01, far: 20 }}
        onCreated={({ camera, gl }) => {
          camera.up.set(0, 0, 1);
          camera.lookAt(CAMERA_TARGET[0], CAMERA_TARGET[1], CAMERA_TARGET[2]);
          gl.setClearColor("#e7ddd0");
        }}
      >
        <fog attach="fog" args={["#e7ddd0", 2.2, 8]} />
        <TruckInterior />
        <PlacedBoxes />
        <PreviewBox />
        <ResettableControls />
      </Canvas>
    </div>
  );
}
