import * as THREE from "three";
import { ParticleSystem } from "./ParticleSystem";
import type { SwarmSystem } from "../system";

/**
 * CPU fallback backend — the untouched v3 reference sim wrapped in its own
 * three.js renderer/scene/camera. Lazy-loaded only when WebGPU is missing,
 * so three.js never reaches the GPU path's bundle.
 *
 * The ParticleSystem class already satisfies SwarmSystem (the interface was
 * shaped after it); this factory owns the rendering shell around it.
 */
export interface CpuApp {
  system: SwarmSystem;
  /** Render one frame (also used by `window.__render()` for pokes). */
  render: () => void;
  resize: (width: number, height: number, dpr: number) => void;
  dispose: () => void;
}

export function createCpuApp(container: HTMLElement, count: number): CpuApp {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    55,
    window.innerWidth / window.innerHeight,
    0.1,
    200,
  );
  camera.position.set(0, 0, 26);

  const system = new ParticleSystem(scene, { camera, count });

  return {
    system,
    render: () => renderer.render(scene, camera),
    resize: (width, height, dpr) => {
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(dpr, 2));
      system.setPixelRatio(dpr);
      system.updateRendererSize(width / 2, height / 2);
    },
    dispose: () => {
      system.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
