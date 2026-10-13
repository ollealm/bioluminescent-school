import * as THREE from "three";
import { GPU_TUNING } from "../params";
import { ParticleSystem } from "./ParticleSystem";
import type { SwarmSystem } from "../system";

/**
 * CPU fallback backend — the v3 engine (carrying the GPU fork's levers)
 * wrapped in its own three.js renderer/scene/camera. Lazy-loaded only when WebGPU is missing,
 * so three.js never reaches the GPU path's bundle.
 *
 * The ParticleSystem class already satisfies SwarmSystem (the interface was
 * shaped after it); this factory owns the rendering shell around it.
 *
 * Rendering matches the GPU's two passes (src/gpu/renderer.ts): the points
 * accumulate additively into a half-float target, then a hue-preserving
 * shoulder resolves them onto the canvas, so a pile-up saturates as the
 * palette's color instead of clipping to white. Without a renderable
 * half-float target the points draw straight to the canvas, as v3 did.
 */
export interface CpuApp {
  system: SwarmSystem;
  /** Render one frame (also used by `window.__render()` for pokes). */
  render: () => void;
  resize: (width: number, height: number, dpr: number) => void;
  dispose: () => void;
  /** Opacity multiplier for a school the count governor has thinned. */
  setOpacityScale: (k: number) => void;
}

// The GPU's fsTonemap (src/gpu/wgsl-render.ts), line for line.
const RESOLVE_VERTEX = /* glsl */ `
  void main() {
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;
const RESOLVE_FRAGMENT = /* glsl */ `
  uniform sampler2D tHdr;
  uniform float uKnee;

  void main() {
    vec4 t = texelFetch(tHdr, ivec2(gl_FragCoord.xy), 0);
    vec3 c = t.rgb;
    float m = max(c.r, max(c.g, c.b));
    float scale = 1.0;
    if (m > uKnee) {
      // Soft shoulder, asymptotic to 1: identity below the knee, compressed
      // above, applied to the max channel so hue ratios are preserved.
      float span = 1.0 - uKnee;
      float mapped = uKnee + span * (1.0 - exp(-(m - uKnee) / span));
      scale = mapped / m;
    }
    // Same composite as the direct path and the GPU: rgb + page·(1−a) on
    // a premultiplied canvas. rgb may exceed a (brightness boosts).
    gl_FragColor = vec4(c * scale, clamp(t.a, 0.0, 1.0));
  }
`;

export function createCpuApp(
  container: HTMLElement,
  count: number,
  capacity = count,
): CpuApp {
  const renderer = new THREE.WebGLRenderer({
    // MSAA does nothing for point sprites (a frozen frame renders
    // pixel-identical without it) and tripled the render cost: 3.0 ms vs
    // 0.96 ms at 2880×1800 on M2 Max. The CPU path runs on the weakest
    // devices, so that third is worth keeping.
    antialias: false,
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

  const system = new ParticleSystem(scene, { camera, count, capacity });

  // WebGL2 renders to (and blends into) RGBA16F with either extension.
  const hdrOk =
    renderer.extensions.has("EXT_color_buffer_float") ||
    renderer.extensions.has("EXT_color_buffer_half_float");
  const drawingSize = new THREE.Vector2();
  let hdr: THREE.WebGLRenderTarget | null = null;
  let resolve: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial> | null = null;
  const resolveScene = new THREE.Scene();
  if (hdrOk) {
    renderer.getDrawingBufferSize(drawingSize);
    // A hidden tab can boot at 0×0; the first real resize sizes it.
    hdr = new THREE.WebGLRenderTarget(Math.max(1, drawingSize.x), Math.max(1, drawingSize.y), {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      generateMipmaps: false,
      depthBuffer: false,
      stencilBuffer: false,
    });
    const resolveMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tHdr: { value: hdr.texture },
        uKnee: { value: GPU_TUNING.tonemapKnee },
      },
      vertexShader: RESOLVE_VERTEX,
      fragmentShader: RESOLVE_FRAGMENT,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
    });
    // One oversized triangle covering the screen.
    const tri = new THREE.BufferGeometry();
    tri.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
    );
    resolve = new THREE.Mesh(tri, resolveMaterial);
    resolve.frustumCulled = false;
    resolveScene.add(resolve);
  }

  return {
    system,
    render: () => {
      if (!hdr) {
        renderer.render(scene, camera);
        return;
      }
      renderer.setRenderTarget(hdr);
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      renderer.render(resolveScene, camera);
    },
    resize: (width, height, dpr) => {
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(dpr, 2));
      system.setPixelRatio(dpr);
      system.updateRendererSize(width / 2, height / 2);
      if (hdr) {
        renderer.getDrawingBufferSize(drawingSize);
        hdr.setSize(Math.max(1, drawingSize.x), Math.max(1, drawingSize.y));
      }
    },
    dispose: () => {
      system.dispose();
      hdr?.dispose();
      resolve?.geometry.dispose();
      resolve?.material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
    setOpacityScale: (k) => system.setOpacityScale(k),
  };
}
