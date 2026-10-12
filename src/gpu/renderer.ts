import { GPU_TUNING, PALETTES, STYLE_IDS } from "../params";
import type { GpuSwarm } from "./GpuSwarm";
import { WGSL_SPRITES, WGSL_TONEMAP } from "./wgsl-render";

/**
 * Renders the swarm straight out of the sim's storage buffers — zero
 * per-frame CPU↔GPU traffic beyond one 144-byte uniform write.
 *
 * Two passes: instanced-quad sprites accumulate additively into an HDR
 * rgba16float target (the reference's exact sprite math), then a
 * hue-preserving shoulder tonemap resolves to the canvas with alpha 0 —
 * with a premultiplied canvas that composites as pure addition over the
 * page's CSS gradient, matching how the WebGL version sat on the page.
 */
export class GpuRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private canvas: HTMLCanvasElement;

  private spritePipeline!: GPURenderPipeline;
  private tonemapPipeline!: GPURenderPipeline;
  private uniformsB: GPUBuffer;
  private uniforms = new Float32Array(36);

  private hdr: GPUTexture | null = null;
  private hdrView!: GPUTextureView;
  private tonemapGroup!: GPUBindGroup;
  private spriteGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private boundGeneration = -1;

  private dpr = 1;

  /** Crowd-dim reference count, one-pole smoothed (see render()). */
  private nRef = -1;
  private lastRenderMs = -1;

  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    private system: GpuSwarm,
  ) {
    this.device = device;
    this.canvas = canvas;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context = canvas.getContext("webgpu") as GPUCanvasContext;
    this.context.configure({
      device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    this.uniformsB = device.createBuffer({
      label: "render-uniforms",
      size: this.uniforms.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.createPipelines();
    this.resize(canvas.clientWidth || 1, canvas.clientHeight || 1, window.devicePixelRatio || 1);
  }

  private createPipelines(): void {
    const dev = this.device;
    const spriteModule = dev.createShaderModule({
      label: "sprites",
      code: WGSL_SPRITES,
    });
    spriteModule.getCompilationInfo().then((info) => {
      for (const m of info.messages) {
        if (m.type === "error")
          console.error(`[wgsl:sprites] ${m.lineNum}:${m.linePos} ${m.message}`);
      }
    });
    this.spritePipeline = dev.createRenderPipeline({
      label: "sprites",
      layout: "auto",
      vertex: { module: spriteModule, entryPoint: "vsSprite" },
      fragment: {
        module: spriteModule,
        entryPoint: "fsSprite",
        targets: [
          {
            format: "rgba16float",
            blend: {
              // Match THREE.AdditiveBlending with non-premultiplied output:
              // gl.blendFunc(SRC_ALPHA, ONE) on all channels.
              color: { srcFactor: "src-alpha", dstFactor: "one" },
              alpha: { srcFactor: "src-alpha", dstFactor: "one" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-strip" },
    });

    const tonemapModule = dev.createShaderModule({
      label: "tonemap",
      code: WGSL_TONEMAP,
    });
    tonemapModule.getCompilationInfo().then((info) => {
      for (const m of info.messages) {
        if (m.type === "error")
          console.error(`[wgsl:tonemap] ${m.lineNum}:${m.linePos} ${m.message}`);
      }
    });
    this.tonemapPipeline = dev.createRenderPipeline({
      label: "tonemap",
      layout: "auto",
      vertex: { module: tonemapModule, entryPoint: "vsFull" },
      fragment: {
        module: tonemapModule,
        entryPoint: "fsTonemap",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  resize(cssWidth: number, cssHeight: number, dpr: number): void {
    this.dpr = Math.min(dpr, 2);
    const w = Math.max(1, Math.floor(cssWidth * this.dpr));
    const h = Math.max(1, Math.floor(cssHeight * this.dpr));
    this.canvas.width = w;
    this.canvas.height = h;
    this.canvas.style.width = `${cssWidth}px`;
    this.canvas.style.height = `${cssHeight}px`;

    this.hdr?.destroy();
    this.hdr = this.device.createTexture({
      label: "hdr",
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrView = this.hdr.createView();
    this.tonemapGroup = this.device.createBindGroup({
      layout: this.tonemapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformsB } },
        { binding: 1, resource: this.hdrView },
      ],
    });
  }

  private syncSpriteGroups(): void {
    if (this.boundGeneration === this.system.generation) return;
    const layout = this.spritePipeline.getBindGroupLayout(0);
    const [a, b] = this.system.dynBuffers();
    const make = (dyn: GPUBuffer) =>
      this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformsB } },
          { binding: 1, resource: { buffer: dyn } },
          { binding: 2, resource: { buffer: this.system.selfBuffer() } },
          { binding: 3, resource: { buffer: this.system.staticsBuffer() } },
          // Re-created on rebuild like the others; the generation check covers it.
          { binding: 4, resource: { buffer: this.system.gridDataBuffer() } },
        ],
      });
    this.spriteGroups = [make(a), make(b)];
    this.boundGeneration = this.system.generation;
  }

  render(): void {
    if (!this.hdr) return;
    const sys = this.system;
    this.syncSpriteGroups();

    // Density-compensated brightness budget (HANDOFF §5): the tuned params
    // describe the look at referenceCount; size²·opacity scales down as the
    // count scales up, so the school reads equally bright at any density.
    const P = sys.params;
    const comp = GPU_TUNING.referenceCount / Math.max(64, sys.getCount());
    const sizeComp = Math.min(GPU_TUNING.maxComp, Math.pow(comp, GPU_TUNING.sizeCompExp));
    const opComp = Math.min(GPU_TUNING.maxComp, Math.pow(comp, GPU_TUNING.opacityCompExp));

    const R = this.uniforms;
    R.set(sys.camera.viewProj, 0);
    R[16] = this.canvas.width;
    R[17] = this.canvas.height;
    R[18] = this.dpr;
    R[19] = sys.simTime;
    R[20] = P.particleSize * sizeComp;
    R[21] = P.opacity * opComp;
    R[22] = P.flashBrightness;
    R[23] = P.twinkle;
    R[24] = P.stretch;
    R[25] = P.motionGlow;
    R[26] = STYLE_IDS[sys.styleName];
    R[27] = GPU_TUNING.tonemapKnee;
    const flash = PALETTES[sys.paletteName].flash;
    const v = parseInt(flash.slice(1), 16);
    R[28] = ((v >> 16) & 255) / 255;
    R[29] = ((v >> 8) & 255) / 255;
    R[30] = (v & 255) / 255;
    R[31] = sys.camera.z;
    // Crowd dim reference: the fine-grid 27-cell count a fish sees at the
    // school's own *live* mean density — count spread over the uniform ball
    // with the school's measured RMS radius (R = rms·√(5/3)). nRef is the
    // school's own mean density, so a contracted school keeps its brightness
    // and only local excess (pile-ups above that mean) is dimmed. A nominal
    // tank-sized ball dimmed the whole body whenever the director's contract
    // mood condensed it. One-pole smoothed (τ = 0.5 s) so the readback's
    // ~2-frame staleness and noise can't flicker the brightness.
    const nowMs = performance.now();
    const dt =
      this.lastRenderMs < 0 ? 1 / 60 : Math.min(0.25, Math.max(0, (nowMs - this.lastRenderMs) / 1000));
    this.lastRenderMs = nowMs;
    const schoolR = Math.max(1, sys.schoolRmsRadius() * Math.sqrt(5 / 3));
    const schoolVol = (4 / 3) * Math.PI * schoolR ** 3;
    const nRefNow = (sys.getCount() / schoolVol) * 27 * GPU_TUNING.fineCellSize ** 3;
    this.nRef = this.nRef < 0 ? nRefNow : this.nRef + (nRefNow - this.nRef) * (1 - Math.exp(-dt / 0.5));
    R[32] = this.nRef;
    R[33] = P.crowdDim ?? 0;
    this.device.queue.writeBuffer(this.uniformsB, 0, R);

    const enc = this.device.createCommandEncoder({ label: "render" });
    const spritePass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.hdrView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    spritePass.setPipeline(this.spritePipeline);
    spritePass.setBindGroup(0, this.spriteGroups![sys.currentParity()]);
    spritePass.draw(4, sys.getCount());
    spritePass.end();

    const canvasView = this.context.getCurrentTexture().createView();
    const tonemapPass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: canvasView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    tonemapPass.setPipeline(this.tonemapPipeline);
    tonemapPass.setBindGroup(0, this.tonemapGroup);
    tonemapPass.draw(3);
    tonemapPass.end();

    this.device.queue.submit([enc.finish()]);
  }

  dispose(): void {
    this.hdr?.destroy();
    this.uniformsB.destroy();
    this.context.unconfigure();
    this.canvas.remove();
  }
}
