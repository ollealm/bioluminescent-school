import { GPU_TUNING, PALETTES, STYLE_IDS } from "../params";
import type { GpuSwarm } from "./GpuSwarm";
import { WGSL_SPRITES, WGSL_TONEMAP } from "./wgsl-render";

/**
 * Renders the swarm straight out of the sim's storage buffers — zero
 * per-frame CPU↔GPU traffic beyond one 128-byte uniform write.
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
  private uniforms = new Float32Array(32);

  private hdr: GPUTexture | null = null;
  private hdrView!: GPUTextureView;
  private tonemapGroup!: GPUBindGroup;
  private spriteGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private boundGeneration = -1;

  private dpr = 1;

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
