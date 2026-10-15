import type { GpuSwarm, SimStage } from "./GpuSwarm";
import type { GpuRenderer } from "./renderer";

/**
 * `await __stages()`: where a frame's GPU time goes on this device.
 *
 * Browsers here have no timestamp queries (Safari), so each stage is timed
 * from the outside: encode it `repeat` times in one command buffer, submit,
 * await onSubmittedWorkDone, and subtract an empty submit's round trip (the
 * same latency the governor's meter carries). Medians over `samples` runs.
 * Built to find the ~14 ms an iPhone 11 spends per frame at any count
 * (docs/NOTES.md); driven over a cable by scripts/device-profile.mjs.
 */

export interface ProfileOptions {
  /** Encodes per timed submit. More amortizes the round trip. Default 10. */
  repeat?: number;
  /** Timed submits per stage; the median is reported. Default 7. */
  samples?: number;
}

export interface StageReport {
  count: number;
  /** Drawing-buffer size the render passes fill. */
  pixels: [number, number];
  /** An empty submit's round trip: the meter's floor on this device. */
  emptySubmitMs: number;
  /** One real frame (update + render), submit to done, as the governor
   * measures it. */
  frameMs: number;
  /** Per stage, one pass's cost: (median submit − empty submit) / repeat. */
  stagesMs: Record<SimStage | "sprites" | "tonemap" | "readback", number>;
  /** Sum of stagesMs: compare with frameMs − emptySubmitMs. */
  sumMs: number;
}

const SIM_STAGES: SimStage[] = ["clear", "count", "scan", "scatter", "sim", "reduce"];

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
};

export async function profileStages(
  device: GPUDevice,
  system: GpuSwarm,
  renderer: GpuRenderer,
  opts: ProfileOptions = {},
): Promise<StageReport> {
  const repeat = opts.repeat ?? 10;
  const samples = opts.samples ?? 7;
  const q = device.queue;
  await q.onSubmittedWorkDone(); // start from an idle queue

  const roundTrip = async (encode?: (enc: GPUCommandEncoder) => void): Promise<number> => {
    const enc = device.createCommandEncoder({ label: "profile" });
    encode?.(enc);
    const cmd = enc.finish();
    const t0 = performance.now();
    q.submit([cmd]);
    await q.onSubmittedWorkDone();
    return performance.now() - t0;
  };
  // Repeated count/scatter passes scramble the grids, and a sim pass over a
  // scrambled grid walks garbage cells (4.4 ms a pass on the M2 Max against
  // a 1.4 ms frame). So every timed submit starts from a clean build.
  const rebuildGrids = (): Promise<number> =>
    roundTrip((enc) => {
      for (const s of ["clear", "count", "scan", "scatter"] as const) system.encodeStage(enc, s, 1);
    });
  const timed = async (encode?: (enc: GPUCommandEncoder) => void): Promise<number> => {
    await rebuildGrids();
    await roundTrip(encode); // warm: first use can compile or allocate
    const xs: number[] = [];
    for (let i = 0; i < samples; i++) {
      await rebuildGrids();
      xs.push(await roundTrip(encode));
    }
    return median(xs);
  };

  const emptySubmitMs = await timed();
  const per = (ms: number): number => Math.max(0, (ms - emptySubmitMs) / repeat);

  const stagesMs = {} as StageReport["stagesMs"];
  for (const stage of SIM_STAGES) {
    stagesMs[stage] = per(await timed((enc) => system.encodeStage(enc, stage, repeat)));
  }
  for (const pass of ["sprites", "tonemap"] as const) {
    stagesMs[pass] = per(await timed((enc) => renderer.encodePass(enc, pass, repeat)));
  }

  // The frame's readback: a 32-byte copy into a mappable buffer, mapped.
  const src = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_SRC });
  const dst = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readback = async (): Promise<number> => {
    const enc = device.createCommandEncoder({ label: "profile:readback" });
    enc.copyBufferToBuffer(src, 0, dst, 0, 32);
    const t0 = performance.now();
    q.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const ms = performance.now() - t0;
    dst.unmap();
    return ms;
  };
  await readback();
  const rb: number[] = [];
  for (let i = 0; i < samples; i++) rb.push(await readback());
  stagesMs.readback = Math.max(0, median(rb) - emptySubmitMs);
  src.destroy();
  dst.destroy();

  // A real frame, as the governor sees it.
  const frames: number[] = [];
  for (let i = 0; i <= samples; i++) {
    const t0 = performance.now();
    system.update(1 / 60);
    renderer.render();
    await q.onSubmittedWorkDone();
    if (i > 0) frames.push(performance.now() - t0);
  }

  const canvas = renderer.canvasSize();
  const round = (x: number): number => Math.round(x * 100) / 100;
  for (const k of Object.keys(stagesMs) as (keyof typeof stagesMs)[]) stagesMs[k] = round(stagesMs[k]);
  return {
    count: system.getCount(),
    pixels: canvas,
    emptySubmitMs: round(emptySubmitMs),
    frameMs: round(median(frames)),
    stagesMs,
    sumMs: round(Object.values(stagesMs).reduce((a, b) => a + b, 0)),
  };
}
