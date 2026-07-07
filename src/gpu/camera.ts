/**
 * Minimal fixed camera for the GPU path — replaces three.js.
 *
 * Matches the reference exactly: PerspectiveCamera(55°, aspect, 0.1, 200) at
 * (0, 0, 26) looking down -z with identity rotation, and the same pointer
 * unprojection onto the z=0 swarm plane.
 */
export class SwarmCamera {
  readonly fovDeg = 55;
  readonly near = 0.1;
  readonly far = 200;
  /** Camera z position; the swarm lives on the z=0 plane. */
  readonly z = 26;
  aspect = 1;

  /** Column-major view-projection matrix (WebGPU [0,1] clip z). */
  readonly viewProj = new Float32Array(16);

  constructor(aspect: number) {
    this.setAspect(aspect);
  }

  setAspect(aspect: number): void {
    this.aspect = Math.max(1e-4, aspect);
    const f = 1 / Math.tan((this.fovDeg * Math.PI) / 360);
    const { near, far } = this;
    const a = far / (near - far); // clip.z scale
    const b = (far * near) / (near - far); // clip.z offset
    // View is a pure translation (0,0,-camZ); fold it into the projection.
    // View-space z: zv = z − camZ (negative in front of the camera).
    //   clip.z = a·zv + b = a·z + (b − a·camZ)
    //   clip.w = −zv     = −z + camZ
    const m = this.viewProj;
    m.fill(0);
    m[0] = f / this.aspect;
    m[5] = f;
    m[10] = a;
    m[11] = -1;
    m[14] = b - a * this.z;
    m[15] = this.z;
  }

  /** −(view-space z) for a world position: distance in front of the camera. */
  viewDepth(worldZ: number): number {
    return this.z - worldZ;
  }

  /**
   * Project an NDC pointer ([-1,1]²) onto the plane z = planeZ.
   *
   * The reference pinned this to z=0 (the school hovers near it by
   * construction); the fork passes the school's live centroid z so the
   * predator column always pierces the school's mid-depth — a school that
   * has breathed or drifted a few units in z would otherwise only be
   * grazed by the depth-squashed influence column.
   */
  unprojectToSwarmPlane(
    ndc: { x: number; y: number },
    planeZ = 0,
  ): { x: number; y: number; z: number } {
    const tanHalf = Math.tan((this.fovDeg * Math.PI) / 360);
    // Ray from (0,0,camZ) through the pixel hits z=planeZ at t = camZ−planeZ.
    const t = this.z - planeZ;
    return {
      x: ndc.x * tanHalf * this.aspect * t,
      y: ndc.y * tanHalf * t,
      z: planeZ,
    };
  }
}
