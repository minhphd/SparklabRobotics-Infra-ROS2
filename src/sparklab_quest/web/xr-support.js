export async function openXRSession(mode, gl, onEnd) {
  const session = await navigator.xr.requestSession(mode, {
    optionalFeatures: ["local-floor"],
  });
  try {
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    const refSpace = await session.requestReferenceSpace("local-floor")
      .catch(() => session.requestReferenceSpace("local"));
    session.addEventListener("end", onEnd, { once: true });
    return { session, refSpace };
  } catch (error) {
    await session.end();
    throw error;
  }
}

export const xyz = (v) => [v.x, v.y, v.z];
export const xyzw = (v) => [v.x, v.y, v.z, v.w];

export const rotateVecByQuat = (v, q) => {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  const c1x = qy * vz - qz * vy;
  const c1y = qz * vx - qx * vz;
  const c1z = qx * vy - qy * vx;
  const c2x = qy * c1z - qz * c1y;
  const c2y = qz * c1x - qx * c1z;
  const c2z = qx * c1y - qy * c1x;
  return [
    vx + 2 * qw * c1x + 2 * c2x,
    vy + 2 * qw * c1y + 2 * c2y,
    vz + 2 * qw * c1z + 2 * c2z,
  ];
};

export const mat4Mul = (a, b) => {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      let value = 0;
      for (let k = 0; k < 4; k++) value += a[row + k * 4] * b[k + column * 4];
      out[row + column * 4] = value;
    }
  }
  return out;
};

export const mat4FromPosQuat = (p, q) => {
  const [x, y, z, w] = q;
  return new Float32Array([
    1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y),     0,
    2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x),     0,
    2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y), 0,
    p[0],              p[1],              p[2],              1,
  ]);
};
