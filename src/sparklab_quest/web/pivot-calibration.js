export const quatToMat3 = (q) => {
  const [x, y, z, w] = q;
  const xx = x*x, yy = y*y, zz = z*z;
  const xy = x*y, xz = x*z, yz = y*z;
  const wx = w*x, wy = w*y, wz = w*z;
  return [
    [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
    [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
    [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
  ];
};

const solve3x3 = (matrix, vector) => {
  const [m00, m01, m02] = matrix[0];
  const [m10, m11, m12] = matrix[1];
  const [m20, m21, m22] = matrix[2];
  const det = m00*(m11*m22 - m12*m21)
    - m01*(m10*m22 - m12*m20)
    + m02*(m10*m21 - m11*m20);
  const trace = Math.abs(m00) + Math.abs(m11) + Math.abs(m22);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-6 * Math.max(1, trace)**3) {
    return { x: null, det };
  }
  const [v0, v1, v2] = vector;
  const dx = v0*(m11*m22 - m12*m21) - m01*(v1*m22 - m12*v2) + m02*(v1*m21 - m11*v2);
  const dy = m00*(v1*m22 - m12*v2) - v0*(m10*m22 - m12*m20) + m02*(m10*v2 - v1*m20);
  const dz = m00*(m11*v2 - v1*m21) - m01*(m10*v2 - v1*m20) + v0*(m10*m21 - m11*m20);
  return { x: [dx/det, dy/det, dz/det], det };
};

export const solvePivot = (samples) => {
  const count = samples.length;
  if (count < 30) return { ok: false, reason: `too few samples (${count})` };
  const meanPosition = [0, 0, 0];
  const meanRotation = [[0,0,0], [0,0,0], [0,0,0]];
  for (const sample of samples) {
    for (let i = 0; i < 3; i++) {
      meanPosition[i] += sample.p[i];
      for (let j = 0; j < 3; j++) meanRotation[i][j] += sample.R[i][j];
    }
  }
  for (let i = 0; i < 3; i++) {
    meanPosition[i] /= count;
    for (let j = 0; j < 3; j++) meanRotation[i][j] /= count;
  }

  const normal = [[0,0,0], [0,0,0], [0,0,0]];
  const rhs = [0, 0, 0];
  for (const sample of samples) {
    const deltaRotation = [[0,0,0], [0,0,0], [0,0,0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        deltaRotation[i][j] = sample.R[i][j] - meanRotation[i][j];
      }
    }
    const deltaPosition = sample.p.map((value, i) => value - meanPosition[i]);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) {
          normal[i][j] += deltaRotation[k][i] * deltaRotation[k][j];
        }
      }
      for (let k = 0; k < 3; k++) rhs[i] += deltaRotation[k][i] * deltaPosition[k];
    }
  }
  const solution = solve3x3(normal, rhs.map((value) => -value));
  if (!solution.x) {
    return { ok: false, reason: `ill-conditioned (det=${solution.det.toExponential(2)})` };
  }

  const offset = solution.x;
  const pivots = samples.map((sample) => sample.p.map((value, row) =>
    value + sample.R[row][0]*offset[0] + sample.R[row][1]*offset[1] + sample.R[row][2]*offset[2]
  ));
  const center = [0, 0, 0];
  for (const pivot of pivots) {
    for (let i = 0; i < 3; i++) center[i] += pivot[i] / count;
  }
  let sumSquares = 0;
  for (const pivot of pivots) {
    for (let i = 0; i < 3; i++) sumSquares += (pivot[i] - center[i]) ** 2;
  }
  return { ok: true, o: offset, rms: Math.sqrt(sumSquares / count), n: count };
};
