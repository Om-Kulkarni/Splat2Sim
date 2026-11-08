"""Write a small synthetic 3DGS .ply (checker floor + 3 coloured pillars) for smoke tests.
Usage: python make_splat.py out.ply   (numpy only)"""
import sys, numpy as np
rng = np.random.default_rng(0)
C0 = 0.28209479177387814
pts, cols = [], []
n = 40000
xy = rng.uniform(-1.5, 1.5, (n, 2)); z = np.zeros(n) + rng.normal(0, .003, n)
chk = ((np.floor(xy[:, 0] / .25) + np.floor(xy[:, 1] / .25)) % 2)[:, None]
pts.append(np.c_[xy, z]); cols.append(np.where(chk > 0, [.8, .8, .8], [.2, .3, .6]))
for cx, cy, col in [(0, 0, (1, .1, .1)), (.6, .4, (.1, 1, .1)), (-.5, .5, (1, 1, .1))]:
    m = 6000; a = rng.uniform(0, 2*np.pi, m); h = rng.uniform(0, .5, m)
    pts.append(np.c_[cx + .1*np.cos(a), cy + .1*np.sin(a), h]); cols.append(np.tile(col, (m, 1)))
P = np.vstack(pts).astype(np.float32); Cc = np.vstack(cols).astype(np.float32); N = len(P)
dc = ((Cc - .5) / C0).astype(np.float32)
names = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2"] + [f"f_rest_{i}" for i in range(45)] + ["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
D = np.zeros((N, len(names)), np.float32)
D[:, 0:3] = P; D[:, 6:9] = dc
D[:, 54] = 4.0                      # opacity logit
D[:, 55:58] = np.log(0.025)         # scale (log)
D[:, 58] = 1.0                      # quat w
hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N + "".join(f"property float {n}\n" for n in names) + "end_header\n"
with open(sys.argv[1], "wb") as f:
    f.write(hdr.encode()); f.write(D.tobytes())
print("wrote", sys.argv[1], N, "gaussians")
