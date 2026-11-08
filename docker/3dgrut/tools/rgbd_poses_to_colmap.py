#!/usr/bin/env python3
"""Build a COLMAP *text* model (cameras/images/points3D) from RGB-D frames + known camera poses.

Dev/smoke helper (not the COLMAP stage): lets 3DGRUT train on RTAB-Map poses directly. The initial
point cloud is back-projected from the sensor depth, so scale stays metric.

Inputs
  --frames DIR      folder with rgb/ and depth/ (timestamp-named PNGs, depth = 16-bit millimetres)
  --poses FILE      RTAB-Map export, TUM format "stamp x y z qx qy qz qw" = world<-base (x fwd, y left, z up)
                    (rtabmap-rgbd_dataset *_poses.txt). Use --optical if poses are already world<-optical.
  --out DIR         writes DIR/images/*.png, DIR/sparse/0/{cameras,images,points3D}.txt
  --fx --fy --cx --cy --width --height   pinhole intrinsics of the RGB image
  --every N         keep every Nth posed frame; --max-images M cap
  --masks DIR       optional folder with <stamp>.png masks (255 keep / 0 ignore) -> DIR/images/<stamp>_mask.png
"""
import argparse, os, shutil, bisect
import numpy as np, cv2


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw); qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])


def R_to_quat(R):  # w, x, y, z (COLMAP order)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; w = s/4; x = (R[2,1]-R[1,2])/s; y = (R[0,2]-R[2,0])/s; z = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w = (R[2,1]-R[1,2])/s; x = s/4; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = s/4; z = (R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = s/4
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", required=True); a.add_argument("--poses", required=True); a.add_argument("--out", required=True)
    for k in ("fx", "fy", "cx", "cy"): a.add_argument("--" + k, type=float, required=True)
    a.add_argument("--width", type=int, default=848); a.add_argument("--height", type=int, default=480)
    a.add_argument("--every", type=int, default=1); a.add_argument("--max-images", type=int, default=0)
    a.add_argument("--optical", action="store_true"); a.add_argument("--masks", default="")
    a.add_argument("--points-every", type=int, default=5, help="use depth of every Nth kept image for the init cloud")
    a.add_argument("--max-points", type=int, default=200000); a.add_argument("--max-depth-m", type=float, default=4.0)
    o = a.parse_args()

    L = np.eye(4); L[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # optical -> rtabmap base
    rows = [l.split() for l in open(o.poses) if l.strip() and not l.startswith("#")]
    rgb_files = sorted(f for f in os.listdir(os.path.join(o.frames, "rgb")) if f.endswith(".png"))
    stamps = np.array([float(f[:-4]) for f in rgb_files])
    items = []
    for r in rows:
        t = float(r[0]); i = int(np.argmin(np.abs(stamps - t)))
        if abs(stamps[i] - t) > 0.02: continue
        v = list(map(float, r[1:8])); T = np.eye(4); T[:3, :3] = quat_to_R(*v[3:7]); T[:3, 3] = v[:3]
        items.append((rgb_files[i], T if o.optical else T @ L))
    items = items[::o.every]
    if o.max_images: items = items[:o.max_images]
    print(f"{len(items)} posed frames kept")

    os.makedirs(os.path.join(o.out, "images"), exist_ok=True); sp = os.path.join(o.out, "sparse", "0"); os.makedirs(sp, exist_ok=True)
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {o.width} {o.height} {o.fx} {o.fy} {o.cx} {o.cy}\n")
    pts, cols = [], []
    with open(os.path.join(sp, "images.txt"), "w") as f:
        for idx, (name, Twc) in enumerate(items):
            Tcw = np.linalg.inv(Twc); q = R_to_quat(Tcw[:3, :3]); t = Tcw[:3, 3]
            f.write(f"{idx+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
            shutil.copy(os.path.join(o.frames, "rgb", name), os.path.join(o.out, "images", name))
            if o.masks and os.path.exists(os.path.join(o.masks, name)):
                shutil.copy(os.path.join(o.masks, name), os.path.join(o.out, "images", name[:-4] + "_mask.png"))
            if idx % o.points_every == 0:
                d = cv2.imread(os.path.join(o.frames, "depth", name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                c = cv2.cvtColor(cv2.imread(os.path.join(o.frames, "rgb", name)), cv2.COLOR_BGR2RGB)
                vs, us = np.mgrid[0:d.shape[0]:4, 0:d.shape[1]:4]; z = d[vs, us]; ok = (z > 0.2) & (z < o.max_depth_m)
                u, v, z = us[ok], vs[ok], z[ok]
                P = np.stack([(u - o.cx) * z / o.fx, (v - o.cy) * z / o.fy, z, np.ones_like(z)], 1) @ Twc.T
                pts.append(P[:, :3]); cols.append(c[v, u])
    P = np.concatenate(pts); C = np.concatenate(cols)
    if len(P) > o.max_points:
        sel = np.random.default_rng(0).choice(len(P), o.max_points, replace=False); P, C = P[sel], C[sel]
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        for i, (p, c) in enumerate(zip(P, C)):
            f.write(f"{i+1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0.5\n")
    print(f"wrote {o.out}: {len(items)} images, {len(P)} points")


if __name__ == "__main__":
    main()
