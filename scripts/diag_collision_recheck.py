"""Fresh collision re-check against the CURRENT PRODUCTION config (loads get_spec()
as-is: grasp #10, 80mm grip slide pos 0.006, renamed geoms, gravcomp). Verifies:
  1. collision-union vs visual-mesh world AABB alignment (faithfulness)
  2. per-piece sanity (all 5 present, non-degenerate)
  3. face (hammer_head_0=c4) leads contact to the nail; no other piece closer
  4. zero spurious contacts at the reset pose (ncon, and per hammer geom)
  5. overlay render (collision colored inside translucent visual) -> Desktop
"""
from __future__ import annotations
import os
import imageio.v3 as iio
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from src.assets.robots.unitree_z1.z1_constants import get_spec, NEAR_NAIL_JOINT_POS

FACE = "hammer_head_0"            # c4 striking face
NECK = "hammer_head_1"            # c3
CLAW = ["hammer_claw_0", "hammer_claw_1"]
HANDLE = "hammer_handle_col"
COLL = [FACE, NECK, *CLAW, HANDLE]
NAMES = {FACE: "FACE c4", NECK: "neck c3", "hammer_claw_0": "claw c1",
         "hammer_claw_1": "claw c2", HANDLE: "handle c0"}
NAIL = np.array([0.5, 0.0, 0.096])
OUT = os.path.expanduser("~/Desktop/hammer_orient/collision_recheck.png")


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def world_verts(m, d, g):
    gid = m.geom(g).id
    R = d.geom_xmat[gid].reshape(3, 3); p = d.geom_xpos[gid]
    mid = m.geom_dataid[gid]; va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[va:va + vn].reshape(-1, 3)
    return (R @ v.T).T + p


def main():
    spec = get_spec()
    spec.visual.global_.offwidth = 700; spec.visual.global_.offheight = 700
    spec.worldbody.add_light(pos=[0.5, -0.3, 0.85], dir=[0, 0.2, -0.9], diffuse=[1, 1, 1])
    spec.worldbody.add_geom(name="_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
                            pos=[0.5, 0, 0.06], rgba=[0.85, 0.85, 0.88, 1], contype=0, conaffinity=0)
    nb = spec.worldbody.add_body(name="_nail", pos=NAIL.tolist())
    nb.add_geom(name="_nh", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.004, 0],
                rgba=[0.1, 0.95, 0.1, 1], contype=1, conaffinity=1)
    m = spec.compile(); d = mujoco.MjData(m)
    for nm, v in NEAR_NAIL_JOINT_POS.items():
        d.qpos[m.jnt_qposadr[m.joint(nm).id]] = v
    mujoco.mj_forward(m, d)

    print("=== 1. ALIGNMENT (collision-union vs visual AABB) ===")
    vv = world_verts(m, d, "hammer_visual")
    cu = np.concatenate([world_verts(m, d, c) for c in COLL], axis=0)
    dlo = np.abs(cu.min(0) - vv.min(0)) * 1000; dhi = np.abs(cu.max(0) - vv.max(0)) * 1000
    print(f"  per-axis lo diff(mm)={np.round(dlo,2)} hi diff(mm)={np.round(dhi,2)}")
    print(f"  MAX corner mismatch = {max(dlo.max(), dhi.max()):.2f} mm")

    print("=== 2. PER-PIECE sanity ===")
    for c in COLL:
        w = world_verts(m, d, c); ext = (w.max(0) - w.min(0)) * 1000
        print(f"  {NAMES[c]:9s} ({c:18s}) nverts={w.shape[0]:4d} extent(mm)={np.round(ext,1)}")

    print("=== 3. REACHABILITY (piece -> nail) ===")
    ng = m.geom("_nh").id; ft = np.zeros(6); dists = {}
    for c in COLL:
        dists[c] = mujoco.mj_geomDistance(m, d, m.geom(c).id, ng, 1.0, ft)
        print(f"  {NAMES[c]:9s} dist to nail = {dists[c]*1000:6.1f} mm")
    closest = min(dists, key=dists.get)
    print(f"  -> CLOSEST = {NAMES[closest]} ({'OK face leads' if closest == FACE else 'WARN'})")

    print("=== 4. CONTACTS at reset ===")
    print(f"  total ncon = {d.ncon}")
    hb = {m.body(b).id for b in ("hammer",)}
    hgeoms = {m.geom(c).id for c in COLL}
    bad = [i for i in range(d.ncon)
           if d.contact[i].geom1 in hgeoms or d.contact[i].geom2 in hgeoms]
    print(f"  contacts involving a hammer geom = {len(bad)} (want 0 at rest)")

    # 5. overlay render
    cols = {FACE: [1, 0.1, 0.1, 1], NECK: [0.95, 0.75, 0.1, 1], CLAW[0]: [0, 0.85, 1, 1],
            CLAW[1]: [0, 0.85, 1, 1], HANDLE: [0.6, 0.6, 0.62, 1]}
    for c, rgba in cols.items():
        m.geom_rgba[m.geom(c).id] = np.array(rgba)
    vid = m.geom("hammer_visual").id
    m.geom_matid[vid] = -1; m.geom_rgba[vid] = np.array([0.85, 0.85, 0.85, 0.3])
    mujoco.mj_forward(m, d)
    allopt = mujoco.MjvOption(); allopt.geomgroup[:] = 1
    face_w = d.geom_xpos[m.geom(FACE).id].copy()

    def shot(az, el, dist, look):
        r = mujoco.Renderer(m, height=680, width=680)
        cam = mujoco.MjvCamera(); cam.lookat[:] = look; cam.distance = dist
        cam.azimuth = az; cam.elevation = el
        r.update_scene(d, cam, scene_option=allopt)
        return r.render().copy()
    head = list(face_w + np.array([0, -0.01, 0.03]))
    body = np.concatenate([shot(150, -8, 0.34, head), shot(90, -8, 0.34, head),
                           shot(150, -22, 0.17, list(face_w))], axis=1)
    bar = Image.new("RGB", (body.shape[1], 60), (18, 18, 18))
    ImageDraw.Draw(bar).text((10, 6),
        f"COLLISION (colored) inside VISUAL (translucent) @ production config | align "
        f"{max(dlo.max(),dhi.max()):.1f}mm | face leads {dists[FACE]*1000:.0f}mm | RED=face",
        fill=(255, 225, 0), font=font(20))
    out = Image.new("RGB", (body.shape[1], 60 + body.shape[0]), (0, 0, 0))
    out.paste(bar, (0, 0)); out.paste(Image.fromarray(body), (0, 60))
    iio.imwrite(OUT, np.asarray(out))
    print(f"\n[collision_recheck] -> {OUT}")


if __name__ == "__main__":
    main()
