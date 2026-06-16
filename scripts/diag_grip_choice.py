"""User (Image #10) circled where the gripper should clamp the handle: closer to the
head. That is a rigid slide of the hammer along the handle axis (body -Y) so the head
moves toward the jaws and the clamped section of handle moves toward the head. Sweep a
few shifts, render the whole hammer in the Image-#10 side view with the gripper jaw
pads highlighted CYAN (so you can see exactly where the clamp lands on the handle),
and save to ~/Desktop/hammer_orient/.

Tell me which shift puts the clamp on your circle; I'll set the geom pos in both XMLs,
move the site to the shifted face centroid, re-tune NEAR_NAIL, and re-verify.
"""
from __future__ import annotations
import os
import imageio.v3 as iio
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from src.assets.robots.unitree_z1.z1_constants import get_spec, NEAR_NAIL_JOINT_POS

GRASP = [0.70710678, 0.0, 0.0, 0.70710678]
HGEOMS = ["hammer_visual", "hammer_head_0", "hammer_head_1", "hammer_claw_0",
          "hammer_claw_1", "hammer_handle_col"]
BASE_POS = np.array([0.0, 0.086, 0.0])
SHIFTS = [-0.07, -0.08, -0.09]    # body-Y; head toward jaws (grip moves toward head)
HEADP = ["hammer_head_0", "hammer_head_1", "hammer_claw_0", "hammer_claw_1"]
OUT = os.path.expanduser("~/Desktop/hammer_orient/grip_choice.png")


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def build(shift):
    spec = get_spec()
    spec.visual.global_.offwidth = 760
    spec.visual.global_.offheight = 620
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.worldbody.add_light(pos=[0.5, -0.3, 0.9], dir=[0, 0.2, -0.9], diffuse=[0.9, 0.9, 0.9])
    spec.worldbody.add_geom(name="_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
                            pos=[0.5, 0, 0.06], rgba=[0.85, 0.85, 0.88, 1], contype=0, conaffinity=0)
    for g in spec.geoms:
        if g.name in HGEOMS:
            g.quat = list(GRASP)
            g.pos = list(BASE_POS + np.array([0.0, shift, 0.0]))
    m = spec.compile()
    d = mujoco.MjData(m)
    for nm, v in NEAR_NAIL_JOINT_POS.items():
        d.qpos[m.jnt_qposadr[m.joint(nm).id]] = v
    for gid in range(m.ngeom):
        bid = m.geom_bodyid[gid]
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gid]) if m.geom_dataid[gid] >= 0 else ""
        if m.geom_dataid[gid] < 0 and bn in ("link06", "gripperMover"):
            m.geom_rgba[gid] = np.array([0, 0.85, 1, 1])   # jaw pads cyan (opaque)
        elif mname and "Gripper" in mname:
            m.geom_matid[gid] = -1
            m.geom_rgba[gid] = np.array([0.8, 0.8, 0.82, 0.25])   # housing translucent
    mujoco.mj_forward(m, d)
    return m, d


def grip_point(m, d):
    """Point on the handle axis closest to the gripper grasp center (ee_center) =
    where the gripper effectively clamps the handle."""
    gid = m.geom("hammer_handle_col").id
    Rg = d.geom_xmat[gid].reshape(3, 3); p = d.geom_xpos[gid]
    mid = m.geom_dataid[gid]; va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = (Rg @ m.mesh_vert[va:va + vn].reshape(-1, 3).T).T + p
    ctr = v.mean(0); axis = np.linalg.svd(v - ctr)[2][0]
    ee = d.xpos[m.body("ee_center_body").id]
    t = float((ee - ctr) @ axis)
    return ctr + t * axis


def head_robot_clearance(m, d):
    """Min distance between hammer head pieces and any non-hammer robot geom
    (gripper/wrist), to catch the head clashing into the gripper at large shifts."""
    ft = np.zeros(6)
    robot_ids = []
    for gid in range(m.ngeom):
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gn.startswith("hammer_") or gn.startswith("_"):
            continue
        if m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0:
            continue
        robot_ids.append(gid)
    clr = 1e9
    for hp in HEADP:
        hid = m.geom(hp).id
        for gid in robot_ids:
            clr = min(clr, mujoco.mj_geomDistance(m, d, hid, gid, 0.4, ft))
    return clr


def main():
    cells = []
    for shift in SHIFTS:
        m, d = build(shift)
        face_w = d.geom_xpos[m.geom("hammer_head_0").id].copy()
        ee = d.xpos[m.body("ee_center_body").id].copy()
        gp = grip_point(m, d)
        clr = head_robot_clearance(m, d)
        print(f"shift {int(shift*1000):+d} mm: head<->gripper/wrist clearance = {clr*1000:.1f} mm")
        look = list((face_w + ee) / 2)
        r = mujoco.Renderer(m, height=600, width=740)
        cam = mujoco.MjvCamera(); cam.lookat[:] = look
        cam.distance = 0.46; cam.azimuth = 330; cam.elevation = -12
        r.update_scene(d, cam)
        scn = r.scene
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]),
                            gp.astype(float), np.eye(3).flatten(),
                            np.array([1, 0.1, 0.1, 1], np.float32))   # RED = grip point
        scn.ngeom += 1
        img = r.render().copy()
        clash = clr < 0.005
        bar = Image.new("RGB", (740, 54), (18, 18, 18))
        db = ImageDraw.Draw(bar)
        db.text((8, 4), f"shift {int(shift*1000):+d} mm   head<->gripper gap {clr*1000:.0f} mm"
                        f"{'  CLASH' if clash else ''}",
                fill=(255, 140, 120) if clash else (255, 225, 0), font=font(24))
        out = Image.new("RGB", (740, 54 + 600), (0, 0, 0))
        out.paste(bar, (0, 0)); out.paste(Image.fromarray(img), (0, 54))
        if clash:
            ImageDraw.Draw(out).rectangle([0, 0, 739, 54 + 600 - 1], outline=(235, 60, 60), width=5)
        cells.append(np.asarray(out))
    iio.imwrite(OUT, np.concatenate(cells, axis=1))
    print(f"[grip_choice] -> {OUT}")
    print("  cyan = gripper jaw pads (the clamp); pick the shift whose clamp sits on your circle")


if __name__ == "__main__":
    main()
