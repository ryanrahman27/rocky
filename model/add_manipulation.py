#!/usr/bin/env python3
"""Derive model/rocky_manip.xml from model/rocky.xml: gripper DOFs + cameras.

The locomotion export welds the gripper shut, because for walking the closed
jaws and their rubber pad are simply the fifth foot. Manipulation needs them
back. Rather than re-exporting from Fusion, this lifts the two jaw meshes out of
the gripper body into their own bodies on slider joints.

The jaws are a rack-and-pinion pair: identical parts mirrored about the gripper
centreline at y = +7.6 mm, exported from CAD in the CLOSED position, so q = 0 is
closed and each opens 30 mm along its own outward Y (60 mm total, which is what
the CAD's gripper_open parameter says).

Also recomputes the gripper's inertial and collision hull without the jaws --
the locomotion hull swallowed them, and a hull with the jaws inside it would
stop a cube ever entering the fingers.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "rocky.xml", HERE / "rocky_manip.xml"

PETG = 1.27
CENTRELINE_Y = 0.0076          # gripper centreline, from the mirrored jaw meshes
JAW_TRAVEL = 0.030             # per jaw; 60 mm total opening

# The jaws are driven by a servo through a 2 mm-lead screw, so the rotor's
# inertia shows up at the jaw as a reflected MASS: I_rotor * (2*pi / lead)^2.
# The leg servos' armature of 5.951e-3 kg m^2 at a 1:345 reduction implies
# I_rotor = 5.0e-8 kg m^2, which through a 2 mm lead comes out at 0.494 kg.
# Skipping this -- the first cut used armature 0.002 and kp 2500 -- makes the
# jaw servo's natural frequency 625 rad/s against a 5 ms timestep, and it
# promptly integrates itself straight through the joint limits.
JAW_ARMATURE = 0.494           # reflected rotor mass at the jaw, kg
JAW_BANDWIDTH = 2.0 * 3.141592653589793 * 10.0     # rad/s, same 10 Hz as the legs
JAW_ZETA = 2.0                 # same overdamped target as the legs
JAW_KP = JAW_ARMATURE * JAW_BANDWIDTH ** 2
JAW_KV = 2.0 * JAW_ZETA * JAW_ARMATURE * JAW_BANDWIDTH
JAW_FORCE = 20.0               # N, the screw's working thrust
FINGER_X = (0.061, 0.074)      # finger extent along the gripper +X
FINGER_Z = (-0.008, 0.008)
FINGER_T = 0.006               # finger thickness in Y

# gripper bodies that stay with the base once the jaws move out
BASE_BODIES = ["Gripper_Base", "Foot_Pad_Manip", "F6700ZZ_WristIdler",
               "STS3215_Gripper", "Horn_Gripper", "Pinion_12T"]
FIXED_MASS = {"STS3215_Gripper": 0.060, "F6700ZZ_WristIdler": 0.0043}
RHO = {"Foot_Pad_Manip": 1.20, "Horn_Gripper": 1.15}


def body_props(name: str):
    m = trimesh.load(HERE / "meshes" / f"gripper_m0__{name}.stl", process=True)
    m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); m.fix_normals()
    vol_cm3 = m.volume * 1e6
    mass = FIXED_MASS.get(name, RHO.get(name, PETG) * vol_cm3 / 1000.0)
    m.density = mass / m.volume
    mp = m.mass_properties
    return m, mass, np.array(mp["center_mass"]), np.array(mp["inertia"]) * (mass / mp["mass"])


def combine(items):
    M = sum(i[0] for i in items)
    C = sum(i[0] * i[1] for i in items) / M
    I = np.zeros((3, 3))
    for m, c, Ic in items:
        d = c - C
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, C, I


def inertial_xml(indent, M, C, I):
    return ('%s<inertial pos="%.9f %.9f %.9f" mass="%.9f" '
            'fullinertia="%.10g %.10g %.10g %.10g %.10g %.10g"/>'
            % (indent, C[0], C[1], C[2], M, I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))


def main() -> int:
    s = SRC.read_text()

    # --- recompute the gripper base and the two jaws -----------------------
    base_items, jaws = [], {}
    for n in BASE_BODIES:
        _, m, c, I = body_props(n)
        base_items.append((m, c, I))
    for n in ("Jaw_R", "Jaw_L"):
        mesh, m, c, I = body_props(n)
        jaws[n] = (mesh, m, c, I)
    gM, gC, gI = combine(base_items)

    # --- collision hull without the jaws ----------------------------------
    keep = [body_props(n)[0] for n in BASE_BODIES if n != "Foot_Pad_Manip"]
    hull = trimesh.convex.convex_hull(trimesh.util.concatenate(keep))
    q = np.unique(np.round(hull.vertices / 0.003).astype(int), axis=0) * 0.003
    hull = trimesh.convex.convex_hull(trimesh.PointCloud(q))
    hull.export(HERE / "meshes" / "hull_gripper_nojaws.stl")

    s = s.replace('<mesh name="hull_gripper" file="hull_gripper.stl"/>',
                  '<mesh name="hull_gripper" file="hull_gripper.stl"/>\n'
                  '    <mesh name="hull_gripper_nojaws" file="hull_gripper_nojaws.stl"/>')

    # --- rebuild the gripper body -----------------------------------------
    old = re.search(r'( *)<body name="gripper_0".*?\n\1</body>\n', s, re.S)
    assert old, "gripper_0 body not found"
    ind = old.group(1)
    i2, i3 = ind + "  ", ind + "    "

    def jaw_block(name, sign, jointname):
        mesh, m, c, I = jaws[name]
        y = CENTRELINE_Y + sign * (FINGER_T / 2.0)
        return "\n".join([
            f'{i2}<body name="{jointname}">',
            f'{i2}  <joint name="{jointname}" type="slide" axis="0 {sign:+d} 0" '
            f'range="0 {JAW_TRAVEL:.4f}" actuatorfrcrange="-{JAW_FORCE:.0f} {JAW_FORCE:.0f}" '
            f'damping="4" armature="{JAW_ARMATURE:.4f}"/>',
            inertial_xml(i2 + "  ", m, c, I),
            f'{i2}  <geom class="visual" mesh="gripper_m0__{name}" material="sandstone_dark"/>',
            # the finger pad is the only part that should ever touch a cube
            f'{i2}  <geom name="{jointname}_pad" class="collision" type="box" '
            f'size="{(FINGER_X[1]-FINGER_X[0])/2:.4f} {FINGER_T/2:.4f} {(FINGER_Z[1]-FINGER_Z[0])/2:.4f}" '
            f'pos="{(FINGER_X[0]+FINGER_X[1])/2:.4f} {y:.4f} {(FINGER_Z[0]+FINGER_Z[1])/2:.4f}" '
            f'condim="6" priority="2" friction="1.4 0.05 0.002" solimp="0.95 0.99 0.001"/>',
            f'{i2}</body>',
        ])

    new = "\n".join([
        f'{ind}<body name="gripper_0" pos="0.0900000000 0 0">',
        f'{i2}<joint name="wrist_0" axis="0 1 0" range="-1.3089969390 1.3089969390" actuatorfrcrange="-2.90 2.90"/>',
        inertial_xml(i2, gM, gC, gI),
        f'{i2}<geom class="visual" mesh="gripper_m0__Gripper_Base" material="sandstone_dark"/>',
        f'{i2}<geom class="visual" mesh="gripper_m0__Foot_Pad_Manip" material="rubber"/>',
        f'{i2}<geom class="visual" mesh="gripper_m0__F6700ZZ_WristIdler" material="metal"/>',
        f'{i2}<geom class="visual" mesh="gripper_m0__STS3215_Gripper" material="metal"/>',
        f'{i2}<geom class="visual" mesh="gripper_m0__Horn_Gripper" material="metal"/>',
        f'{i2}<geom class="visual" mesh="gripper_m0__Pinion_12T" material="metal"/>',
        f'{i2}<geom name="gripper_0_collision" class="collision" type="mesh" mesh="hull_gripper_nojaws"/>',
        f'{i2}<site name="foot_0" pos="0.0743601445 0 0" size="0.0100000000" group="4"/>',
        # The foot sphere lives on its own massless, welded body. Geometrically
        # it is exactly where rocky.xml puts it -- the locomotion policy sees no
        # change -- but a body can be named in a <contact><exclude>, and a geom
        # cannot. The sphere sits on the jaw centreline at the fingertip, which
        # is precisely the volume a cube has to occupy to be grasped, so the
        # stacking scene excludes it from the cubes and lets the pads do the work.
        f'{i2}<body name="foot_0_tip">',
        f'{i2}  <geom name="foot_0_collision" class="collision" type="sphere" size="0.0100000000" '
        f'pos="0.0743601445 0 0" condim="6" priority="1"/>',
        f'{i2}</body>',
        # IK target: the point midway between the closed finger pads
        f'{i2}<site name="grasp" pos="{sum(FINGER_X)/2:.4f} {CENTRELINE_Y:.4f} 0" size="0.004" '
        f'rgba="0 1 0 0.6" group="4"/>',
        # far enough back and high enough to see past the gripper body, tilted
        # 40 deg down the approach axis so the fingers sit mid-frame
        f'{i2}<camera name="wrist" pos="0.020 {CENTRELINE_Y:.4f} 0.050" '
        f'xyaxes="0 -1 0 0.342 0 0.940" fovy="70"/>',
        jaw_block("Jaw_R", +1, "jaw_r"),
        jaw_block("Jaw_L", -1, "jaw_l"),
        f'{ind}</body>',
    ]) + "\n"
    s = s[:old.start()] + new + s[old.end():]

    # --- head camera on the carapace --------------------------------------
    s = s.replace('      <site name="imu" pos="0 0 0.0125" size="0.005" group="4"/>',
                  '      <site name="imu" pos="0 0 0.0125" size="0.005" group="4"/>\n'
                  '      <camera name="head" pos="0.020 0 0.245" '
                  'xyaxes="0 -1 0 0.707 0 0.707" fovy="68"/>')

    # The two finger pads touch each other when closed, which would generate
    # contact forces for the whole of every episode the gripper is empty.
    # They never need to collide with each other; only with what they grasp.
    s = s.replace('  <sensor>',
                  '  <contact>\n'
                  '    <exclude body1="jaw_r" body2="jaw_l"/>\n'
                  '  </contact>\n\n'
                  '  <sensor>')

    # --- actuators for the jaws -------------------------------------------
    # rack and pinion off the same STS3215: 2.9 N m through a 6 mm pinion radius
    # is ~480 N, far more than the jaws need, so the limit here is deliberate.
    s = s.replace('  <sensor>',
                  '  <actuator>\n'
                  f'    <position name="jaw_r" joint="jaw_r" kp="{JAW_KP:.4g}" kv="{JAW_KV:.4g}" '
                  f'forcerange="-{JAW_FORCE:.0f} {JAW_FORCE:.0f}" ctrlrange="0 {JAW_TRAVEL:.3f}"/>\n'
                  f'    <position name="jaw_l" joint="jaw_l" kp="{JAW_KP:.4g}" kv="{JAW_KV:.4g}" '
                  f'forcerange="-{JAW_FORCE:.0f} {JAW_FORCE:.0f}" ctrlrange="0 {JAW_TRAVEL:.3f}"/>\n'
                  '  </actuator>\n\n'
                  '  <sensor>')

    s = s.replace('<mujoco model="rocky">', '<mujoco model="rocky_manip">')
    DST.write_text(s)

    # --- rebuild the keyframe by joint NAME -------------------------------
    # The jaws are children of gripper_0, so they land in the middle of the
    # joint tree (straight after wrist_0), not at the end. Appending zeros to
    # the old qpos would silently shift every leg-1-to-4 joint by two slots.
    import mujoco
    src, dst = mujoco.MjModel.from_xml_path(str(SRC)), mujoco.MjModel.from_xml_path(str(DST))
    home = {mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_JOINT, j):
            src.key_qpos[0][src.jnt_qposadr[j]] for j in range(1, src.njnt)}
    qpos = list(src.key_qpos[0][:7])
    for j in range(1, dst.njnt):
        qpos.append(home.get(mujoco.mj_id2name(dst, mujoco.mjtObj.mjOBJ_JOINT, j), 0.0))
    assert len(qpos) == dst.nq, (len(qpos), dst.nq)
    s = re.sub(r'<key name="home" qpos="[^"]+"/>',
               '<key name="home" qpos="%s"/>' % " ".join("%.10f" % v for v in qpos), s)
    DST.write_text(s)
    print(f"wrote {DST}")
    print("  gripper base: %.1f g (was 133.1 g with jaws)" % (gM * 1000))
    for n in ("Jaw_R", "Jaw_L"):
        print("  %s: %.2f g" % (n, jaws[n][1] * 1000))
    print("  hull without jaws: %d faces" % len(hull.faces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
