"""The MJCF is generated from CAD; `model_params.py` is typed by hand.

These tests are the contract between them: if anyone edits one without the
other, or regenerates the model from a changed Fusion design, this fails.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rocky import ROCKY_STANDALONE_XML, ROCKY_URDF, ROCKY_XML  # noqa: E402
from rocky import model_params as P  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(ROCKY_XML))


@pytest.fixture(scope="module")
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    return d


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def test_joint_set_and_order(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert names[0] == "floating_base_joint"
    assert tuple(names[1:]) == P.JOINT_NAMES


def test_joint_ranges(model):
    for k in range(P.N_LEGS):
        assert np.allclose(model.jnt_range[_jid(model, f"sweep_{k}")], P.SWEEP_RANGE, rtol=0, atol=1e-9)
        assert np.allclose(model.jnt_range[_jid(model, f"lift_{k}")], P.LIFT_RANGE, rtol=0, atol=1e-9)
        assert np.allclose(model.jnt_range[_jid(model, f"elbow_{k}")], P.elbow_range(k), rtol=0, atol=1e-9)
    assert np.allclose(model.jnt_range[_jid(model, "wrist_0")], P.WRIST_RANGE, rtol=0, atol=1e-9)


def test_link_offsets_and_mount_angles(model):
    for k in range(P.N_LEGS):
        coxa = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{k}")
        femur = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"femur_{k}")
        tibia = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{k}")
        assert np.allclose(model.body_pos[coxa], P.mount_position(k), rtol=0, atol=1e-9)
        assert np.allclose(model.body_pos[femur], [P.COXA_LEN, 0, 0], rtol=0, atol=1e-9)
        assert np.allclose(model.body_pos[tibia], [P.FEMUR_LEN, 0, 0], rtol=0, atol=1e-9)
        # mount yaw
        q = model.body_quat[coxa]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        assert math.cos(yaw - P.MOUNT_ANGLES[k]) > 1 - 1e-9
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_0")
    assert np.allclose(model.body_pos[grip], [P.WRIST_X, 0, 0], rtol=0, atol=1e-9)


def test_joint_axes(model):
    for k in range(P.N_LEGS):
        assert np.allclose(model.jnt_axis[_jid(model, f"sweep_{k}")], [0, 0, 1], atol=1e-9)
        assert np.allclose(model.jnt_axis[_jid(model, f"lift_{k}")], [0, 1, 0], atol=1e-9)
        assert np.allclose(model.jnt_axis[_jid(model, f"elbow_{k}")], [0, 1, 0], atol=1e-9)


def test_total_mass(model):
    assert model.body_subtreemass[1] == pytest.approx(P.TOTAL_MASS, abs=0.002)


def test_foot_spheres(model):
    for k in range(P.N_LEGS):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{k}_collision")
        assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        assert model.geom_size[g, 0] == pytest.approx(P.FOOT_RADIUS[k], abs=1e-9)
        assert model.geom_condim[g] == 6 and model.geom_priority[g] == 1
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{k}")
        assert np.allclose(model.site_pos[s], model.geom_pos[g], atol=1e-9)


def test_actuator_constants(model):
    assert model.dof_armature[6] == pytest.approx(P.ARMATURE, rel=1e-6)
    for k in range(P.N_LEGS):
        j = _jid(model, f"lift_{k}")
        assert model.jnt_actfrclimited[j]
        assert np.allclose(model.jnt_actfrcrange[j], [-P.EFFORT_LIMIT, P.EFFORT_LIMIT], atol=1e-6)


def test_home_keyframe_is_the_cad_stance(model, data):
    assert model.nkey == 1
    for k in range(P.N_LEGS):
        assert data.qpos[model.jnt_qposadr[_jid(model, f"lift_{k}")]] == pytest.approx(P.HOME_LIFT, abs=1e-9)
        assert data.qpos[model.jnt_qposadr[_jid(model, f"elbow_{k}")]] == pytest.approx(P.HOME_ELBOW, abs=1e-9)
    assert data.qpos[2] == pytest.approx(P.BASE_HEIGHT, abs=1e-9)


def test_feet_are_level_and_just_touching(model, data):
    zs = []
    for k in range(P.N_LEGS):
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{k}")
        zs.append(data.site_xpos[s][2] - P.FOOT_RADIUS[k])
    assert max(zs) - min(zs) < 1e-4, "feet not level in the home pose"
    assert 0.0 <= min(zs) < 1e-3, "feet should start just above the ground"


def test_no_self_collision_in_home_pose(model, data):
    assert data.ncon == 0


def test_sensors_the_rl_task_requires(model):
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)}
    assert {"imu_ang_vel", "imu_lin_vel", "root_angmom"} <= names


def test_urdf_matches_mjcf(model, data):
    """The URDF is a second export of the same robot; it must agree exactly."""
    mu = mujoco.MjModel.from_xml_path(str(ROCKY_URDF))
    du = mujoco.MjData(mu)
    qmap = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): data.qpos[model.jnt_qposadr[i]]
        for i in range(1, model.njnt)
    }
    for i in range(mu.njnt):
        du.qpos[mu.jnt_qposadr[i]] = qmap[mujoco.mj_id2name(mu, mujoco.mjtObj.mjOBJ_JOINT, i)]
    mujoco.mj_forward(mu, du)
    base = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")]
    for bi in range(1, mu.nbody):
        name = mujoco.mj_id2name(mu, mujoco.mjtObj.mjOBJ_BODY, bi)
        bj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.allclose(du.xpos[bi], data.xpos[bj] - base, rtol=0, atol=1e-9), name
        assert mu.body_mass[bi] == pytest.approx(model.body_mass[bj], abs=1e-12), name
        assert np.allclose(mu.body_inertia[bi], model.body_inertia[bj], rtol=0, atol=1e-12), name


def test_it_stands_up(tmp_path):
    """Two seconds of physics in the CAD pose: upright, five feet down."""
    m = mujoco.MjModel.from_xml_path(str(ROCKY_STANDALONE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = d.qpos[7:7 + m.nu]
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    up = d.xmat[1].reshape(3, 3)[:, 2]
    assert math.degrees(math.acos(np.clip(up[2], -1, 1))) < 1.0
    touching = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        for c in d.contact[: d.ncon]
    } | {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        for c in d.contact[: d.ncon]
    }
    assert {f"foot_{k}_collision" for k in range(P.N_LEGS)} <= touching
    assert not any("base_collision" in n for n in touching if n)
    assert abs(d.qpos[2] - P.BASE_HEIGHT) < 0.01, "sagged more than 10 mm"
