"""Echolocation: reading the world off a ring of rangefinders and two touch pads.

Rocky has no eyes. This module is everything the robot is allowed to know about
where the cubes are -- slant ranges from the emitters in the gaps between its
limbs, a five-ray fan on the gripper for the last few centimetres, and whether
the finger pads are pressing on anything.

Nothing here reads a cube's true pose. The only inputs are the sensor array and
the robot's own joint encoders, which is the point: a policy trained on this can
be handed to a robot that genuinely cannot see.

The estimator is deliberately plain:

1.  Most rays hit the floor, so the floor tells you how high the emitters are.
    Take the median of `range * sin(depression)` and you have the ride height
    without needing a state estimator to tell you.
2.  A ray that comes back shorter than the floor would have is touching
    something. Convert it to a point in the body frame.
3.  Single-link cluster those points. Each cluster is an object.

A ray strikes the near top edge of a cube rather than its middle, so the raw
cluster sits about half a cube short. `CENTRE_BIAS` pushes it back in along the
line of sight; the residual error is measured in tests/test_sonar.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Push a hit this far along the line of sight to get from the surface the ray
#: touched to the middle of a 40 mm cube.
CENTRE_BIAS = 0.020
#: A return this much shorter than the floor prediction is an object.
DETECT_MARGIN = 0.020
#: Anything this close to the body axis is the robot itself.
SELF_RADIUS = 0.20
#: Ignore a hit this close to one of the robot's own limb joints, in plan view.
LIMB_RADIUS = 0.085
#: Points closer together than this belong to the same object.
CLUSTER_RADIUS = 0.070
#: A cube's own hits never span more than about 55 mm. A cluster wider than
#: this is two cubes that happened to be close enough to link, which happens
#: around 0.38 m where both of them light up the same elevation band.
SPLIT_SPREAD = 0.075


@dataclass(frozen=True)
class Detection:
    """One object, in the body frame."""

    xy: np.ndarray
    """Estimated centre, m."""
    rays: int
    """How many rays touched it. One is a rumour; four is a cube."""
    spread: float
    """Largest gap between contributing hits, m. Two cubes merged into one
    cluster show up here long before they show up in the range."""

    @property
    def range(self) -> float:
        return float(np.hypot(*self.xy))

    @property
    def bearing(self) -> float:
        return float(math.atan2(self.xy[1], self.xy[0]))


class Sonar:
    """Reads the body ring, the gripper fan and the touch pads out of MjData.

    Geometry comes from the compiled model -- emitter positions and ray
    directions are read off the sites themselves -- so the estimator cannot
    drift away from whatever `model/add_manipulation.py` last emitted.
    """

    def __init__(self, model):
        import mujoco

        self.m = model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                 for i in range(model.nsensor)]
        self.ring = [n for n in names if n.startswith("sonar_")]
        self.fan = [n for n in names if n.startswith("feel_")]
        self.pads = [n for n in names if n.startswith("touch_")]
        if not self.ring:
            raise ValueError("this model has no sonar; regenerate with add_manipulation.py")
        adr = {n: model.sensor_adr[i] for i, n in enumerate(names)}
        self.ring_adr = np.array([adr[n] for n in self.ring])
        self.fan_adr = np.array([adr[n] for n in self.fan])
        self.pad_adr = np.array([adr[n] for n in self.pads])

        def site_frame(name):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            quat = model.site_quat[sid]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            return model.site_pos[sid].copy(), R.reshape(3, 3)[:, 2].copy()

        frames = [site_frame(n) for n in self.ring]
        self.origin = np.stack([f[0] for f in frames])      # (n, 3) body frame
        self.direction = np.stack([f[1] for f in frames])   # (n, 3) body frame, unit
        #: sin of each ray's depression below the body's own horizontal
        self.sin_depression = -self.direction[:, 2]
        self.azimuth = np.arctan2(self.direction[:, 1], self.direction[:, 0])
        self.cutoff = float(model.sensor_cutoff[
            [i for i, n in enumerate(names) if n == self.ring[0]][0]] or 0.0)
        self._up_adr = adr.get("imu_upvector")
        self._gyro_adr = adr.get("imu_ang_vel")

    # -- raw ---------------------------------------------------------------
    def ranges(self, data) -> np.ndarray:
        """Slant range per ring ray, m. A miss comes back as the cutoff."""
        r = data.sensordata[self.ring_adr].copy()
        r[r < 0.0] = self.cutoff if self.cutoff > 0.0 else np.inf
        return r

    def feel(self, data) -> np.ndarray:
        """The gripper's own fan, m."""
        r = data.sensordata[self.fan_adr].copy()
        r[r < 0.0] = self.cutoff if self.cutoff > 0.0 else np.inf
        return r

    def feel_hits(self, data, base_pos, base_quat, max_range: float = 0.30) -> np.ndarray:
        """Body-frame (x, y, z) of whatever the gripper's fan is pointing at.

        The fan moves with the arm, so its pose comes from the robot's own
        forward kinematics -- here, MuJoCo's site frames, which are exactly what
        joint encoders plus a URDF give you on hardware. Nothing about the cube
        is consulted.
        """
        import mujoco

        r = self.feel(data)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(base_quat, float))
        R = R.reshape(3, 3)
        origin = np.asarray(base_pos, float)
        floor = self.floor_z(data)
        out = []
        for name, rng in zip(self.fan, r):
            if not np.isfinite(rng) or rng > max_range:
                continue
            sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
            p = data.site_xpos[sid]
            z = data.site_xmat[sid].reshape(3, 3)[:, 2]
            hit = R.T @ ((p + z * rng) - origin)
            # Anything at floor level is the floor. A cube stands 40 mm proud.
            if hit[2] > floor + 0.015:
                out.append(hit)
        return np.array(out) if out else np.zeros((0, 3))

    def touch(self, data) -> np.ndarray:
        """Normal force on each finger pad, N."""
        return data.sensordata[self.pad_adr].copy()

    def gripping(self, data, threshold: float = 0.5) -> bool:
        """Both pads loaded: there is something between the fingers.

        This is the only honest way a blind robot knows its grasp worked. One
        pad alone means it shoved the cube instead of taking it.
        """
        return bool(np.all(self.touch(data) > threshold))

    # -- estimate ----------------------------------------------------------
    #: Ring emitters sit this far above the base origin (read from the model).
    @property
    def ring_z(self) -> float:
        return float(self.origin[0, 2])

    def floor_z(self, data) -> float:
        """Where the floor is in the body frame, from the sonar and the IMU."""
        return -self.floor_plane(self.ranges(data), self.up(data))

    def up(self, data) -> np.ndarray:
        """Which way is up, in the body frame, from the IMU."""
        if self._up_adr is None:
            return np.array([0.0, 0.0, 1.0])
        u = data.sensordata[self._up_adr:self._up_adr + 3].astype(float)
        n = float(np.linalg.norm(u))
        return np.array([0.0, 0.0, 1.0]) if n < 1e-9 else u / n

    def floor_plane(self, ranges: np.ndarray, up: np.ndarray) -> float:
        """Height of the floor below the body origin, along the IMU's up, m.

        Every ray that lands on flat ground implies the same answer, so the
        median over the ring is it -- robust while fewer than half the rays are
        looking at something else, which two 40 mm cubes never manage.

        The IMU matters more here than it looks. A ray 5 deg below horizontal
        reaches the floor 1.4 m away, and 2 deg of body pitch -- which the wave
        gait produces every stride -- moves that by more than half a metre. Left
        flat, the shallow bands report a phantom object at about a metre on
        whichever side the robot happens to be leaning, and the tracker walks
        off after it. That is exactly what the first version did.
        """
        denom = self.direction @ up
        finite = np.isfinite(ranges) & (ranges > 1e-6) & (denom < -1e-3)
        if not finite.any():
            return float("nan")
        implied = -(self.origin[finite] @ up) - ranges[finite] * denom[finite]
        return float(np.median(implied))

    def emitter_height(self, ranges: np.ndarray, up=None) -> float:
        """Ride height of the emitter ring above the floor, m."""
        u = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float)
        return self.floor_plane(ranges, u) + float(self.origin[0] @ u)

    def residual(self, ranges: np.ndarray, up=None, plane: float | None = None) -> np.ndarray:
        """How much shorter each ray is than bare floor would be, m."""
        u = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float)
        c = self.floor_plane(ranges, u) if plane is None else plane
        denom = self.direction @ u
        with np.errstate(divide="ignore", invalid="ignore"):
            floor = -(c + self.origin @ u) / denom
        floor = np.where(denom < -1e-3, floor, np.inf)
        return np.clip(floor - ranges, 0.0, None)

    def hits(self, data, limbs=None) -> np.ndarray:
        """Body-frame (x, y) of every ray that touched something off the floor.

        `limbs` is where the robot's own knees and feet are, in plan view, from
        its joint encoders (`rocky.kinematics.limb_points`). Without it the ring
        spends a walking gait reporting its own swinging legs as objects: the
        emitters sit in the gaps between the limbs at rest, but a leg in swing
        travels through a gap, and every stride throws off a phantom at a third
        of a metre.
        """
        r = self.ranges(data)
        res = self.residual(r, self.up(data))
        sel = res > DETECT_MARGIN
        if not sel.any():
            return np.zeros((0, 2))
        p = self.origin[sel] + self.direction[sel] * r[sel, None]
        flat = self.direction[sel, :2]
        flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
        pts = p[:, :2] + flat * CENTRE_BIAS
        keep = np.linalg.norm(pts, axis=1) > SELF_RADIUS
        if limbs is not None and len(limbs):
            limbs = np.asarray(limbs, float)[:, :2]
            far = np.linalg.norm(pts[:, None, :] - limbs[None, :, :], axis=-1).min(axis=1)
            keep &= far > LIMB_RADIUS
        return pts[keep]

    def detect(self, data, min_rays: int = 1, split: bool = True, limbs=None) -> list[Detection]:
        """Cluster the hits into objects, nearest first."""
        pts = self.hits(data, limbs)
        groups = _cluster(pts, CLUSTER_RADIUS)
        if split:
            groups = [g for grp in groups for g in _split_wide(pts, grp)]
        out: list[Detection] = []
        for group in groups:
            if len(group) < min_rays:
                continue
            g = pts[group]
            out.append(Detection(xy=g.mean(axis=0), rays=len(group), spread=_spread(g)))
        out.sort(key=lambda dtn: dtn.range)
        return out

    def profile(self, data) -> np.ndarray:
        """Per-azimuth residual, flattened -- a compact observation for a policy.

        The raw ring is one number per ray; this keeps the shape of the array
        but expresses it as "how much nearer than the ground", which is
        invariant to ride height and therefore to the gait bobbing up and down.
        """
        return self.residual(self.ranges(data), self.up(data))


def _spread(g: np.ndarray) -> float:
    if len(g) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)))


def _split_wide(points: np.ndarray, group: list[int]) -> list[list[int]]:
    """Cut an over-wide cluster along its long axis until every piece is cube-sized."""
    g = points[group]
    if len(group) < 2 or _spread(g) <= SPLIT_SPREAD:
        return [group]
    centred = g - g.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    t = centred @ axis
    lo = [group[i] for i in range(len(group)) if t[i] <= np.median(t)]
    hi = [group[i] for i in range(len(group)) if t[i] > np.median(t)]
    if not lo or not hi:
        return [group]
    return _split_wide(points, lo) + _split_wide(points, hi)


def observation(sonar: "Sonar", data, qadr, names, belief=None) -> np.ndarray:
    """Everything a blind policy is allowed to see, as one flat vector.

    No images anywhere: sonar residuals, the gripper's fan, the touch pads, the
    joints and the IMU. Residuals rather than raw ranges, because "how much
    nearer than the ground" does not move when the gait bobs the body up and
    down, and the policy should not have to learn that it does not matter.

    `belief` is where the robot currently thinks the two cubes are, in its own
    frame, which is the one derived quantity in here. It earns its place: the
    pick-and-place lasts fifteen seconds and the cubes are under the gripper and
    inaudible for most of it, so a policy conditioned on two frames of raw sensor
    has no way to know where it was going. The robot works this out for itself
    from the feel sweep, so conditioning on it is not smuggling in ground truth
    -- it is the split between perception and control that any real stack has.

    What the belief does NOT say is which half of the job is in progress. The
    touch pads do: loaded means carrying, and descending while carrying is a
    place rather than a pick.
    """
    q = np.array([data.qpos[qadr[n]] for n in names])
    dq = np.array([data.qvel[qadr[n] - 1] for n in names])
    b = np.zeros(4) if belief is None else np.asarray(belief, float).reshape(-1)[:4]
    return np.concatenate([
        sonar.profile(data),
        np.minimum(sonar.feel(data), 0.5),
        sonar.touch(data),
        q, dq,
        data.sensordata[sonar._gyro_adr:sonar._gyro_adr + 3] if sonar._gyro_adr is not None
        else np.zeros(3),
        sonar.up(data),
        b,
    ]).astype(np.float32)


def _cluster(points: np.ndarray, radius: float) -> list[list[int]]:
    """Single-linkage clustering, small-n and obvious rather than clever."""
    n = len(points)
    seen = [False] * n
    groups: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack, group = [i], []
        seen[i] = True
        while stack:
            k = stack.pop()
            group.append(k)
            for j in range(n):
                if not seen[j] and np.hypot(*(points[j] - points[k])) <= radius:
                    seen[j] = True
                    stack.append(j)
        groups.append(group)
    return groups


@dataclass
class Pair:
    """Two objects, as the robot currently believes them, in the body frame."""

    left: np.ndarray
    right: np.ndarray
    age: float
    """Seconds since the last time the sonar actually saw them."""

    @property
    def midpoint(self) -> np.ndarray:
        return (self.left + self.right) / 2.0

    @property
    def axis(self) -> np.ndarray:
        return self.left - self.right

    @property
    def separation(self) -> float:
        return float(np.hypot(*self.axis))


class PairTracker:
    """Holds on to two objects across the moments the sonar cannot see them.

    Everything here is in the body frame and stays there. The robot never builds
    a world map, because it has nothing to build one from -- no vision, no GPS,
    no external pose. When it moves, the belief is carried along by dead
    reckoning off the IMU: rotate by the gyro, translate by the velocimeter.
    That is enough to survive the few seconds of squaring up, when the cubes
    pass behind the front limb and go silent.

    It confirms before it believes. A cube at three quarters of a metre lights up
    exactly one ray, and so does a swinging leg, a foot about to land, and the
    far edge of a clipped return. Candidates therefore accumulate evidence over
    several pings and are only promoted once they have been in the same place
    more than once -- the difference between a controller that walks to the
    cubes and one that chases its own feet in a circle, which is what the first
    version did for the whole of a 60 second episode.

    The decay is the parameter that matters. At long range a cube is not heard
    on every ping but on perhaps one in three, and evidence that halves in a
    quarter of a second never reaches the bar however long you wait: at 0.75 a
    candidate hit every third ping settles at 1.7 against a threshold of 2.5,
    and seven episodes out of twelve spent their whole time searching a room
    with two cubes plainly in it. A 1.5 second half-life accumulates
    intermittent evidence and still forgets a one-off inside two seconds.
    """

    def __init__(self, sonar: Sonar, max_age: float = 8.0,
                 accept_range: tuple[float, float] = (0.20, 1.30),
                 confirm: float = 3.0, decay: float = 0.95, gate: float = 0.09):
        self.sonar = sonar
        self.max_age = max_age
        self.accept_range = accept_range
        self.confirm = confirm
        self.decay = decay
        self.gate = gate
        self.reset()

    def reset(self) -> None:
        self._cands: list[list] = []      # [xy, score]
        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None
        self._age = 0.0
        self.seen_ever = False

    def candidates(self) -> list[np.ndarray]:
        """Confirmed objects, strongest first -- including a lone one.

        Two cubes at the far end of a spawn are 2 deg wide against a 5 deg ray
        spacing, so one of them is usually heard well before the other. Steering
        at the one you can hear is how you get close enough to hear both.
        """
        return [c[0].copy() for c in sorted(self._cands, key=lambda c: -c[1])
                if c[1] >= self.confirm]

    def update(self, data, dt: float, ang_vel_z: float, lin_vel_xy, limbs=None) -> Pair | None:
        """One ping. Dead-reckon what we hold, then fold in what we just heard."""
        for c in self._cands:
            c[0] = _carry(c[0], dt, ang_vel_z, lin_vel_xy)
            c[1] *= self.decay
        if self._left is not None:
            self._left = _carry(self._left, dt, ang_vel_z, lin_vel_xy)
            self._right = _carry(self._right, dt, ang_vel_z, lin_vel_xy)
            self._age += dt

        lo, hi = self.accept_range
        heard = False
        for det in self.sonar.detect(data, limbs=limbs):
            if not (lo <= det.range <= hi):
                continue
            heard = True
            best, bd = None, self.gate
            for c in self._cands:
                dist = float(np.hypot(*(det.xy - c[0])))
                if dist < bd:
                    best, bd = c, dist
            if best is None:
                self._cands.append([det.xy.copy(), 1.0])
            else:
                best[0] = 0.5 * best[0] + 0.5 * det.xy
                best[1] += 1.0
        self._cands = [c for c in self._cands if c[1] > 0.3][-24:]

        firm = sorted((c for c in self._cands if c[1] >= self.confirm),
                      key=lambda c: -c[1])
        if len(firm) >= 2:
            a, b = firm[0][0], firm[1][0]
            left, right = (a, b) if math.atan2(a[1], a[0]) > math.atan2(b[1], b[0]) else (b, a)
            self._left, self._right = left.copy(), right.copy()
            self.seen_ever = True
            # `age` is time since the sonar last actually heard something, not
            # time since the belief last moved: the candidates keep their
            # evidence for a second or two after the room goes quiet, and a
            # staleness clock that never ticks is no clock at all.
            if heard:
                self._age = 0.0

        if self._left is None or self._age > self.max_age:
            return None
        return Pair(self._left.copy(), self._right.copy(), self._age)


def _carry(p: np.ndarray, dt: float, wz: float, v) -> np.ndarray:
    """Move a body-frame point as the body itself moves under it."""
    c, s = math.cos(-wz * dt), math.sin(-wz * dt)
    rotated = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])
    return rotated - np.asarray(v, float)[:2] * dt
