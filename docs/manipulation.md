# Manipulation: brace, pick, stack

Rocky has five identical limbs and one of them, leg 0, carries a gripper. Walking,
it is a leg. Manipulating, the other four brace and it becomes a 4-DOF arm with a
parallel jaw. This is the note on how that switch works, what the arm can reach
once it has happened, and everything the geometry does not let it do.

## The handoff

Two controllers, one baton:

| phase | controller | what moves |
|---|---|---|
| `walk` | locomotion (`rocky.approach` + a wave gait or the trained policy) | all five limbs |
| `settle` | neither | nothing; the stance is held |
| `stack` | `rocky.stack.StackController` | leg 0 only; the other four hold the brace |

`rocky.approach.Approach` turns "the cubes are over there" into the `(vx, vy, wz)`
twist the locomotion controller already consumes, so the trained policy drops into
the same slot the scripted gait occupies in `scripts/stack_demo.py`.

It drives to a **pose**, not a range and bearing. Parking anywhere on a circle
355 mm from the pair's midpoint satisfies "close enough, facing them", and some of
those parks leave one cube at 357 mm and the other at 385 mm, the far one right on
the edge of what the arm can place at. Standing on the perpendicular bisector of
the two cubes makes the two ranges equal by construction. That one change took the
scripted stack from 2-of-4 to 20-of-20.

## The arm

`rocky.arm` is the kinematics. The chain, in leg 0's frame:

    sweep (about leg Z)  ->  coxa   42   mm
    lift  (about leg Y)  ->  femur 130   mm
    elbow (about leg Y)  ->  tibia  90   mm to the wrist axis
    wrist (about leg Y)  ->         67.5 mm to the point between the closed pads

`forward` is checked against the compiled MJCF's `grasp` site to 3e-12 m
(`tests/test_arm.py`), so the analytic chain and the model cannot drift apart.

Lift, elbow and wrist share an axis, so the arm is planar inside the swept vertical
plane: two joints place the grasp point and the third sets the **approach angle**,
the direction the gripper's own +X points, positive up. −90° is straight down.

The wrist's ±75° range is what makes that angle interesting. A straight-down grasp
is available out to about 360 mm from the body axis and then runs out, so
`arm.inverse` with no approach given scans the angles the joints will actually hold
at that target and takes the steepest one with 4° of clearance at every stop. Near
the body that is −90°; at 400 mm it has slackened to about −35°.

`StackController` interpolates the approach angle along each segment the same way it
interpolates the target, and caches the angle **per waypoint** so a point that ends
one segment and starts the next gets the same answer. Without that the wrist jumps
30° at the seams.

## What the geometry costs

Four things about this robot make the obvious version of this script fail, and all
four are visible in the code as named constants:

**The gripper is also a foot.** `foot_0_collision` is a 10 mm sphere at the
fingertip, on the jaw centreline — exactly the volume a cube has to occupy to be
grasped. It sits on its own massless welded body, `foot_0_tip`, purely so that
`model/rocky_cubes.xml` can name it in a `<contact><exclude>`: MuJoCo can exclude
bodies, not geoms. The gripper hull and the jaw pads still collide with the cubes.

**The gripper body stops 17 mm short of the pads.** Aim a top-down grasp at the
centre of a 40 mm cube and the hull buries itself 3 mm into the cube's top face; the
descent stalls there and the jaws close on nothing. `StackPlan.grasp_rise` puts the
grasp point 8 mm above the cube centre, which lands the pads across the upper half
of the face and leaves the hull 5 mm clear. The same rise is added at the place, so
the cube still ends up where it was asked to go.

**Widening leg 0 with the rest walks it into the cubes.** The brace spreads the four
stance legs to 1.18× the walking footprint; leg 0 goes the other way, to 0.88×
(`arm.BracePose.arm_tuck`), and lifts from there. The first version widened all five
and sent the red cube a metre downrange.

**The shoulder boot and the femur sleeve are a bearing, not a collision pair.** The
real meshes clear each other by 0.05–0.10 mm at every sweep angle. Their convex
hulls bridge that gap and interpenetrate by up to 15 mm, which MuJoCo resolves with
contact forces over 140 N the moment a leg sweeps off centre. Walking barely noticed
— its sweep amplitude is about 10° — but the arm reaching for a cube jammed solid,
with both servos saturated at 2.9 N·m and the gripper stuck 34 mm above the target.
`model/build_model.py` now emits `<exclude body1="base" body2="femur_k"/>` for all
five legs.

## Servo sag

The jaws are driven through a 2 mm-lead screw, so the rotor's inertia reaches the jaw
as a *reflected mass*: `I_rotor × (2π/lead)² = 0.494 kg`. The first cut used an
armature of 0.002 kg and a gain of 2500 N/m, which puts the servo's natural frequency
at 625 rad/s against a 5 ms timestep — it integrated itself straight through the joint
limits and the jaws ended up 64 mm outside their travel. The gains now come from the
same 10 Hz, ζ=2 target the leg servos use.

The leg servos are position sources with finite stiffness, so holding the arm out at a
third of a metre leaves a standing error of τ/kp — about 2.6° of sweep under the load
of the arm plus a cube, which is 16 mm of placement error at that radius and enough to
leave the red cube teetering on a corner. `StackController` integrates the encoder
error back into its command, which is what the servo bus would do on hardware.

Two guards, both learned the hard way:

- It only integrates while the error is under 5° (`trim_band`). A servo pressing the
  cube it is carrying into the cube it is stacking on is tens of degrees behind and
  going nowhere; integrating that drove the gripper 26 mm past the target and toppled
  the stack it had just built.
- It bleeds away during the joint-space fold and unfold. Several degrees of standing
  sweep offset carried into the stow walks the gripper sideways into the stack.

## The working envelope

Measured by running the scripted sequence across the workspace with the robot already
parked (`dz` within 6 mm of 40, lateral offset under 14 mm):

| cube-pair range | 80 mm apart | 115 mm apart | 150 mm apart |
|---|---|---|---|
| 300 mm | fail | fail | fail |
| 320 mm | fail | ok | ok |
| 340 mm | fail | ok | ok |
| 360 mm | ok | ok | ok |
| 380 mm | ok | ok | ok |

and by bearing, at 115 mm separation: ±15° works at 360 mm, −15° to 0° at 320 mm,
±25° fails at both. Inside 320 mm the arm folds far enough to bring the gripper body
back over the cubes; below about 110 mm of separation the gripper straddling one cube
fouls the other.

So the approach controller parks at 355 mm with the pair square to the robot, which
puts both cubes at ~368 mm and about ±10° — comfortably inside the band on every
axis.

## Running it

```bash
python scripts/stack_demo.py --video docs/rocky_stack.mp4   # one episode, filmed
python scripts/stack_demo.py --episodes 20                  # success rate
python scripts/stack_demo.py --episodes 50 --dataset data/stack.npz
```

`--dataset` writes what a VLA needs for the manipulation half: the head and wrist
camera views, the full joint state, the action the script issued and the phase label,
at 20 Hz, for the `stack` segment only. The walk is not in it — the locomotion policy
already handles that, and the handoff is the point where the VLA takes over.

## Models

- `model/rocky.xml` — locomotion. Unchanged except for the base/femur excludes.
- `model/rocky_manip.xml` — derived by `model/add_manipulation.py`: jaws with slide
  joints and their two actuators, the `grasp` site, the `head` and `wrist` cameras,
  and `foot_0_tip`. 25 qpos, 19 joints. The shared joints and all five foot sites are
  identical to `rocky.xml`, so a locomotion checkpoint runs on it unchanged.
- `model/rocky_cubes.xml` — the stacking scene: floor, two 40 mm / 35 g cubes, a
  `scene` camera, the 16 walking servos, and the cube contact excludes.
