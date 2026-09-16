# Blind: stacking by sound and touch

Rocky has no eyes. Eridians perceive shape by sound — it is why he could never
see the stars, and why everything he and Grace exchanged had to be handed over
physically rather than pointed at. Taking that seriously means the robot gets no
camera at all, and with it goes the case for a vision-language policy: there is
no image to condition on. What is left is a ring of rangefinders standing in for
the sonar, the joint encoders, an IMU, and two touch pads.

`scripts/blind_stack_demo.py` runs the whole task on that and nothing else. No
part of it reads a cube's true pose.

## Where the emitters had to go

This was decided entirely by the body plan, and it took three tries.

**Over the body axis.** The elevation needed to see something at arm's length is
steep enough that the ray hits the robot's own shoulder on the way out. Measured:
a 20° cone straight ahead came back at 0.11 m — the coxa — in *every* gait phase,
including the one where the front limb is airborne, because the coxa never leaves
that azimuth.

**On the carapace rim.** Clears the coxa, and buries the emitters inside the
femur instead.

**In the gaps.** What is actually true is that a limb blocks about 12° either
side of its own azimuth and no mounting changes that — the same problem a real
legged robot has with a body lidar. So the emitters live in the five gaps between
the limbs: 45 bearings, 5° apart, 40° of coverage per gap, six elevation bands
each. 270 rays, and over an empty floor every one of them reads clean.

The bands are a ladder in range. With the emitter ~120 mm above the floor and the
target *d* further out, a 40 mm cube shows up when tan(depression) falls between
(0.12−0.04)/d and 0.12/d, so 5°, 8°, 12°, 17°, 23° and 30° cover roughly 1.3,
0.85, 0.55, 0.40, 0.31 and 0.25 m from the body axis.

## Why the robot walks sideways

It follows from the gaps. Facing the cubes squarely puts them behind the front
limb, where nothing can be heard. So the approach controller holds them **36° off
the body axis** — the gap centre — and closes crabwise, which the wave gait does
perfectly well since it has lateral velocity. Inside 0.55 m it drops the crab and
squares up for the arm, the cubes go quiet, and the belief coasts on IMU dead
reckoning until the gripper takes over.

That is not a trick to make the demo work. It is what this body plan has to do to
watch where it is going, and it is the single most alien-looking thing the robot
does.

## The estimator

`src/rocky/sonar.py`, and deliberately plain:

1. Most rays hit the floor, so **the floor tells you how high you are**. Every
   ray that lands on flat ground implies the same plane offset; the median over
   the ring is it. No state estimator required.
2. A ray that comes back **shorter than the floor would be** is touching
   something. Convert it to a point in the body frame.
3. **Single-link cluster** those points. Each cluster is an object, and a cluster
   wider than 75 mm is two cubes that happened to link, so it gets split.

Three things this got wrong first, each of which cost an episode to find:

**The IMU is not optional.** A ray 5° below horizontal reaches the floor 1.4 m
away, and 2° of body pitch — which the wave gait produces every stride — moves
that by more than half a metre. Predicting the floor as if the robot were level
makes the shallow bands report a phantom object at about a metre on whichever
side it happens to be leaning. The tracker locked onto it and walked away from
the cubes for the entire episode. The floor plane is now fitted against the IMU's
own up-vector.

**A swinging leg goes through the gap.** The emitters are clear at the neutral
stance, which is what the first check tested, but a limb in swing travels right
across the beams. The robot knows where its own limbs are — `limb_points` takes
the joint encoders and returns every knee and foot — and hits within 85 mm of one
are discarded.

**One ray is a rumour.** A cube at three quarters of a metre lights up exactly
one ray, and so does a foot about to land. `PairTracker` accumulates evidence
across pings and only promotes a candidate once it has been in the same place
more than once. The decay on that evidence is the parameter that matters: at
long range a cube is heard on perhaps one ping in three, and evidence that halves
in a quarter of a second never reaches the bar however long you wait — a
candidate hit every third ping settles at 1.7 against a threshold of 2.5, and
seven episodes out of twelve spent their entire time searching a room with two
cubes plainly in it.

Measured, with the robot standing still: bearing RMS **0.76°** (worst 1.5°, which
is 12 mm at 0.45 m), range error under 25 mm, and both cubes resolved as a pair in
16 of 16 layouts from 0.32 to 0.60 m.

## Searching

The sonar reaches about 1.3 m and the robot spawns up to 1.2 m from the cubes, so
the first thing it usually has to do is find them. Two attempts at that failed
for reasons worth writing down:

- **Turning and creeping at once** traces a circle of radius v/ω, which at these
  rates is 60 mm. The robot appears to be searching and is standing still.
- **Alternating turn and walk legs** is a random walk: the walking leg goes
  wherever the last turn happened to leave it pointing, and over 35 s it covers
  almost no ground.

What works is cruising straight ahead while yawing back and forth — closing the
range and sweeping the gaps across the room at the same time. With that, 11 of 12
spawns acquire both cubes within 10 to 20 seconds. If only one is ever heard, the
robot walks up to that one anyway after 16 seconds and lets the hand sort out
what is actually there.

## Feeling for the grasp

The ring gets the robot parked. It cannot do the grasp: squared up, the cubes are
in the front limb's shadow.

So the arm becomes the sensor. Five more rangefinders fan out from the gripper —
from **beyond** the fingertips, because fired from inside the hand every one of
them lands on the robot's own foot sphere at 38 mm and never sees anything else.
The brace lifts leg 0, the arm sweeps 60° across the front at about 26°/s, and
hits that stand proud of the sonar-estimated floor are clustered into two cubes.

Measured against ground truth: **4–6 mm**, which is better than the ring by an
order of magnitude and good enough to grasp on.

## Knowing whether it worked

A blind robot's only way of telling whether it actually picked the cube up is the
pads. Both loaded means something is between the fingers; one alone means it
shoved the cube instead of taking it. The demonstrator checks halfway through the
lift, and on a miss it re-runs the feel sweep and tries again rather than
carefully stacking nothing on nothing.

That check is also why this task is worth training a policy on. The scripted
version knows the cube is there because it just measured it. A learned one has to
decide, from the same signals, whether to close now or feel around some more.

## What a policy sees

No images anywhere:

| | size | |
|---|---|---|
| sonar ring residual | 270 | metres nearer than bare floor, per ray |
| gripper fan | 5 | metres |
| touch | 2 | newtons per pad |
| joint positions | 18 | rad, m for the jaws |
| joint velocities | 18 | |
| IMU | 6 | gyro and up-vector |
| cube belief | 4 | body-frame xy of each cube |

323 floats — smaller than a single 16×16 image patch, and every one of them is
something the hardware would actually have.

The cube belief is the one derived quantity, and it earns its place. The
pick-and-place lasts fifteen seconds, and for most of it the cubes are under the
gripper and inaudible; a policy conditioned on two frames of raw sensor has no
way to know where it was going. The robot works the belief out for itself from
the feel sweep, so conditioning on it is not smuggling in ground truth — it is
the split between perception and control that any real stack has.

What the belief does *not* say is which half of the job is in progress. The touch
pads do: loaded means carrying, and descending while carrying is a place rather
than a pick. That is the whole disambiguation, and it is a sensor reading.

## The policy

`scripts/train_diffusion.py` is Diffusion Policy (Chi et al. 2023) at its
smallest useful size: a DDPM over 8-step action chunks conditioned on the last
two observations, cosine schedule, 50 denoising steps. 323 floats in and no
images means an MLP rather than a ResNet, and it trains on a CPU.

- **Chunks**, because the demonstrator's action depends on where it is in a
  fifteen-second sequence, and "descending onto the cube" looks much like
  "lowering it onto the other one" from a single frame.
- **Diffusion**, because the interesting moments are multimodal: when the pads
  say the grasp missed, the demonstrator sometimes nudges and closes again and
  sometimes lifts off and re-sweeps, and averaging those gives a motion that does
  neither.

Two things that were wrong before any number from it meant anything:

**The sampler.** The textbook DDPM step divides by √α, which is a factor of
thirty at the noisy end of a 50-step cosine schedule, and any error in the
predicted noise is amplified by it. Sampled chunks came out about two radians
from anything the demonstrator ever did, off a model whose training loss was
perfectly healthy. Reconstructing x₀, clamping it to the range normalised actions
live in, and taking the posterior mean fixed it — 1914 mrad to 12.

**The split.** Windows overlap by all but one frame, so holding out random
windows scores the model on data it has all but memorised. Whole episodes are
held out instead. Runs are also cut at every discontinuity rather than only at
episode boundaries, because a failed grasp sends the demonstrator back to the
feel sweep and a window straddling that gap would teach a jump that never
happened.

`scripts/eval_diffusion.py` rolls it out. Perception — brace, lift, sweep — stays
scripted, because it is a sensing routine rather than a skill; the policy
replaces the fifteen seconds after it, on a receding horizon that executes four
steps of each predicted chunk and re-plans.

### Results

320 episodes generated across two workers, randomised wider than the success
envelope (cubes 100–165 mm apart, ±18° of bearing, 320–400 mm out, robot parked
anywhere in a 70 mm band, 5 mrad of jitter on every arm target). **283 of 320
stacked**, and those 283 are the training set: 81,787 samples, 53,487 windows
after filtering to the `stack` phase.

Open loop, 120 epochs, 28 episodes held out entirely:

| | median | p90 |
|---|---|---|
| training chunk error | 0.22 mrad | 0.013 rad |
| held-out chunk error | 0.23 mrad | 0.013 rad |

Identical on data it has never seen, which is what 283 episodes buys over the 11
this started with.

Closed loop, 30 fresh episodes with the policy driving:

| | |
|---|---|
| red cube on top and fully supported | **19/30 (63%)** |
| inside the strict 14 mm the script is scored by | 12/30 (40%) |
| offset when it lands | median 13.3 mm, worst 18.6 mm |
| the scripted demonstrator, same test | 283/320 (88%) |

So it transfers, and it is worse than what it was cloned from — which is the
normal state of affairs for behaviour cloning and worth saying plainly rather
than quoting the open-loop number and moving on. The gap is about 3 mm of
placement accuracy and a chunk of outright failures.

One thing the failures are *not*: indecision at the gripper. Logging the jaw
command through failed rollouts, it swings cleanly between fully open and fully
closed and spends under a fifth of its time anywhere in between, and peak pad
force reaches the full 20 N even in episodes that end badly. The policy grips.
What it does with the cube afterwards is where it loses them.
