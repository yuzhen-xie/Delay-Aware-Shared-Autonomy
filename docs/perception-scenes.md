# Scene geometry, perception and deterministic layouts

Two sources of geometry are implemented. Synthetic task-space runs use known
scene coordinates. Recorded RGB-D frames can instead be processed by the
calibrated color-and-geometry pipeline below. No RGB-D recordings, measured
object dimensions or camera calibration from a physical session are bundled.
The supplied numerical geometry is a declared benchmark design, not a recovered
measurement of an earlier experiment.

## Coordinate and object contract

All controller positions use the right-handed robot base frame and metres.
`Scene.goals[3,3]` contains three ordered candidate grasp positions; an inferred
goal is a categorical index in `{0,1,2}`. `place[3]` is the desired placement
position and `start[3]` is the initial TCP position. Orientation is held by the
robot interface. This benchmark does not learn a six-dimensional grasp pose.

`centers[J,3]` contains obstacle **centers** `O_j`, rather than sampled surfaces.
`radii[J]` contains positive bounding-sphere radii. The Gaussian potential field
uses center displacement. Collision evaluation subtracts both obstacle radius
and TCP radius: `||p-O_j|| - r_j - r_TCP`. The optional surface-distance risk
configuration changes the risk distance explicitly; it does not change the
meaning of `O_j`. Non-spherical physical obstacles require conservatively
measured enclosing spheres or an independently implemented geometric model.

Default benchmark grasp positions are `[.56,-.25,.12]`, `[.64,0,.12]` and
`[.55,.24,.12]` m. Placement is `[.80,.29,.12]` m, initial TCP position
`[.16,-.10,.46]` m, and nominal lift height `.48` m. Initial obstacle centers
are `[.43,0,.23]` and `[.35,-.28,.19]` m, with radii `.075` and `.060` m.
The current resolved YAML is authoritative if these values are overridden.

The manifest's nominal object envelope is an axis-aligned `40 x 40 x 60` mm
box centered on each candidate grasp position, configurable using
`scene.object_size_m`. Its half-diagonal supplies a conservative static
separation radius. This envelope is a proposed layout constraint. Runtime
simulation checks a `25` mm TCP sphere; carried-object and robot-link collisions
are not modeled by that runtime check. Physical geometry and grasp transforms
must be measured before using a layout on a robot.

## RGB-D method and required inputs

The implemented detector is `registered_rgbd_color_geometry_v1`. It selects the
largest 4-connected component inside a declared inclusive RGB interval for each
uniquely colored target/obstacle. This is a controlled-laboratory detector for
colored objects or markers. It is not a learned open-world detector and does not
infer unseen obstacles. Choose disjoint color intervals using calibration images
under the actual illumination, freeze them before evaluation, and retain those
images and thresholds in the local session record. Overlapping detections,
missing objects and inadequate depth cause an error rather than guessed geometry.

RGB must be `H x W x 3` on the `0..255` **RGB** scale, with depth `H x W`
registered to the same undistorted camera image. Convert a BGR camera buffer
before import. The camera SDK must supply synchronized acquisition timestamps,
registration and distortion removal; this module does not assume arbitrary raw
sensor streams are already aligned. Intrinsics must correspond to the processed
image resolution. `depth_scale_m=.001` converts millimetre depth units to metres;
use `1.0` for a depth image already in metres. Zero, nonfinite, out-of-range and
depth outlier samples are discarded.

For valid pixel `(u,v)` and optical-axis depth `Z`, deprojection is
`X=(u-cx)*Z/fx`, `Y=(v-cy)*Z/fy`, `Z=Z`. The camera optical frame uses
`+Z` forward, `+X` along increasing image column and `+Y` along increasing image
row. After selecting the color component, the depth rule retains samples within
`max(.010 m, 3*1.4826*MAD)` of its median depth. Defaults require at least
30 retained samples and depths in `[.1,3.0]` m. These thresholds are explicit
proposed acquisition defaults, not measured sensor error bounds.

There are two supported center models:

1. `geometry: marker`: deproject the median retained image coordinate and depth,
   then add the explicitly calibrated `center_offset_camera_m`. A visible surface
   point becomes a geometric center only when this offset is known. The default
   zero offset identifies the visible marker point as the goal. For an oriented
   object, provide the correct camera-frame offset for that pose; a fixed offset
   does not estimate changing object orientation.
2. `geometry: sphere`: fit the known `radius_m` to the retained 3D surface points
   using robust nonlinear least squares on `||point-center||-radius`, with a
   `soft_l1` loss. The fitted center must be behind the visible surface, have a
   full-rank local Jacobian and meet the default `5` mm RMS residual threshold.
   The threshold is an admission criterion, not an uncertainty guarantee. Radius
   must come from a measured object or a declared synthetic object. A visible
   sphere's front-surface depth is never reported directly as its center.

Camera-to-base calibration uses `robotics.calibration.rigid_registration` with
at least three noncollinear matching landmarks. Save its SVD rigid transform,
landmark residuals, camera identifier, calibration date and independent check
point errors. The transform convention is `p_base = R_base_camera*p_camera + t`.
`detect_scene` checks a proper rotation and homogeneous bottom row; it cannot
independently verify an incorrect calibration record. Detection outputs preserve
the full transform, intrinsics, calibration ID, source timestamp, sequence,
segmentation thresholds, depth counts and sphere-fit residuals.

## File processing example

Save a recorded aligned frame without pickled objects:

```python
import numpy as np
np.savez_compressed("frame.npz", rgb=rgb_array, depth=registered_depth_array)
```

The JSON specification has this structure. Replace intrinsics, transform, colors,
dimensions and timestamps with the acquisition record; the numbers below only
illustrate the schema.

```json
{
  "intrinsics": {"width": 640, "height": 480, "fx": 600.0, "fy": 600.0,
                 "cx": 320.0, "cy": 240.0},
  "camera_to_base": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "calibration_id": "REPLACE_WITH_ACTUAL_CALIBRATION_ID",
  "timestamp_s": 1.0,
  "sequence": 1,
  "update_mode": "snapshot",
  "depth_scale_m": 0.001,
  "objects": [
    {"object_id":"goal_red", "role":"goal", "geometry":"marker",
     "rgb_lower":[180,0,0], "rgb_upper":[255,70,70],
     "center_offset_camera_m":[0,0,0]},
    {"object_id":"goal_green", "role":"goal", "geometry":"marker",
     "rgb_lower":[0,180,0], "rgb_upper":[70,255,70]},
    {"object_id":"goal_blue", "role":"goal", "geometry":"marker",
     "rgb_lower":[0,0,180], "rgb_upper":[70,70,255]},
    {"object_id":"obstacle_yellow", "role":"obstacle", "geometry":"sphere",
     "rgb_lower":[180,180,0], "rgb_upper":[255,255,70], "radius_m":0.06}
  ],
  "scene": {"goal_ids":["goal_red","goal_green","goal_blue"],
            "place":[0.8,0.29,0.12], "start":[0.16,-0.10,0.46]}
}
```

The optional `scene` section specifies the categorical goal order and the
separately calibrated placement/start positions. Processing is available from
the Python interface:

```python
from dasa.robotics.perception import perceive_file
perceive_file("frame.npz", "perception.json", "snapshot.json")
```

The output contains all detections and, when `scene` is present, a nested
base-frame `Scene` with provenance `calibrated_rgbd_geometry`. It is a geometry
record, not evidence that a robot experiment or camera evaluation has occurred.

## Snapshot and online update policies

Existing simulations use a complete known scene fixed for the trial. The existing
device session loads a scene before execution; it does not autonomously open a
camera or refresh obstacle geometry. A physical static-scene protocol should
capture and validate a complete RGB-D snapshot immediately before each trial,
retain its ID in the trial log, and stop if an object moves outside that assumed
static layout. Placement and start coordinates require their own recorded
calibration or perception.

For an explicitly integrated online camera loop, use `PerceptionStream` with
`update_mode="online"` detections. A whole scene is replaced at one acquisition
timestamp. Sequence and acquisition time must both strictly increase;
calibration ID cannot change mid-stream. The default maximum age is `.2` s,
configurable explicitly; stale/future snapshots are rejected. The application
must use a common monotonic time base or document a clock transform, call the
camera pipeline at its measured acquisition rate, and stop or enter a validated
hold on an exception. The stream supplies validated geometry to such a loop; it
does not assert that camera acquisition meets the control servo deadline.

The potential field assumes all relevant modeled goals and obstacles are known
at its current snapshot. Occluded, unsegmented or out-of-view objects violate
that assumption. Record missed detections and age overruns as acquisition
failures; do not silently reuse an unlimited-age scene. A more general detector
can supply the same center/radius contract, but needs separate validation of
detection recall, center error, latency and uncertainty inflation.

## Frozen scene manifests

`scenes.generate_manifest` wraps the existing randomization function with explicit
rejection constraints and records the accepted coordinates. It does not alter
legacy runs retrospectively. New constrained and legacy sampling have distinct
interpretations; analyze them using their recorded generation configuration.

```python
from dasa.config import load_config
from dasa.scenes import write_scene_manifest, load_scene_manifest

config = load_config("configs/default.yaml")
write_scene_manifest(config, "layouts.json", count=100,
                     scenario="extra_obstacles", seed=20261020)
manifest = load_scene_manifest("layouts.json")
```

`generator_version="constrained_translation_v1"` uses a master seed, scene index
and attempt number to construct `numpy.random.SeedSequence([seed,index,attempt])`.
The resulting unsigned 64-bit `sampling_seed` starts a fresh generator for each
attempt. Scene indices increase first; attempts increase within each index.
The legacy sampler consumes randomness in this order: common XYZ translation,
per-goal XY shift magnitudes, per-goal XY shift signs. Added obstacle offsets are
fixed, then receive the same common translation. NumPy version is preserved.

| Scenario | Sampling before constraints |
|---|---|
| `demonstrated` | Common X/Y translation uniform in `[-.025,.025]` m; common Z translation uniform in `[-.00625,.00625]` m. |
| `novel` | Same common translation; independent signed per-goal X/Y offsets with magnitudes uniform in `[.05,.15]` m; one extra sphere at `[.49,-.12,.31]` m plus common translation. |
| `wider` | Novel per-axis target offsets multiplied by `1.6`, giving `[.08,.24]` m magnitudes before clipping. |
| `extra_obstacles` | Novel target offsets and three added centers at `[.49,-.12,.31]`, `[.68,.14,.34]`, `[.29,.12,.28]` m plus common translation. |

All non-demonstrated target X/Y coordinates are clipped to `[.30,.78]` and
`[-.42,.42]` m. The stated ranges are **per-axis**, not Euclidean displacement
ranges; clipping and rejection change the final distribution. Added obstacle
radii default to `.055` m. Actual coordinates in the manifest, rather than an
unqualified “5–15 cm perturbation”, define an evaluated layout.

The constraints check finite base-frame geometry; task-object and obstacle
envelopes inside the configured Cartesian workspace; pairwise target separation;
nonintersecting target/obstacle and obstacle/obstacle spheres; task points within
a declared `1.1` m base-centered radius; and collision-free nominal lifted TCP
waypoint segments with a `5` mm separation margin. Every target is checked.
The base-radius test is a simple workspace screen, not an inverse-kinematics or
whole-arm reachability certificate. Runtime policy trajectories can still collide
despite the existence of the nominal corridor. A physical session additionally
needs joint limits, calibrated kinematics, full collision geometry, grasp and
payload checks.

Each entry records `scene_id`, accepted layout index, accepted attempt,
`sampling_seed`, every attempted seed with rejection reasons, and all scene
coordinates. The manifest includes object dimensions, complete geometry-relevant
configuration, constraints, coordinate units, generator version and SHA-256
integrity hashes. `load_scene_manifest` checks the hashes and revalidates every
accepted scene. Hashes detect accidental modification; they do not authenticate
an experiment. Replay the saved coordinates for matched condition comparisons
rather than independently resampling layouts for each condition.
