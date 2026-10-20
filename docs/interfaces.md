# Signal and execution contracts

All Cartesian quantities use the calibrated robot base frame. Positions are in
metres, velocities in metres per second, forces in newtons, and timestamps in seconds.

Demonstration NPZ files contain `time[T]`, `position[T,3]`, `velocity[T,3]`,
`raw_force[T,3]`, `observed_state[T,6]`, `phase[T]`, and a JSON scalar `metadata`.
Metadata contains schema version, episode/operator identifiers, scene geometry,
intended goal, provenance and independent outcome annotations. Array loading
disables pickle. The phases are pickup, lift, transport/place and complete.

Partition files list relative demonstration paths in train/validation/test arrays.
Operators and episodes cannot overlap partitions. Refinement combines admitted new
demonstrations with original samples using explicit positive weights. Admission
requires correct-object placement, no collision/bounds violation, valid timestamps
and completion inside the configured task budget. Requested and accepted counts
are separate outputs.

Simulated trials also preserve decision time, pre-decision state, observation and
command source times, posterior, latent state, arbitration coefficients, reference,
nominal/executed forces, obstacle clearances and the energy ledger. Time samples
and repeated test trajectories are not independent statistical units. Analyses use
paired operator or training-repeat means, sign permutations, paired bootstrap
intervals and Holm correction. Missing outcomes stay missing.

`HSMMFilter` maintains a state-age joint belief with explicit duration hazards.
Its terminal state holds after skill completion. Its reference covariance includes
both within-state and between-state uncertainty. The optional online controller
uses this uncertainty for arbitration while preserving stage-gated path execution.

`MultiportPassivity` accepts exactly three sampled effort/flow pairs, each with
either a force or velocity actuator. Outbound power is positive at each port;
network-observed inbound energy is provided separately and must not count a
received energy packet twice. The controller distributes required damping among
eligible positive-power ports and updates finite stored energy. This sampled-port
contract is distinct from the implicit-midpoint robot-work constraint in
`ThreePortTank`.

Rigid calibration uses noncollinear point correspondences and an SVD rotation with
positive determinant. Keep camera extrinsics, tool offsets and independent check
point errors with local device records. Nominal DH kinematics do not replace a
manufacturer calibration. The RTDE interface holds tool orientation and controls
translation; a device-specific 6D pose policy needs a separate actuator integration.

Operator questionnaires and consent/approval metadata are read from actual CSV/YAML
records. The empty templates supply field names; they contain no participant data.

## Workstation feedback acknowledgement

The UDP workstation bridge transmits a delayed Cartesian state and measured
wrench together. `feedback_delay_seconds` is a one-way host buffer duration for
both channels. Command packets have no additional configured artificial delay;
real receive/compute/network latency remains present. No feedback frame is sent
until a real recorded sample has spent that duration in the buffer. The first
handshake therefore holds motion and sends zero reflected force.

Host replies contain `reflected_force_N[3]` and an optional `feedback` object:

```json
{
  "sequence": 12,
  "reflected_force_N": [0, 0, 0],
  "feedback": {
    "observation_sequence": 8,
    "source_time_s": 0.40,
    "host_source_time_s": 100.40,
    "host_send_time_s": 101.20,
    "frame": "base",
    "position_m": [0.16, -0.10, 0.46],
    "velocity_m_s": [0, 0, 0],
    "measured_wrench_N_Nm": [0, 0, 0, 0, 0, 0]
  }
}
```

The values illustrate the protocol. `source_time_s` is relative to session start;
the two `host_*_time_s` values use the host monotonic clock. Device timestamps are
logged independently and never subtracted from host time without an explicit
clock calibration. `observation_sequence` identifies this exact state/wrench
sample. A device bridge must first present that state in its operator display,
then echo its sequence in the command packet as `observation_sequence`. Merely
receiving a UDP frame is insufficient for a device bridge to assert presentation.
The echoed identifier must describe the displayed state associated with that
command sample. `measured_wrench_N_Nm` is telemetry; a haptic actuator must use
the separately limited `reflected_force_N` command and its device interlocks.
The workstation trusts this interface contract; it cannot prove which image or
state an operator actually saw. Native master SDK display and haptic rendering
still require device-specific integration.

The host retains at most 4096 sent frames. Its hierarchy uses the acknowledged
frame's state and acquisition time, rather than an unsent internal buffer value.
Acknowledgements must refer to retained sent frames, must not regress and must
arrive within `limits.feedback_acknowledgement_seconds`, defaulting to the
existing `watchdog_seconds` value (`.10` s in the example). Reusing a frame within
that deadline supports zero-order hold. Feedback age is measured from its host
acquisition time, so it includes the configured buffer delay and subsequent
bridge/display/command latency. The configured delay is not asserted to equal
the measured end-to-end age.

Motion, gripper commands and nonzero force reflection require valid acknowledged
feedback, master enable, acceptable robot safety mode and the existing workspace,
singularity and predicted-clearance interlocks. Missing, unknown, stale or
regressing acknowledgements cause a hold. Force reflection uses the delayed
wrench in the outgoing frame, bounded by the configured force limit; it uses
zero force when execution is disabled or an interlock fails. This sampled
velocity/admittance interface does not implement or establish a physical
three-port passivity result.

`master_keyboard.py` reads replies, displays the delayed TCP position in its
diagnostic console and echoes the sequence it displayed. It supplies keyboard
commands and zero measured master velocity; it does not render haptics or replace
a native visual/haptic device bridge.

Workstation recording schema 2 preserves acknowledged and sent frame sequences,
source time, measured feedback age, command receive time, delayed wrench,
reflected force and actual per-sample execution flags. Handshake samples have
`feedback_acknowledged=false` and `observed_state=null`; they cannot be labeled as
executed commands. The record contains configured channel delay, nominal and
actual host cadence, and robot model. It does not embed the controller IP.
Reflection work is a host-sampled estimate using the preceding held output force,
the newly received master velocity and the measured host receive interval. It is
not a synchronized device-port energy measurement.

`import_recording` requires schema 2 feedback evidence for an executed measured
demonstration. It rejects executed samples without acknowledgement and rejects
unacknowledged gaps after the initial handshake. Only the initial unexecuted
handshake prefix is omitted, with its count recorded. All retained samples and
their timing audit stay available; no internal gaps are silently removed. The
annotation's `delay_seconds` and `source_condition` must agree with the recorded
buffer setting and condition. Additional numeric NPZ arrays preserve
`observed_source_time[T]`, `feedback_age[T]`, `command_receive_time[T]`,
`acknowledged_observation_sequence[T]`, `master_sequence[T]` and `executed[T]`.
The source and command receive times use the host session clock; `time[T]` is
the current robot-state sampling time, so it can precede command receipt by part
of a host cycle. Use the explicit measured `feedback_age` rather than assuming
all timestamps occur simultaneously. A SHA-256 digest links the demonstration metadata
to its source recording. This import validates recording consistency; independently
annotated task success, collision and placement remain required.
