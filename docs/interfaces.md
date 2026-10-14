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
