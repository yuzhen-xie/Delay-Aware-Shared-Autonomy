# Resolved execution parameters

These values describe this configuration. They do not establish values used in an earlier study.

| Parameter | Value | Unit | Definition |
| --- | --- | --- | --- |
| `seed` | `20261014` | count | Resolved execution setting; see module contract for structural meaning |
| `simulation.control_dt` | `0.001` | s | Implicit-midpoint integration and energy ledger interval |
| `simulation.decision_dt` | `0.02` | s | Held nominal command and inference interval |
| `simulation.max_seconds` | `18.0` | s | Simulation timeout; independent from study/device timeout |
| `simulation.mass` | `1.2` | kg | Translational point-mass plant; not UR10 inertia identification |
| `simulation.damping` | `12.0` | N s/m | Translational viscous damping |
| `simulation.robot_radius` | `0.025` | m | TCP collision sphere radius |
| `simulation.force_limit` | `18.0` | N | Norm limit before the energy constraint |
| `simulation.workspace_min` | `[0.02, -0.55, 0.02]` | m | Base-frame Cartesian lower bounds |
| `simulation.workspace_max` | `[0.95, 0.55, 0.85]` | m | Base-frame Cartesian upper bounds |
| `simulation.grasp_tolerance` | `0.045` | m | Abstract grasp-center distance tolerance |
| `simulation.place_tolerance` | `0.055` | m | Abstract placement-center tolerance |
| `simulation.dwell_seconds` | `0.16` | s | Required continuous low-speed dwell |
| `simulation.max_grasp_speed` | `0.18` | m/s | Grasp and placement speed threshold |
| `simulation.lift_height` | `0.48` | m | Base-frame lift height |
| `simulation.feedback_delay` | `0.8` | s, one-way | Slave observation-to-master channel |
| `simulation.command_delay` | `0.0` | s, one-way | Master command-to-slave channel |
| `simulation.jitter_seconds` | `0.0` | s | Uniform per-query jitter half-width, clipped at zero delay |
| `scene.provenance` | `"known_scene_simulation"` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `scene.frame` | `"base"` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `scene.goals` | `[[0.56, -0.25, 0.12], [0.64, 0.0, 0.12], [0.55, 0.24, 0.12]]` | m | Three candidate grasp centers in base coordinates |
| `scene.place` | `[0.8, 0.29, 0.12]` | m | Shared placement center in base coordinates |
| `scene.start` | `[0.16, -0.1, 0.46]` | m | TCP initial position in base coordinates |
| `scene.obstacles` | `[{"id": "center", "center": [0.43, 0.0, 0.23], "radius": 0.075}, {"id": "left", "center": [0.35, -0.28, 0.19], "radius": 0.06}]` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `scene.training_translation` | `0.025` | m | Common xyz uniform half-width; z is multiplied by 0.25 |
| `scene.novel_translation` | `[0.05, 0.15]` | m per xy axis | Independent signed shifts, followed by clipping |
| `scene.added_obstacle_radius` | `0.055` | m | Novel obstacle sphere radius |
| `scene.object_size_m` | `[0.04, 0.04, 0.06]` | m | Proposed static envelope for constrained manifests; not simulated contact geometry |
| `motion.attractive_gain` | `22.0` | N/m | k_att; fixed engineering default |
| `motion.repulsive_gain` | `0.2` | N m | k_rep in Gaussian radial field |
| `motion.repulsive_sigma` | `0.11` | m | Gaussian radial width |
| `motion.repulsive_limit` | `8.0` | N per obstacle | Per-obstacle repulsion norm cap |
| `motion.attractive_weight` | `1.0` | 1 | beta_1 alias: multiplier of attractive field |
| `motion.repulsive_weight` | `1.0` | 1 | beta_2 alias: multiplier of repulsive field |
| `motion.distance_min` | `0.08` | m | d_min / ell_min in the configured distance convention |
| `motion.distance_max` | `0.25` | m | d_max / ell_max in the configured distance convention |
| `motion.lambda_gain` | `10.0` | 1 | K_lambda: sigmoid slope for distance normalized by d_max-d_min |
| `motion.risk_distance` | `"center"` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `motion.kalman_process` | `0.08` | N^2/sample | Q=q I_3, not continuous-time noise density |
| `motion.kalman_measurement` | `0.8` | N^2 | R=r I_3 |
| `motion.kalman_initial_variance` | `1.0` | N^2 | P_0=p I_3; x_0=0 N |
| `arbitration.stiffness` | `28.0` | N/m | K^P=k I_3 |
| `arbitration.damping` | `5.0` | N s/m | K^V=b I_3 |
| `arbitration.covariance_lower` | `3e-05` | m^2 | Lower trace threshold for c_A |
| `arbitration.covariance_upper` | `0.006` | m^2 | Upper trace threshold for c_A |
| `arbitration.roughness_gamma` | `0.012` | s/N | gamma_j; one common sensitivity, not participant-specific fitted values |
| `arbitration.roughness_window` | `15` | samples | Moving mean of force-derivative norms |
| `arbitration.epsilon` | `1e-06` | 1 | Positive denominator regularizer for relative authority |
| `arbitration.low_confidence_threshold` | `0.12` | 1 | Guarded extension only |
| `arbitration.correction_distance` | `0.12` | m | Guarded extension uses surface clearance |
| `arbitration.rho_window_seconds` | `1.0` | s | Sliding window for removed/nominal positive work |
| `arbitration.fixed_weight` | `0.5` | 1 | C4 alpha |
| `arbitration.guarded_corrective_max` | `0.15` | 1 | Guarded extension alpha cap |
| `passivity.initial_energy` | `1.0` | J | Initial shared-control storage |
| `passivity.maximum_energy` | `8.0` | J | Storage capacity; autonomous mode starts at this value |
| `passivity.input_efficiency` | `0.9` | 1 | Credited fraction of positive sampled human input work |
| `passivity.tolerance` | `1e-08` | J | Numerical budget violation tolerance |
| `intent.beta` | `3.0` | rad^-1 | Boltzmann inverse angular cost scale; not a KL weight |
| `intent.confidence_threshold` | `0.6` | 1 | IPA and sustained early-prediction threshold |
| `intent.sustained_seconds` | `0.2` | s | Continuous correct/confident period for EPT |
| `intent.memory_decay` | `0.9` | 1 | Memory baseline evidence forgetting factor |
| `intent.hidden_dim` | `64` | count | Resolved execution setting; see module contract for structural meaning |
| `intent.latent_dim` | `8` | count | Resolved execution setting; see module contract for structural meaning |
| `intent.sequence_window` | `16` | count | Resolved execution setting; see module contract for structural meaning |
| `intent.transformer_heads` | `4` | count | Resolved execution setting; see module contract for structural meaning |
| `intent.transformer_layers` | `2` | count | Resolved execution setting; see module contract for structural meaning |
| `intent.batch_size` | `256` | count | Resolved execution setting; see module contract for structural meaning |
| `intent.epochs` | `35` | count | Resolved execution setting; see module contract for structural meaning |
| `intent.learning_rate` | `0.001` | 1/update | Adam learning rate |
| `intent.objective` | `"joint_elbo"` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `intent.kl_weight` | `1.0` | 1 | Exactly one for standard ELBO modes |
| `intent.deviation_weight` | `1.0` | 1 | One for joint ELBO; zero for action-only ELBO; explicit legacy otherwise |
| `intent.deviation_objective` | `"joint"` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `skill.states` | `7` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.trajectory_samples` | `140` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.covariance_floor` | `1e-05` | m^2 | Gaussian covariance eigenvalue lower bound |
| `skill.duration_floor` | `1.5` | resampled samples | Duration standard-deviation lower bound |
| `skill.em_iterations` | `6` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.lqt_control_cost` | `0.01` | relative cost | R=r I_3 in the LQT discretization |
| `skill.original_weight` | `1.0` | 1 per episode | Original demonstration likelihood weight |
| `skill.new_weight` | `1.0` | 1 per episode | Accepted new demonstration likelihood weight |
| `skill.refinement_count` | `20` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.online_filter` | `false` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `skill.goal_search_fraction` | `0.6` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `skill.goal_index_floor` | `4` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.lift_segment_fraction` | `0.22` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `skill.lookahead_samples` | `5` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.progress_advance_samples` | `3` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.proximity_threshold_m` | `0.035` | m | Advance path cursor near the current reference |
| `skill.goal_snap_samples` | `4` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.place_snap_samples` | `9` | count | Resolved execution setting; see module contract for structural meaning |
| `skill.place_snap_distance_m` | `0.05` | m | Snap final reference to placement center |
| `bootstrap.segment_min_seconds` | `0.45` | s | Synthetic segment duration lower bound |
| `bootstrap.cruise_speed_m_s` | `[0.16, 0.24]` | m/s | Synthetic quintic duration sampling range |
| `bootstrap.minimum_segment_samples` | `12` | count | Resolved execution setting; see module contract for structural meaning |
| `bootstrap.lateral_std_m` | `0.008` | m | Synthetic per-segment zero-endpoint lateral variation |
| `bootstrap.force_noise_std_N` | `0.15` | N | Synthetic inverse-dynamics force observation noise |
| `operator.kp` | `26.0` | N/m | Simulated operator PD position gain |
| `operator.kd` | `7.0` | N s/m | Simulated operator PD velocity gain |
| `operator.expert_noise` | `0.15` | N | Synthetic correlated force-noise innovation scale |
| `operator.noisy_noise` | `0.9` | N | Synthetic correlated force-noise innovation scale |
| `operator.corrective_gain` | `1.0` | 1 | Simulated outward correction multiplier |
| `experiments.intent_delays` | `[0.0, 0.4, 0.8, 1.5]` | s, one-way feedback | No command delay in the primary profile |
| `experiments.sharing_delays` | `[0.0, 0.4, 0.8, 1.5]` | s, one-way feedback | No command delay in the primary profile |
| `experiments.conditions` | `["C1", "C2", "C3", "C4", "delay_br_shared", "C3_guarded"]` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `experiments.scenarios` | `["demonstrated", "novel", "wider", "extra_obstacles"]` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `experiments.bootstrap_demonstrations` | `20` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.validation_demonstrations` | `6` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.test_demonstrations` | `12` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.simulated_operators` | `12` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.trials_per_condition` | `2` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.independent_repeats` | `10` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.autonomy_trials` | `100` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.bootstrap_samples` | `5000` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.permutation_samples` | `20000` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.port_bench_steps` | `20000` | count | Resolved execution setting; see module contract for structural meaning |
| `experiments.refinement_budgets` | `[0, 5, 10, 20]` | 1 / category | Resolved execution setting; see module contract for structural meaning |
| `study.participants` | `20` | count | Resolved execution setting; see module contract for structural meaning |
| `study.trial_repeats` | `1` | count | Resolved execution setting; see module contract for structural meaning |
| `study.practice_trials` | `8` | count | Resolved execution setting; see module contract for structural meaning |
| `study.max_seconds` | `60.0` | s | Proposed human-study timeout; historical value requires protocol record |
