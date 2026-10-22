from collections import deque
from collections.abc import Mapping

import numpy as np


class PassivityFault(RuntimeError):
    pass


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite components")
    return result.copy()


def _number(value, name, lower=0.0, strict=False):
    result = float(value)
    if not np.isfinite(result) or result < lower or (strict and result == lower):
        raise ValueError(f"Invalid {name}")
    return result


def allocation_weights(alpha, blend, rho, three_port=True):
    values = np.array([alpha, blend, rho], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Authority, motion blend and dissipation ratio must lie in [0,1]")
    weights = np.array([(1 - alpha) * (1 - rho), alpha, 1 - blend])
    if not three_port:
        weights[2] = 0.0
    if weights.sum() <= 1e-15:
        weights[1] = 1.0
    return weights / weights.sum()


class DissipationWindow:
    def __init__(self, seconds, epsilon_j=1e-12):
        self.seconds = _number(seconds, "dissipation window", strict=True)
        self.epsilon = _number(epsilon_j, "energy denominator", strict=True)
        self.samples = deque()
        self.removed = self.nominal = 0.0

    def add(self, timestamp, removed_j, nominal_positive_j):
        removed = _number(removed_j, "removed work")
        nominal = _number(nominal_positive_j, "nominal positive work")
        self.samples.append((float(timestamp), min(removed, nominal), nominal))
        self.removed += min(removed, nominal)
        self.nominal += nominal
        while self.samples and self.samples[0][0] < timestamp - self.seconds:
            _, old_removed, old_nominal = self.samples.popleft()
            self.removed -= old_removed
            self.nominal -= old_nominal

    @property
    def ratio(self):
        return float(np.clip(self.removed / max(self.nominal, self.epsilon), 0, 1))


class CumulativeEnergyReceiver:
    def __init__(self, maximum_increment_j=8.0, tolerance_j=1e-8):
        self.maximum_increment = _number(maximum_increment_j, "maximum energy credit", strict=True)
        self.tolerance = _number(tolerance_j, "energy tolerance", strict=True)
        self.peer_session = None
        self.sequence = -1
        self.cumulative = 0.0

    def accept(self, packet):
        if not isinstance(packet, Mapping) or packet.get("schema_version") != 1:
            raise PassivityFault("Invalid energy transport packet")
        sender = packet.get("sender_session")
        sequence = packet.get("sequence")
        if not isinstance(sender, str) or not sender or len(sender) > 128:
            raise PassivityFault("Invalid energy sender session")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise PassivityFault("Invalid energy packet sequence")
        try:
            cumulative = _number(packet.get("cumulative_export_j"), "cumulative exported energy")
        except (TypeError, ValueError) as error:
            raise PassivityFault("Invalid cumulative energy counter") from error
        if self.peer_session is not None and sender != self.peer_session:
            raise PassivityFault("Energy sender changed during an active recording")
        if self.peer_session is None:
            # Establish a baseline; energy sent before this acquisition is not spendable again.
            self.peer_session = sender
            self.sequence, self.cumulative = sequence, cumulative
            return 0.0
        if sequence < self.sequence:
            if cumulative > self.cumulative + self.tolerance:
                raise PassivityFault("An older packet carries a newer energy counter")
            return 0.0
        if sequence == self.sequence:
            if abs(cumulative - self.cumulative) > self.tolerance:
                raise PassivityFault("Duplicate energy sequence has conflicting contents")
            return 0.0
        increment = cumulative - self.cumulative
        if increment < -self.tolerance or increment > self.maximum_increment + self.tolerance:
            raise PassivityFault("Energy counter reset or an excessive credit increment was received")
        self.peer_session = sender
        self.sequence, self.cumulative = sequence, max(self.cumulative, cumulative)
        return max(0.0, increment)


class _PortBudget:
    def __init__(self, config, session_id, name):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("A nonempty acquisition-session identifier is required")
        physical = config.get("passivity", {}).get("physical", {})
        master = name == "master"
        initial_default, maximum_default = (0.05, 2.0) if master else (0.95, 8.0)
        self.initial = _number(physical.get(f"{name}_initial_energy_j", initial_default), "initial energy")
        self.maximum = _number(physical.get(f"{name}_maximum_energy_j", maximum_default),
                               "maximum energy", strict=True)
        if self.initial > self.maximum:
            raise ValueError("Initial port energy exceeds its capacity")
        self.speed_bound = _number(physical.get(f"{name}_speed_bound_m_s", 1.0 if master else .08),
                                   "observed speed bound", strict=True)
        self.hold_bound = _number(physical.get(f"{name}_maximum_hold_seconds", .005 if master else .024),
                                  "maximum command hold", strict=True)
        self.export_fraction = _number(physical.get(f"{name}_energy_export_fraction", .5),
                                       "export fraction")
        if self.export_fraction > 1:
            raise ValueError("The exported input-energy fraction cannot exceed one")
        self.tolerance = _number(physical.get("tolerance_j", 1e-8), "energy tolerance", strict=True)
        self.guard = _number(physical.get("coordination_guard_energy_j", .05), "coordination guard")
        self.receiver = CumulativeEnergyReceiver(physical.get("maximum_credit_increment_j", 8.0),
                                                 self.tolerance)
        self.window = DissipationWindow(config.get("arbitration", {}).get("rho_window_seconds", 1.0))
        self.session_id, self.name = session_id, name
        self.energy = self.initial
        self.reserved = self.received = self.absorbed_local = self.delivered = self.spilled = 0.0
        self.exported = self.signed_work = self.last_work = 0.0
        self.timestamp = self.position = self.measured_force = None
        self.command = np.zeros(3)
        self.nominal = np.zeros(3)
        self.weights = np.array([.5, .5, 0.0])
        self.packet_sequence = 0
        self.sample_open = False
        self.fault_reason = None
        self.last_dt = 0.0
        self.last_removed_work = self.last_nominal_work = 0.0

    def _fault(self, message):
        self.fault_reason = message
        raise PassivityFault(message)

    def _add_storage(self, amount):
        capacity = max(0.0, self.maximum - self.reserved)
        excess = max(0.0, self.energy + amount - capacity)
        self.energy = min(capacity, self.energy + amount)
        self.spilled += excess

    def accept_packet(self, packet):
        if packet is None:
            return
        if isinstance(packet, (list, tuple)):
            for value in packet:
                self.accept_packet(value)
            return
        credited = self.receiver.accept(packet)
        self.received += credited
        self._add_storage(credited)
        weights = packet.get("allocation_weights")
        if self.name == "master" and weights is not None and packet["sequence"] == self.receiver.sequence:
            weights = _vector(weights, "remote allocation weights")
            if np.any(weights < 0) or abs(weights.sum() - 1.0) > 1e-8:
                self._fault("Invalid remote allocation weights")
            self.weights = weights

    def observe(self, timestamp, force, position, velocity, remote_packet=None, held_command=None):
        if self.fault_reason:
            self._fault(self.fault_reason)
        timestamp = _number(timestamp, "monotonic timestamp")
        force = _vector(force, "observed force")
        position, velocity = _vector(position, "observed position"), _vector(velocity, "observed velocity")
        if np.linalg.norm(velocity) > self.speed_bound + 1e-6:
            self._fault("Measured port speed exceeds the reservation bound")
        if self.timestamp is not None:
            dt, displacement = timestamp - self.timestamp, position - self.position
            if dt <= 0 or dt > self.hold_bound + 1e-9:
                self._fault("Measured command interval exceeds the port hold bound")
            if np.linalg.norm(displacement) > self.speed_bound * dt + 1e-6:
                self._fault("Measured displacement exceeds the reservation bound")
            if self.sample_open:
                self._fault("A port observation arrived before an output command was registered")
            effort = .5 * (self.measured_force + force) if held_command is None \
                else _vector(held_command, "previous zero-order-held command")
            work = float(effort @ displacement)
            reservation = self.reserved
            if work > reservation + self.tolerance:
                self._fault("Observed output work exceeded the reserved command budget")
            self.reserved = 0.0
            self.last_work, self.signed_work = work, self.signed_work + work
            if work >= 0:
                self.delivered += work
                refund = reservation - work
                if refund >= 0:
                    self._add_storage(refund)
                else:
                    self.energy += refund
                    if self.energy < -self.tolerance:
                        self._fault("Observed output work exhausted the port storage")
                    self.energy = max(0.0, self.energy)
            else:
                self._add_storage(reservation)
                absorbed = -work
                exported = self.export_fraction * absorbed
                self.exported += exported
                self.absorbed_local += absorbed - exported
                self._add_storage(absorbed - exported)
            nominal_positive = max(float(self.nominal @ displacement), 0.0)
            corrected_positive = max(float(self.command @ displacement), 0.0)
            removed = max(0.0, nominal_positive - corrected_positive)
            self.window.add(timestamp, removed, nominal_positive)
            self.last_nominal_work, self.last_removed_work = nominal_positive, removed
            self.last_dt = dt
        self.timestamp, self.position, self.measured_force = timestamp, position, force
        self.sample_open = True
        self.accept_packet(remote_packet)

    def limit(self, nominal, velocity, dt, allocation_index):
        if self.fault_reason:
            self._fault(self.fault_reason)
        if not self.sample_open:
            self._fault("Each output command requires a fresh measured port observation")
        nominal, velocity = _vector(nominal, "nominal port force"), _vector(velocity, "port velocity")
        if np.linalg.norm(velocity) > self.speed_bound + 1e-6:
            self._fault("Command projection velocity exceeds the reservation bound")
        dt = _number(dt, "controller interval", strict=True)
        if dt > self.hold_bound + 1e-9:
            self._fault("Requested command interval exceeds the hold bound")
        allowance = max(0.0, self.energy - self.guard * self.weights[allocation_index])
        force = nominal.copy()
        predicted = self.hold_bound * float(force @ velocity)
        speed2 = float(velocity @ velocity)
        if predicted > allowance and speed2 > 1e-18:
            force -= (predicted - allowance) * velocity / (self.hold_bound * speed2)
        # The norm bound reserves for any direction of motion during the held command.
        maximum_norm = allowance / (self.speed_bound * self.hold_bound)
        norm = float(np.linalg.norm(force))
        if norm > maximum_norm:
            force *= maximum_norm / norm
        reservation = float(np.linalg.norm(force)) * self.speed_bound * self.hold_bound
        self.reserved = min(self.energy, reservation)
        self.energy -= self.reserved
        self.nominal, self.command = nominal, force
        self.sample_open = False
        return force.copy()

    @property
    def rho(self):
        return self.window.ratio

    def packet(self):
        result = {"schema_version": 1, "sender_session": self.session_id,
                  "sequence": self.packet_sequence, "cumulative_export_j": self.exported,
                  "allocation_weights": self.weights.tolist()}
        self.packet_sequence += 1
        return result

    def diagnostics(self):
        residual = (self.initial + self.received + self.absorbed_local - self.delivered
                    - self.spilled - self.energy - self.reserved)
        return {"port": self.name, "stored_energy_j": self.energy,
                "reserved_energy_j": self.reserved, "received_energy_j": self.received,
                "absorbed_local_energy_j": self.absorbed_local, "exported_energy_j": self.exported,
                "delivered_positive_work_j": self.delivered, "signed_observed_work_j": self.signed_work,
                "spilled_energy_j": self.spilled, "balance_residual_j": residual,
                "last_observed_work_j": self.last_work, "last_dt_seconds": self.last_dt,
                "last_removed_work_j": self.last_removed_work,
                "last_nominal_positive_work_j": self.last_nominal_work, "rho": self.rho,
                "rho_window_removed_work_j": max(0.0, self.window.removed),
                "rho_window_nominal_positive_work_j": max(0.0, self.window.nominal),
                "allocation_weights": self.weights.tolist(), "maximum_hold_seconds": self.hold_bound,
                "speed_bound_m_s": self.speed_bound, "fault_reason": self.fault_reason,
                "energy_scope": "sampled_command_or_measured_device_port_with_bounded_holds",
                "whole_robot_servo_passivity_asserted": False}


class MasterPassivity:
    def __init__(self, config, session_id):
        self.port = _PortBudget(config, session_id, "master")

    def begin_sample(self, timestamp, measured_device_force_N, position_m, velocity_m_s,
                     remote_packet=None):
        self.port.observe(timestamp, measured_device_force_N, position_m, velocity_m_s, remote_packet)

    def accept_packet(self, packet):
        self.port.accept_packet(packet)

    def limit_force(self, reflected_N, velocity_m_s, dt):
        return self.port.limit(reflected_N, velocity_m_s, dt, 0)

    @property
    def rho(self):
        return self.port.rho

    def packet(self):
        return self.port.packet()

    def diagnostics(self):
        result = self.port.diagnostics()
        result["observed_force_definition"] = "device_applied_force_in_robot_base_frame"
        result["human_force_sensor_assumed"] = False
        return result


class PhysicalPassivity:
    def __init__(self, config, session_id):
        self.port = _PortBudget(config, session_id, "robot")
        physical = config.get("passivity", {}).get("physical", {})
        self.autonomy_initial = _number(physical.get("autonomy_initial_energy_j", .25),
                                         "initial internal autonomy storage")
        self.autonomy_maximum = _number(physical.get("autonomy_maximum_energy_j", 2.0),
                                         "maximum internal autonomy storage", strict=True)
        if self.autonomy_initial > self.autonomy_maximum:
            raise ValueError("Initial autonomy storage exceeds its capacity")
        self.autonomy_energy = self.autonomy_initial
        self.autonomy_gain = _number(physical.get("autonomy_coordination_gain", 1.0),
                                      "autonomy coordination gain")
        self.autonomy_damping = _number(config.get("arbitration", {}).get("damping", 5.0),
                                         "autonomy damping")
        self.autonomy_work = self.autonomy_removed = self.autonomy_spilled = 0.0
        self.autonomy_last_removed = 0.0
        self.previous_external = self.previous_position = None
        self.external_work = 0.0
        self.last_corrected_velocity = np.zeros(3)
        self.autonomy_updated = False
        self.coordination_rho = 0.0

    def set_transparency_ratio(self, ratio):
        ratio = _number(ratio, "combined physical-port dissipation ratio")
        if ratio > 1:
            raise ValueError("The dissipation ratio must lie in [0,1]")
        self.coordination_rho = ratio

    def begin_sample(self, timestamp, observed_external_wrench_N, robot_position_m,
                     measured_robot_velocity_m_s, master_energy_packet=None,
                     executed_command_force_N=None):
        external = _vector(observed_external_wrench_N, "observed external wrench")
        position = _vector(robot_position_m, "robot position")
        command = self.port.command if executed_command_force_N is None else executed_command_force_N
        self.port.observe(timestamp, external, position, measured_robot_velocity_m_s,
                          master_energy_packet, held_command=command)
        if self.previous_position is not None:
            self.external_work += float(.5 * (self.previous_external + external)
                                        @ (position - self.previous_position))
        self.previous_position, self.previous_external = position, external
        self.autonomy_updated = False

    def accept_packet(self, packet):
        self.port.accept_packet(packet)

    def limit_autonomy_velocity(self, force_N, velocity_m_s, alpha, blend, dt, three_port=True):
        if not self.port.sample_open or self.autonomy_updated:
            raise PassivityFault("PC3 requires one fresh observation per reference update")
        force, velocity = _vector(force_N, "autonomy effort"), _vector(velocity_m_s, "autonomy velocity")
        dt = _number(dt, "autonomy interval", strict=True)
        if dt > self.port.hold_bound + 1e-9:
            raise PassivityFault("Autonomy update interval exceeds the hold bound")
        self.port.weights = allocation_weights(alpha, blend, self.coordination_rho, three_port)
        weight = self.port.weights[2]
        corrected = velocity.copy()
        self.autonomy_last_removed = 0.0
        if three_port and weight > 0:
            # This internal-port constraint supplies no additional robot-port credit.
            allowance = self.autonomy_energy / (1 + self.autonomy_gain * weight)
            nominal_work = dt * float(force @ velocity)
            spring = force + self.autonomy_damping * velocity
            if nominal_work > allowance:
                linear = float(spring @ velocity)
                quadratic = self.autonomy_damping * float(velocity @ velocity)
                power_limit = allowance / dt
                discriminant = max(0.0, linear**2 - 4 * quadratic * power_limit)
                denominator = linear + np.sqrt(discriminant)
                scale = 0.0 if denominator <= 0 else 2 * power_limit / denominator
                corrected *= float(np.clip(scale, 0, 1))
            corrected_effort = spring - self.autonomy_damping * corrected
            work = dt * float(corrected_effort @ corrected)
            if work > self.autonomy_energy + self.port.tolerance:
                raise PassivityFault("Internal autonomy work exceeded its storage")
            energy = self.autonomy_energy - work
            self.autonomy_spilled += max(0.0, energy - self.autonomy_maximum)
            self.autonomy_energy = float(np.clip(energy, 0, self.autonomy_maximum))
            removed = max(0.0, nominal_work - work)
            self.autonomy_work += work
            self.autonomy_removed += removed
            self.autonomy_last_removed = removed
        self.last_corrected_velocity = corrected
        self.autonomy_updated = True
        return corrected.copy()

    def limit_robot_force(self, nominal_N, velocity_m_s, alpha, blend, dt, three_port=True):
        self.port.weights = allocation_weights(alpha, blend, self.coordination_rho, three_port)
        return self.port.limit(nominal_N, velocity_m_s, dt, 1)

    @property
    def rho(self):
        return self.port.rho

    def packet(self):
        return self.port.packet()

    def diagnostics(self):
        result = self.port.diagnostics()
        result.update({"command_port_scope": "commanded_cartesian_wrench_measured_tcp_displacement",
                       "external_wrench_signed_work_j": self.external_work,
                       "external_wrench_credit_enabled": False,
                       "autonomy_internal_energy_j": self.autonomy_energy,
                       "autonomy_internal_signed_work_j": self.autonomy_work,
                       "autonomy_internal_removed_work_j": self.autonomy_removed,
                       "autonomy_last_removed_work_j": self.autonomy_last_removed,
                       "autonomy_internal_spilled_energy_j": self.autonomy_spilled,
                       "autonomy_balance_residual_j": self.autonomy_initial - self.autonomy_work
                       - self.autonomy_spilled - self.autonomy_energy,
                       "autonomy_velocity_input_m_s": self.last_corrected_velocity.tolist(),
                       "autonomy_port_is_internal_not_additional_source": True,
                       "allocation_implementation": "port_weighted_storage_guards_and_PC3_work_projection",
                       "reservation_assumptions": ["bounded_measured_velocity_and_displacement",
                                                   "bounded_zero_order_hold_duration",
                                                   "actual_device_force_or_applied_command_port",
                                                   "no_unobserved_actuator_energy_guarantee"]})
        return result
