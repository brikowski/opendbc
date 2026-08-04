import math

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, rate_limit, make_tester_present_msg, structs
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.common.pid import PIDController
from opendbc.car.honda import hondacan
from opendbc.car.honda.values import CAR, CruiseButtons, HONDA_BOSCH, HONDA_BOSCH_CANFD, HONDA_BOSCH_RADARLESS, \
                                     HONDA_BOSCH_TJA_CONTROL, HONDA_NIDEC_ALT_PCM_ACCEL, CarControllerParams
from opendbc.car.interfaces import CarControllerBase

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState

# CUSTOM TUNE (ody-op-long): speed-scheduled baseline gasfactor for the Odyssey 5G MMR; the
# live-learned gasfactor is a residual trim on top (effective = base(vEgo) * trim). Check-mark
# shape, not monotonic. Each value is the converged effective gasfactor over many drives, so the
# trim sits near 1.0 and only the cold-start ramp depends on the seed: too high over-gasses the
# launch, too low is low-cruise sluggishness. Both have happened - validate_log's "GASFACTOR vs
# SEED" report tracks all four and is the source of truth, so don't re-derive by hand.
# TODO: delete excessive comments before trying to submit a PR.
GAS_FACTOR_SPEED_BP = [0.0, 8.0, 15.0, 22.0]   # m/s
GAS_FACTOR_SPEED_V = [0.72, 0.54, 0.56, 0.60]

# CUSTOM TUNE (ody-op-long): integral gain of the supplemental brake_pid. Briefly set to 0 on
# 2026-07-29 to test whether its windup drove the downhill surge; SUPERSEDED before being driven.
# An experimentalMode on/off A/B within route 00000030 showed our BRAKE_REQUEST toggle rate is
# ~44/min on descents REGARDLESS of plan smoothness - the plan got 2.4x smoother and our toggling
# did not improve at all - so the cycle is sustained by the domain threshold below, not by this
# integral. Restored to 0.5 so DOMAIN_HYST_EXIT can be tested on its own.
# TODO: delete excessive comments before trying to submit a PR.
BRAKE_PID_KI = 0.5

# CUSTOM TUNE (ody-op-long): maximum one-sided hysteresis on the gas/brake domain. Enter brake at
# the bare threshold (a brake application is NEVER delayed); above 10 m/s, leave only once
# switch_accel clears it by this band. The band ramps from zero at 5 m/s to this value at 10 m/s:
# applying the full 0.50 at a stop kept BRAKE_REQUEST latched and GAS_COMMAND inactive until the
# planner asked for >+0.51 m/s^2, delaying ordinary +0.03..+0.51 starts by up to 1.3 s in replay.
#
# Descending at set speed, hill_brake = sin(pitch)*g is -0.20..-0.23 m/s^2 against a -0.20 threshold,
# so grade compensation parks switch_accel on the boundary and every plan wiggle pulses
# BRAKE_REQUEST - the driver-reported "taps the brake, gravity speeds it back up".
#
# Sized to EXCEED the plan's own ripple, which is the one thing a hysteresis band has to do: that
# ripple is 0.51 m/s^2 p-p at set speed (0.81 in experimentalMode). Two smaller bands were road-tested
# and did nothing - symmetric 0.06, then 0.20 on route 00000033, which gave 27.4 descent
# BRAKE_REQUEST toggles/min against 25.0 with no hysteresis at all. Both were sized to a knee in an
# open-loop crossing-rate sweep; that sweep has underpredicted on-road crossings ~2.7x every time it
# has been checked (see agents.md), so size to the ripple instead. The 0.50 routes recovered the
# no-hysteresis descent-toggle distribution; a terrain-matched road drive is still required to show
# a benefit over that baseline. TODO: delete excessive comments before trying to submit a PR.
DOMAIN_HYST_EXIT = 0.50


def compute_gb_honda_bosch(accel, speed):
  # TODO returns 0s, is unused
  return 0.0, 0.0


def compute_gb_honda_nidec(accel, speed):
  creep_brake = 0.0
  creep_speed = 2.3
  creep_brake_value = 0.15
  if speed < creep_speed:
    creep_brake = (creep_speed - speed) / creep_speed * creep_brake_value
  gb = float(accel) / 4.8 - creep_brake
  return np.clip(gb, 0.0, 1.0), np.clip(-gb, 0.0, 1.0)


def compute_gas_brake(accel, speed, fingerprint):
  if fingerprint in HONDA_BOSCH:
    return compute_gb_honda_bosch(accel, speed)
  else:
    return compute_gb_honda_nidec(accel, speed)


# TODO not clear this does anything useful
def actuator_hysteresis(brake, braking, brake_steady):
  # hyst params
  brake_hyst_on = 0.02    # to activate brakes exceed this value
  brake_hyst_off = 0.005  # to deactivate brakes below this value
  brake_hyst_gap = 0.01   # don't change brake command for small oscillations within this value

  # *** hysteresis logic to avoid brake blinking. go above 0.1 to trigger
  if (brake < brake_hyst_on and not braking) or brake < brake_hyst_off:
    brake = 0.
  braking = brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if brake == 0.:
    brake_steady = 0.
  elif brake > brake_steady + brake_hyst_gap:
    brake_steady = brake - brake_hyst_gap
  elif brake < brake_steady - brake_hyst_gap:
    brake_steady = brake + brake_hyst_gap
  brake = brake_steady

  return brake, braking, brake_steady


def brake_pump_hysteresis(apply_brake, apply_brake_last, last_pump_ts, ts):
  pump_on = False

  # reset pump timer if:
  # - there is an increment in brake request
  # - we are applying steady state brakes and we haven't been running the pump
  #   for more than 20s (to prevent pressure bleeding)
  if apply_brake > apply_brake_last or (ts - last_pump_ts > 20. and apply_brake > 0):
    last_pump_ts = ts

  # once the pump is on, run it for at least 0.2s
  if ts - last_pump_ts < 0.2 and apply_brake > 0:
    pump_on = True

  return pump_on, last_pump_ts


def process_hud_alert(hud_alert):
  alert_fcw = False
  alert_steer_required = False

  # Make sure FCW is prioritized over steering required
  # TODO: implement separate available LDW alert
  if hud_alert == VisualAlert.fcw:
    alert_fcw = True
  elif hud_alert in (VisualAlert.steerRequired, VisualAlert.ldw):
    alert_steer_required = True

  return alert_fcw, alert_steer_required


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.params = CarControllerParams(CP)
    self.CAN = hondacan.CanBus(CP)
    self.tja_control = CP.carFingerprint in HONDA_BOSCH_TJA_CONTROL

    self.braking = False
    self.brake_steady = 0.
    self.brake_last = 0.
    self.apply_brake_last = 0
    self.last_pump_ts = 0.
    self.stopping_counter = 0

    self.accel = 0.0
    self.speed = 0.0
    self.gas = 0.0
    self.brake = 0.0
    self.last_torque = 0.0

    # CUSTOM TUNE (ody-op-long): live-learning gas/wind feedforward for the Odyssey 5G MMR.
    # Scales GAS_COMMAND only, never ACCEL_COMMAND, so it cannot stack with Honda Bosch's own
    # internal brake PID (opendbc #2347). Factors re-learn each drive instead of persisting:
    # persistence needs openpilot Params, i.e. opendbc reaching up into the app layer.
    # effective gasfactor = base(vEgo) * trim - speed shape in GAS_FACTOR_SPEED_V, this is the
    # residual trim, so un-driven speeds sit at the baseline instead of a stale scalar.
    # Telemetry: actuatorsOutput.gas is repurposed to the effective gasfactor, .brake to
    # windfactor (jotpluggler "Long - Learning").
    # TODO: delete excessive comments before trying to submit a PR.
    self.bosch_last_gas = 0
    # Low-pass the IMU pitch: orientationNED[1] is noisy and feeds sin(pitch)*g straight into the
    # gas feedforward. Toyota's noise filter only - NOT its high-pass (amplifies transients for a
    # PCM Honda doesn't have) nor its min(pitch,0) clamp (we keep bidirectional grade comp, so
    # uphill gas assist survives).
    # TODO: delete excessive comments before trying to submit a PR.
    self.pitch = FirstOrderFilter(0.0, 0.5, DT_CTRL)
    self.gasfactor = 1.0            # residual trim on top of the speed-scheduled baseline
    self.gasfactor_effective = 1.0  # base(vEgo) * trim, exposed in telemetry (updated in update())
    self.windfactor = 0.5
    # Saturation guards - init exactly the names update() reads.
    self.gasfactor_before_gasmax = self.gasfactor
    self.windfactor_before_brake = self.windfactor_before_gasmax = self.windfactor

    # Bosch extra-brake controller: integral-only (k_p=0) and one-directional (pos_limit=0,
    # can only ADD braking, never remove it). It supplements Honda Bosch's mushy internal
    # brake response when we're already asking to decelerate; it does not fight it. This is
    # the one piece that feeds ACCEL_COMMAND, so it's the thing to watch first on-road.
    # TODO: delete excessive comments before trying to submit a PR.
    self.brake_pid = PIDController(k_p=([0.], [0.]), k_i=([0.], [BRAKE_PID_KI]),
                                   pos_limit=0.0, neg_limit=-2.0, rate=50)
    self.brake_pid.reset()
    self.in_brake_domain = False   # hysteretic domain state (see DOMAIN_HYST_EXIT)

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    hud_v_cruise = hud_control.setSpeed / CS.v_cruise_factor if hud_control.speedVisible else 255
    pcm_cancel_cmd = CC.cruiseControl.cancel

    # CUSTOM TUNE (ody-op-long): keep the IMU pitch filter updated every frame (100 Hz) for
    # continuity; the grade term itself is computed in the Odyssey block below.
    # TODO: delete excessive comments before trying to submit a PR.
    if len(CC.orientationNED) == 3:
      self.pitch.update(CC.orientationNED[1])

    if CC.longActive:
      accel = actuators.accel
      gas, brake = compute_gas_brake(actuators.accel, CS.out.vEgo, self.CP.carFingerprint)
    else:
      accel = 0.0
      gas, brake = 0.0, 0.0

    # *** rate limit steer ***
    limited_torque = rate_limit(actuators.torque, self.last_torque, -self.params.STEER_DELTA_DOWN * DT_CTRL,
                                self.params.STEER_DELTA_UP * DT_CTRL)
    self.last_torque = limited_torque

    # *** apply brake hysteresis ***
    pre_limit_brake, self.braking, self.brake_steady = actuator_hysteresis(brake, self.braking, self.brake_steady)

    # *** rate limit after the enable check ***
    self.brake_last = rate_limit(pre_limit_brake, self.brake_last, -2., DT_CTRL)

    # vehicle hud display, wait for one update from 10Hz 0x304 msg
    alert_fcw, alert_steer_required = process_hud_alert(hud_control.visualAlert)

    # **** process the car messages ****

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_torque = int(np.interp(-limited_torque * self.params.STEER_MAX,
                                 self.params.STEER_LOOKUP_BP, self.params.STEER_LOOKUP_V))

    # Send CAN commands
    can_sends = []

    # tester present - w/ no response (keeps radar disabled)
    if self.CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) and self.CP.openpilotLongitudinalControl:
      if self.frame % 10 == 0:
        can_sends.append(make_tester_present_msg(0x18DAB0F1, 1, suppress_response=True))

    # Send steering command.
    can_sends.append(hondacan.create_steering_control(self.packer, self.CAN, apply_torque, CC.latActive, self.tja_control))

    # wind brake from air resistance decel at high speed
    wind_brake = np.interp(CS.out.vEgo, [0.0, 2.3, 35.0], [0.001, 0.002, 0.15])
    # all of this is only relevant for HONDA NIDEC
    max_accel = np.interp(CS.out.vEgo, self.params.NIDEC_MAX_ACCEL_BP, self.params.NIDEC_MAX_ACCEL_V)
    # TODO this 1.44 is just to maintain previous behavior
    pcm_speed_BP = [-wind_brake,
                    -wind_brake * (3 / 4),
                    0.0,
                    0.5]
    # The Honda ODYSSEY seems to have different PCM_ACCEL
    # msgs, is it other cars too?
    if not CC.longActive:
      pcm_speed = 0.0
      pcm_accel = int(0.0)
    elif self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL:
      pcm_speed_V = [0.0,
                     np.clip(CS.out.vEgo - 3.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 0.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 5.0, 0.0, 100.0)]
      pcm_speed = float(np.interp(gas - brake, pcm_speed_BP, pcm_speed_V))
      pcm_accel = int(1.0 * self.params.NIDEC_GAS_MAX)
    else:
      pcm_speed_V = [0.0,
                     np.clip(CS.out.vEgo - 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 5.0, 0.0, 100.0)]
      pcm_speed = float(np.interp(gas - brake, pcm_speed_BP, pcm_speed_V))
      pcm_accel = int(np.clip((accel / 1.44) / max_accel, 0.0, 1.0) * self.params.NIDEC_GAS_MAX)

    if not self.CP.openpilotLongitudinalControl:
      if self.frame % 2 == 0 and self.CP.carFingerprint not in HONDA_BOSCH_RADARLESS | HONDA_BOSCH_CANFD:
        can_sends.append(hondacan.create_bosch_supplemental_1(self.packer, self.CAN))
      # If using stock ACC, spam cancel command to kill gas when OP disengages.
      if pcm_cancel_cmd:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.CANCEL, self.CP.carFingerprint))
      elif CC.cruiseControl.resume:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.RES_ACCEL, self.CP.carFingerprint))

    else:
      # Send gas and brake commands.
      if self.frame % 2 == 0:
        ts = self.frame * DT_CTRL

        if self.CP.carFingerprint in HONDA_BOSCH:
          if self.CP.carFingerprint == CAR.HONDA_ODYSSEY_5G_MMR:
            # ===== CUSTOM TUNE (ody-op-long): live-learning gas + supplemental brake =====
            # Scoped to the Odyssey only so other Bosch Hondas keep stock behavior (we can't
            # road-test them). See __init__ for the design rationale and PR #2347 context.
            # TODO: delete excessive comments before trying to submit a PR.

            min_gas = self.params.BOSCH_GAS_LOOKUP_BP[0]

            # Aero drag decel in m/s^2, scaled live by self.windfactor (base curve approximate -
            # windfactor learns the residual), plus grade from the low-passed IMU pitch. Both
            # feed the gas side only; see opendbc #2165.
            # DO NOT REMOVE the windfactor/drag term as "dead". On a low-speed route it sits at
            # its 0.1 floor (drag is negligible there), which looks like a dead knob but is
            # correct speed-adaptive behavior - verified on-road at highway speed, where it
            # carries steady-state cruise gas. Removing it dumps that onto the gasfactor learner.
            wind_brake_ms2 = np.interp(CS.out.vEgo, [0.0, 13.4, 22.4, 31.3, 40.2], [0.000, 0.049, 0.136, 0.267, 0.441])
            hill_brake = math.sin(self.pitch.x) * ACCELERATION_DUE_TO_GRAVITY

            # gas_pedal_force = desired accel + aero drag + grade, all in m/s^2. Uses raw
            # accel (not self.accel) so the brake_pid addon doesn't feed the gas side.
            gas_pedal_force = accel + wind_brake_ms2 * self.windfactor + hill_brake

            # Single source of truth for the gas/brake domain, mirrored onto the wire by
            # create_acc_commands:
            #  - threshold: speed-raised per opendbc PR #2342 (stock -0.2 at/above 10 m/s,
            #    +0.01 below 5 m/s) so BRAKE_REQUEST holds through a low-speed stop instead of
            #    releasing early and letting the van creep into the lead.
            #  - switch input: grade/drag-compensated gas_pedal_force at speed, but the raw
            #    planner accel below 5 m/s - near a stop the hill term alone (~+0.34 m/s^2 at
            #    0.035 rad pitch) can push gas_pedal_force over the threshold while the
            #    planner is still braking.
            # The brake_pid and learning gates below key off this same decision, so the
            # supplemental brake can never wind up while the wire is in the gas domain (the
            # phantom-windup landmine seen on route 805f87f5.../0000008f) and the gas factors
            # never learn from frames where GAS_COMMAND is actually off.
            # TODO: delete excessive comments before trying to submit a PR.
            min_gas_accel = float(np.interp(CS.out.vEgo, [5.0, 10.0], [0.01, min_gas]))
            switch_accel = accel if CS.out.vEgo < 5.0 else gas_pedal_force
            # One-sided hysteresis (see DOMAIN_HYST_EXIT): entry is the bare threshold, while exit
            # needs to clear a speed-scheduled band. There is deliberately no exit band at or below
            # 5 m/s: low-speed switching already uses raw planner accel, and holding the full 0.50
            # across an engaged stop transmitted BRAKE_REQUEST=1 with gas inactive against positive
            # +0.03..+0.51 m/s^2 start requests. Ramp the band back in with the grade-compensated
            # switch input over 5-10 m/s. Carried across ACTIVE frames, so the decision is passed to
            # create_acc_commands rather than recomputed there.
            #
            # Route 00000034 exposed why disengagement must clear the state: while the driver was
            # controlling the car, the latched brake domain let brake_pid integrate to about
            # -2 m/s^2 even though create_acc_commands correctly sent no brake. Re-engaging then
            # leaked that stale command onto the wire for 0.20 s against a positive planner request.
            # TODO: delete excessive comments before trying to submit a PR.
            domain_hyst_exit = float(np.interp(CS.out.vEgo, [5.0, 10.0], [0.0, DOMAIN_HYST_EXIT]))
            if not CC.longActive:
              self.in_brake_domain = False
            elif switch_accel < min_gas_accel:
              self.in_brake_domain = True
            elif switch_accel > min_gas_accel + domain_hyst_exit:
              self.in_brake_domain = False
            in_brake_domain = self.in_brake_domain
            in_gas_domain = not in_brake_domain
            brake_domain = in_brake_domain   # what the wire must mirror

            # Supplemental brake authority: integral-only, one-directional (can only ADD
            # braking). Supplements Honda Bosch's mushy internal brake response; resets any
            # time the wire is not actually braking.
            if in_brake_domain and (CS.out.vEgo > 1e-3):
              brake_addon = self.brake_pid.update(error=accel - CS.out.aEgo, speed=CS.out.vEgo)
              target_accel = min(accel, accel + brake_addon)
            else:
              self.brake_pid.reset()
              target_accel = accel
            self.accel = float(np.clip(target_accel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))

            # Speed-scheduled baseline gasfactor; self.gasfactor is the live-learned residual
            # trim on top of it. effective = base(vEgo) * trim (see __init__ for rationale).
            base_gasfactor = float(np.interp(CS.out.vEgo, GAS_FACTOR_SPEED_BP, GAS_FACTOR_SPEED_V))

            # Live-learn the gas/wind correction factors, only while openpilot controls the
            # gas (longControlState == pid) and the driver's foot is off the pedal.
            if (actuators.longControlState == LongCtrlState.pid) and (not CS.out.gasPressed):
              gas_error = self.accel - CS.out.aEgo
              # The gas_pedal_force > min_gas term keeps the (gas_pedal_force - min_gas) learn
              # weight positive (a below-5 m/s downhill can be in the gas domain on planner
              # accel while gas_pedal_force sits under the lookup floor -> no gas is actually
              # commanded, so there's no signal to learn from and the weight would flip sign).
              if in_gas_domain and gas_pedal_force > min_gas:
                # Learn rate: faster at low speed, slower at cruise. Nudges the residual trim
                # only - the speed shape is carried by base_gasfactor. Ours, unverified.
                learn_divisor = np.interp(CS.out.vEgo, [0., 15., 25.], [150, 200, 400])
                self.gasfactor = np.clip(self.gasfactor + gas_error / learn_divisor * (gas_pedal_force - min_gas), 0.01, 3.0)
              if (not CS.out.brakePressed) and (CS.out.vEgo > 0.0):
                wind_learn_divisor = 500  # ours, unverified
                wind_adjust = 1 + wind_brake_ms2 / wind_learn_divisor
                self.windfactor = np.clip(self.windfactor * (wind_adjust if (gas_error > 0) else 1.0 / wind_adjust), 0.1, 3.0)
              if in_brake_domain:  # don't reduce windfactor while braking, allow increases
                self.windfactor = max(self.windfactor, self.windfactor_before_brake)
              else:
                self.windfactor_before_brake = self.windfactor
              if gas_pedal_force >= self.params.BOSCH_ACCEL_MAX:  # at accel max the signal is saturated: allow decreases only
                self.gasfactor = min(self.gasfactor, self.gasfactor_before_gasmax)
                self.windfactor = min(self.windfactor, self.windfactor_before_gasmax)
              else:
                self.gasfactor_before_gasmax = self.gasfactor
                self.windfactor_before_gasmax = self.windfactor

            self.gasfactor_effective = base_gasfactor * self.gasfactor
            requested_gas = float(np.interp((gas_pedal_force - min_gas) * self.gasfactor_effective + min_gas,
                                             self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))
            # CUSTOM TUNE: rate-limit the command that can actually reach the wire, and clear the
            # ramp state whenever gas is ineligible. Advancing bosch_last_gas while braking or
            # inactive precharged an unsent command, so the first live GAS_COMMAND after a handoff
            # jumped to 192-255 counts on routes 37/3a/3b despite the intended 60-count limit.
            # TODO: delete excessive comments before trying to submit a PR.
            if CC.longActive and in_gas_domain:
              self.gas = min(requested_gas, self.bosch_last_gas + 60)
              self.bosch_last_gas = self.gas
            else:
              self.gas = 0.0
              self.bosch_last_gas = 0.0
          else:
            # Stock behavior for all other Bosch Hondas: fixed threshold, switch on raw accel
            # (None/None keeps create_acc_commands identical to upstream).
            self.accel = float(np.clip(accel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))
            self.gas = float(np.interp(accel, self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))
            switch_accel = None
            min_gas_accel = None
            brake_domain = None

          stopping = actuators.longControlState == LongCtrlState.stopping
          self.stopping_counter = self.stopping_counter + 1 if stopping else 0
          can_sends.extend(hondacan.create_acc_commands(self.packer, self.CAN, CC.enabled, CC.longActive, self.accel, self.gas,
                                                        self.stopping_counter, self.CP.carFingerprint, switch_accel, min_gas_accel,
                                                        brake_domain))
        else:
          apply_brake = np.clip(self.brake_last - wind_brake, 0.0, 1.0)
          apply_brake = int(np.clip(apply_brake * self.params.NIDEC_BRAKE_MAX, 0, self.params.NIDEC_BRAKE_MAX - 1))
          pump_on, self.last_pump_ts = brake_pump_hysteresis(apply_brake, self.apply_brake_last, self.last_pump_ts, ts)

          pcm_override = True
          can_sends.append(hondacan.create_brake_command(self.packer, self.CAN, apply_brake, pump_on,
                                                         pcm_override, pcm_cancel_cmd, alert_fcw, CS.stock_brake))
          self.apply_brake_last = apply_brake
          self.brake = apply_brake / self.params.NIDEC_BRAKE_MAX

    # Send dashboard UI commands.
    if self.frame % 10 == 0:
      if self.CP.openpilotLongitudinalControl:
        # On Nidec, this also controls longitudinal positive acceleration
        can_sends.append(hondacan.create_acc_hud(self.packer, self.CAN.pt, self.CP, CC.enabled, pcm_speed, pcm_accel,
                                                 hud_control, hud_v_cruise, CS.is_metric, CS.acc_hud))

      steering_available = CS.out.cruiseState.available and CS.out.vEgo > max(self.params.STEER_GLOBAL_MIN_SPEED, self.CP.minSteerSpeed)
      can_sends.extend(hondacan.create_lkas_hud(self.packer, self.CAN.lkas, self.CP, hud_control, CC.latActive,
                                                steering_available, alert_steer_required, CS.lkas_hud))

      if self.CP.openpilotLongitudinalControl:
        # TODO: combining with create_acc_hud block above will change message order and will need replay logs regenerated
        if self.CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS):
          can_sends.append(hondacan.create_radar_hud(self.packer, self.CAN.pt))
        if self.CP.carFingerprint == CAR.HONDA_CIVIC_BOSCH:
          can_sends.append(hondacan.create_legacy_brake_command(self.packer, self.CAN.pt))
        if self.CP.carFingerprint not in HONDA_BOSCH:
          self.speed = pcm_speed
          self.gas = pcm_accel / self.params.NIDEC_GAS_MAX

    new_actuators = actuators.as_builder()
    new_actuators.speed = self.speed
    new_actuators.accel = self.accel
    if self.CP.carFingerprint == CAR.HONDA_ODYSSEY_5G_MMR:
      # CUSTOM TUNE (ody-op-long): expose the learned factors in logs for tuning. The real
      # gas command is still visible on the CAN channel (/sendcan/.../ACC_CONTROL/GAS_COMMAND),
      # so we repurpose these telemetry fields to watch the factors converge in the jotpluggler
      # Long - Learning tab. gas = *effective* gasfactor (speed baseline * residual trim), so
      # it's directly comparable across speeds; brake = windfactor. NOT the actual gas/brake.
      # TODO: delete excessive comments before trying to submit a PR.
      new_actuators.gas = float(self.gasfactor_effective)
      new_actuators.brake = float(self.windfactor)
    else:
      new_actuators.gas = self.gas
      new_actuators.brake = self.brake
    new_actuators.torque = self.last_torque
    new_actuators.torqueOutputCan = apply_torque

    self.frame += 1
    return new_actuators, can_sends
