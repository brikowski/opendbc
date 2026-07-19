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

# CUSTOM TUNE (ody-op-long): speed-scheduled baseline gasfactor for the Odyssey 5G MMR. The
# live-learned gasfactor is a residual trim on top of this (effective = base(vEgo) * trim).
# Shape is a check-mark, NOT monotonic: high at launch, dipping to a low-cruise minimum, then
# rising again with speed.
#  - 8/15/22 m/s -> 0.35/0.45/0.60: measured on route 805f87f5.../00000088 (2026-07-16).
#  - 0 m/s -> 0.90: a launch from a dead stop needs far more gas per unit accel (static
#    friction, torque converter not yet coupled) than low-speed cruise. The earlier flat 0.35
#    floor below 8 m/s starved launches and felt sluggish; route 805f87f5.../0000008c showed
#    the live trim clawing ~2x up to reach ~0.8 effective at launch, and that catch-up lag was
#    the sluggishness. Launch value is an estimate from that convergence - re-verify/refine.
# TODO: delete excessive comments before trying to submit a PR.
GAS_FACTOR_SPEED_BP = [0.0, 8.0, 15.0, 22.0]   # m/s
GAS_FACTOR_SPEED_V = [0.90, 0.35, 0.45, 0.60]


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

    # CUSTOM TUNE (ody-op-long): live-learning gas/wind feedforward correction for the
    # Odyssey 5G MMR, adapted from mvl-boston's opendbc (commit 1ef9db8). Only ever scales
    # the GAS_COMMAND magnitude via the gas/wind factors below - it never touches
    # ACCEL_COMMAND - so it cannot stack with Honda Bosch's own internal brake PID the way an
    # openpilot-side longitudinal kp/ki would (see .agents PR #2347 note).
    # Persistence is intentionally dropped vs. the mvl reference: no openpilot Params
    # dependency (that would be opendbc reaching up into the app layer), so the factors
    # re-learn each drive from these initial values.
    #
    # Speed-scheduled gas baseline + scalar trim: the converged gasfactor is speed-dependent
    # in a check-mark shape - high at launch (~0.9 at 0 m/s), dipping to a low-cruise minimum
    # (~0.35 at ~8 m/s), then rising with speed (~0.6 at ~22 m/s). See GAS_FACTOR_SPEED_* for
    # the per-point sources. A single scalar can't sit in every regime, so the speed shape
    # lives in that fixed table and self.gasfactor is a residual trim (seeded 1.0) the loop
    # still learns live on top of it. effective gasfactor = base(vEgo) * self.gasfactor. This
    # keeps transitions between speeds instant (no re-learning lag), preserves the low-state
    # self-correcting design, and leaves un-driven speeds at the sane baseline instead of a
    # stale scalar.
    # Watch /carOutput/actuatorsOutput/gas (repurposed to the *effective* gasfactor) and
    # /brake (windfactor) in the jotpluggler Long - Learning tab. NOTE: the base table is
    # from one drive - re-verify and refine as more logs come in. Does NOT address the fast
    # transient factor swings around hard accel/decel events (that's a learn-rate lever).
    # TODO: delete excessive comments before trying to submit a PR.
    self.bosch_last_gas = 0
    # Low-pass the IMU pitch (0.5s), matching Toyota's carcontroller. orientationNED[1] is
    # noisy and feeds sin(pitch)*g straight into the gas feedforward, so raw pitch makes the
    # GAS_COMMAND jittery. We only take Toyota's noise-reduction filter, NOT its high-pass
    # (that amplifies transients for Toyota's PCM, which Honda's ECU doesn't share) or its
    # min(pitch,0) downhill clamp (we keep bidirectional grade comp - calibration is clean, no
    # offset to guard against - so uphill gas assist is preserved).
    # TODO: delete excessive comments before trying to submit a PR.
    self.pitch = FirstOrderFilter(0.0, 0.5, DT_CTRL)
    self.gasfactor = 1.0            # residual trim on top of the speed-scheduled baseline
    self.gasfactor_effective = 1.0  # base(vEgo) * trim, exposed in telemetry (updated in update())
    self.windfactor = 0.5
    # Saturation guards. NOTE: the mvl reference initializes *_before_maxgas but then reads
    # *_before_gasmax in update() - an uninitialized-attribute bug. We init all names used.
    self.gasfactor_before_maxgas = self.gasfactor_before_gasmax = self.gasfactor
    self.windfactor_before_maxgas = self.windfactor_before_brake = self.windfactor_before_gasmax = self.windfactor

    # Bosch extra-brake controller: integral-only (k_p=0) and one-directional (pos_limit=0,
    # can only ADD braking, never remove it). It supplements Honda Bosch's mushy internal
    # brake response when we're already asking to decelerate; it does not fight it. This is
    # the one piece that feeds ACCEL_COMMAND, so it's the thing to watch first on-road.
    # TODO: delete excessive comments before trying to submit a PR.
    self.brake_pid = PIDController(k_p=([0.], [0.]), k_i=([0.], [0.5]),
                                   pos_limit=0.0, neg_limit=-2.0, rate=50)
    self.brake_pid.reset()

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    hud_v_cruise = hud_control.setSpeed / CS.v_cruise_factor if hud_control.speedVisible else 255
    pcm_cancel_cmd = CC.cruiseControl.cancel

    # CUSTOM TUNE (ody-op-long): grade compensation from IMU pitch (Odyssey long control).
    # TODO: delete excessive comments before trying to submit a PR.
    min_gas = self.params.BOSCH_GAS_LOOKUP_BP[0]
    gas_pedal_force = 0.0
    if len(CC.orientationNED) == 3:
      self.pitch.update(CC.orientationNED[1])
    hill_brake = math.sin(self.pitch.x) * ACCELERATION_DUE_TO_GRAVITY

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
    # CUSTOM TUNE (ody-op-long): aero drag decel in real m/s^2 units, used by the Odyssey
    # gas feedforward below (scaled live by self.windfactor). Base values from the mvl
    # reference; exact magnitude isn't critical since windfactor learns the residual.
    # TODO: delete excessive comments before trying to submit a PR.
    wind_brake_ms2 = np.interp(CS.out.vEgo, [0.0, 13.4, 22.4, 31.3, 40.2], [0.000, 0.049, 0.136, 0.267, 0.441])
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

            # gas_pedal_force = desired accel + aero drag + grade, all in m/s^2. Uses raw
            # accel (not self.accel) so the brake_pid addon doesn't feed the gas side. Computed
            # before the brake_pid so it can gate it (see below).
            gas_pedal_force = accel + wind_brake_ms2 * self.windfactor + hill_brake

            # Supplemental brake authority: integral-only, one-directional. Gate on
            # gas_pedal_force (the same grade/drag-compensated quantity the gas/brake domain
            # switch uses at speed), NOT raw accel. When accel is mildly negative but grade/drag
            # pushes gas_pedal_force back above the threshold, we're still in the GAS domain
            # (BRAKE_REQUEST=0) and this brake is never applied. Gating on raw accel there let
            # this integral-only PID wind up to ~-1.1 m/s^2 of unused brake (seen post-launch on
            # route 805f87f5.../0000008f, t~277-291 with gas_pedal_force ~+0.05) - a landmine
            # that would dump a hard phantom brake the instant gas_pedal_force dipped below the
            # threshold (e.g. a hill crest). Only wind up when we'll actually brake.
            if (gas_pedal_force < min_gas) and (CS.out.vEgo > 1e-3):
              brake_addon = self.brake_pid.update(error=accel - CS.out.aEgo, speed=CS.out.vEgo)
              targetaccel = min(accel, accel + brake_addon)
            else:
              self.brake_pid.reset()
              targetaccel = accel
            self.accel = float(np.clip(targetaccel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))

            # Speed-scheduled baseline gasfactor; self.gasfactor is the live-learned residual
            # trim on top of it. effective = base(vEgo) * trim (see __init__ for rationale).
            base_gasfactor = float(np.interp(CS.out.vEgo, GAS_FACTOR_SPEED_BP, GAS_FACTOR_SPEED_V))

            # Live-learn the gas/wind correction factors, only while openpilot controls the
            # gas (longControlState == pid) and the driver's foot is off the pedal.
            if (actuators.longControlState == LongCtrlState.pid) and (not CS.out.gasPressed):
              gas_error = self.accel - CS.out.aEgo
              if gas_error != 0.0 and gas_pedal_force > min_gas:
                # Odyssey-specific learn rate (our own tuning, commit bc2dc47f1 - re-verify,
                # not an mvl-tested value): faster at low speed, slower at cruise. Nudges the
                # residual trim; the speed shape itself is carried by base_gasfactor.
                learn_speed = int(np.interp(CS.out.vEgo, [0., 15., 25.], [150, 200, 400]))
                self.gasfactor = np.clip(self.gasfactor + gas_error / learn_speed * (gas_pedal_force - min_gas), 0.01, 3.0)
              if gas_error != 0.0 and (not CS.out.brakePressed) and (CS.out.vEgo > 0.0):
                wind_learn_speed = 500  # Odyssey-specific (our tuning, re-verify)
                wind_adjust = 1 + wind_brake_ms2 / wind_learn_speed
                self.windfactor = np.clip(self.windfactor * (wind_adjust if (gas_error > 0) else 1.0 / wind_adjust), 0.1, 3.0)
              if gas_pedal_force <= min_gas:  # don't reduce windfactor while braking, allow increases
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
            self.gas = float(np.interp((gas_pedal_force - min_gas) * self.gasfactor_effective + min_gas,
                                       self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))
            # Limit gas ramp to 60 units/frame (matches stock; higher can make the powertrain
            # ignore the gas command entirely).
            max_gas = max(60, self.bosch_last_gas + 60)
            self.gas = min(self.gas, max_gas)
            self.bosch_last_gas = self.gas
            # PR #2342 stop-hold: pass vEgo so the gas/brake switch raises its threshold at
            # low speed and keeps braking through a stop (prevents pre-stop brake release).
            stop_hold_vego = CS.out.vEgo
          else:
            # Stock behavior for all other Bosch Hondas. gas_pedal_force == accel keeps the
            # gas/brake switch in create_acc_commands identical to upstream.
            self.accel = float(np.clip(accel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))
            self.gas = float(np.interp(accel, self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))
            gas_pedal_force = accel
            stop_hold_vego = None  # stock fixed threshold for non-Odyssey Bosch

          stopping = actuators.longControlState == LongCtrlState.stopping
          self.stopping_counter = self.stopping_counter + 1 if stopping else 0
          can_sends.extend(hondacan.create_acc_commands(self.packer, self.CAN, CC.enabled, CC.longActive, self.accel, self.gas,
                                                        self.stopping_counter, self.CP.carFingerprint, gas_pedal_force, stop_hold_vego))
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
