import math
from collections import deque

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, rate_limit, make_tester_present_msg, structs
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.honda import hondacan
from opendbc.car.honda.values import CAR, CruiseButtons, HondaFlags, CarControllerParams
from opendbc.car.interfaces import CarControllerBase

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState


ODYSSEY_LOW_SPEED_DOMAIN_VEGO = 5.0
# Include the float32 representation of a nominal -0.10 m/s2 carControl request.
ODYSSEY_GAS_BRIDGE_ENTRY = -0.101
ODYSSEY_GAS_BRIDGE_COMMAND = -60.0
ODYSSEY_GRADE_FILTER_TAU = 0.5
ODYSSEY_GRADE_RAMP_ACCEL = 0.30
ODYSSEY_GRADE_GAIN = 0.6
ODYSSEY_UPHILL_GAS_ACCEL_MAX = 1.0
ODYSSEY_NEGATIVE_GRADE_ACCEL_MAX = 0.10
ODYSSEY_RESPONSE_DELAY_FRAMES = 25  # 0.5 s Bosch longitudinal delay at 50 Hz
ODYSSEY_RESPONSE_MAX_COUNTS = 100.0
ODYSSEY_RESPONSE_COUNTS_PER_ACCEL = 200.0
ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL = 500.0
ODYSSEY_RESPONSE_SLEW_COUNTS = 10.0
ODYSSEY_BRAKE_GRADE_GAIN = 0.3
ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS = 200.0
# Keep mild negative road-speed requests out of friction braking. Brake selection remains based on
# the raw request; bounded uphill gas demand may delay only an already-active gas release.
ODYSSEY_ROAD_BRAKE_ENTRY = -0.30


def odyssey_command_domains(accel, speed, previous_brake=False, previous_gas=False, bridge_active=False,
                            gas_accel=None):
  """Keep low-speed stop authority and separate road-speed coast from friction braking."""
  gas_accel = accel if gas_accel is None else gas_accel
  gas_selected = accel > 0.0
  if speed >= ODYSSEY_LOW_SPEED_DOMAIN_VEGO:
    gas_selected |= not previous_brake and not previous_gas and ODYSSEY_GAS_BRIDGE_ENTRY <= accel < 0.0
    gas_selected |= previous_gas and gas_accel > CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]
    if bridge_active and accel < ODYSSEY_GAS_BRIDGE_ENTRY:
      gas_selected = False
    brake_selected = accel < ODYSSEY_ROAD_BRAKE_ENTRY or (previous_brake and accel < 0.0)
  else:
    brake_selected = accel <= 0.0
  # A positive request must release the brake domain immediately so the two commands remain
  # mutually exclusive, matching the stateful brake-permit behavior used by other OEM ports.
  return gas_selected, brake_selected and not gas_selected


def odyssey_gas_command(accel, mapped_gas, gas_selected, previous_gas, bridge_active):
  """Pre-activate the gas domain with the stock-supported negative transition command."""
  bridge_active = gas_selected and accel < 0.0 and (bridge_active or not previous_gas)
  gas = ODYSSEY_GAS_BRIDGE_COMMAND if bridge_active else mapped_gas
  return (gas if gas_selected else 0.0), bridge_active


def odyssey_uphill_gas_accel_with_cap(accel, pitch):
  """Return the grade-relative gas-map input and the fraction limited by its uphill cap."""
  if pitch > 0.0 and ODYSSEY_ROAD_BRAKE_ENTRY < accel < 0.0:
    gas_split = CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]
    if accel <= gas_split:
      x = (accel - ODYSSEY_ROAD_BRAKE_ENTRY) / (gas_split - ODYSSEY_ROAD_BRAKE_ENTRY)
    else:
      x = accel / gas_split
    grade_weight = x * x * (3.0 - 2.0 * x)
    grade_accel = min(math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * ODYSSEY_GRADE_GAIN,
                      ODYSSEY_NEGATIVE_GRADE_ACCEL_MAX)
    return accel + grade_accel * grade_weight, 0.0
  if accel <= 0.0:
    return accel, 0.0
  x = min(accel / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
  grade_weight = x * x * (3.0 - 2.0 * x)
  grade_accel = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * grade_weight * ODYSSEY_GRADE_GAIN
  if grade_accel >= 0.0:
    mapped = max(accel, min(accel + grade_accel, ODYSSEY_UPHILL_GAS_ACCEL_MAX))
    cap_fraction = float(np.clip((accel + grade_accel - mapped) / grade_accel, 0.0, 1.0)) if grade_accel > 0.0 else 0.0
    return mapped, cap_fraction
  return max(CarControllerParams.BOSCH_GAS_LOOKUP_BP[0], accel + grade_accel), 0.0


def odyssey_uphill_gas_accel(accel, pitch):
  """Translate net acceleration to grade-relative gas demand without changing ACCEL_COMMAND."""
  return odyssey_uphill_gas_accel_with_cap(accel, pitch)[0]


def odyssey_brake_accel(accel, pitch):
  """Translate the requested net deceleration into Honda's grade-relative brake command."""
  if accel >= 0.0:
    return accel
  grade_accel = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * ODYSSEY_BRAKE_GRADE_GAIN
  return min(accel + grade_accel, 0.0)


def odyssey_low_speed_gas_command(gas, accel, speed):
  """Bound positive-gas feedforward where Odyssey acceleration exceeds the net request."""
  if accel <= 0.0 or gas <= 0.0:
    return gas

  def smoothstep(value):
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)

  speed_weight = smoothstep((speed - 8.0) / 4.0) * smoothstep((24.0 - speed) / 4.0)
  request_weight = smoothstep((accel - 0.4) / 0.4) * smoothstep((2.0 - accel) / 0.4)
  return max(0.0, gas - ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS * speed_weight * request_weight)


class OdysseyGasResponse:
  """Compare a delayed gas-domain request with the vehicle's measured response."""
  def __init__(self):
    self.correction = 0.0
    self.requests = deque(maxlen=ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
    self.pitch_peak = None
    self.gear = None
    self.error = FirstOrderFilter(0.0, 0.3, DT_CTRL * 2, initialized=False)

  def reset(self):
    self.__init__()

  def clear_observer(self):
    self.requests.clear()
    self.pitch_peak = None
    self.gear = None
    self.error = FirstOrderFilter(0.0, 0.3, DT_CTRL * 2, initialized=False)

  def update(self, request, aego, pitch, speed, gear, eligible, cap_fraction=0.0):
    if not eligible or not all(math.isfinite(v) for v in (request, aego, pitch, speed, cap_fraction)):
      self.reset()
      return 0.0

    if (self.gear is not None and gear != self.gear) or (self.pitch_peak is not None and pitch < self.pitch_peak - 0.01):
      self.clear_observer()
    self.gear = gear
    self.pitch_peak = pitch if self.pitch_peak is None else max(self.pitch_peak, pitch)
    self.requests.append(request)

    target = 0.0
    if len(self.requests) > ODYSSEY_RESPONSE_DELAY_FRAMES:
      delayed_request = self.requests[0]
      residual = self.error.update(delayed_request - aego)
      response_gain = ODYSSEY_RESPONSE_COUNTS_PER_ACCEL + (ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL -
                                                          ODYSSEY_RESPONSE_COUNTS_PER_ACCEL) * float(np.clip(cap_fraction, 0.0, 1.0))
      target = float(np.clip(response_gain * residual,
                             -ODYSSEY_RESPONSE_MAX_COUNTS, ODYSSEY_RESPONSE_MAX_COUNTS))
      if request < delayed_request - 0.08:
        target = min(target, 0.0)
      elif request > delayed_request + 0.08:
        target = max(target, 0.0)
    self.correction = float(np.clip(target, self.correction - ODYSSEY_RESPONSE_SLEW_COUNTS,
                                     self.correction + ODYSSEY_RESPONSE_SLEW_COUNTS))
    return self.correction


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


def compute_gas_brake(accel, speed, CP):
  if CP.flags & HondaFlags.BOSCH:
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
    self.tja_control = bool(CP.flags & HondaFlags.BOSCH_TJA_CONTROL)

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
    self.odyssey_brake_selected = False
    self.odyssey_gas_selected = False
    self.odyssey_gas_bridge_active = False
    self.odyssey_gas_response = OdysseyGasResponse()
    self.odyssey_pitch = FirstOrderFilter(0.0, ODYSSEY_GRADE_FILTER_TAU, DT_CTRL)

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    hud_v_cruise = hud_control.setSpeed / CS.v_cruise_factor if hud_control.speedVisible else 255
    pcm_cancel_cmd = CC.cruiseControl.cancel
    odyssey_pitch_valid = self.CP.carFingerprint == CAR.HONDA_ODYSSEY_5G_MMR and len(CC.orientationNED) == 3
    if odyssey_pitch_valid:
      self.odyssey_pitch.update(CC.orientationNED[1])

    if CC.longActive:
      accel = actuators.accel
      gas, brake = compute_gas_brake(actuators.accel, CS.out.vEgo, self.CP)
    else:
      accel = 0.0
      gas, brake = 0.0, 0.0
      self.odyssey_brake_selected = False
      self.odyssey_gas_selected = False
      self.odyssey_gas_bridge_active = False
      self.odyssey_gas_response.reset()

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
    apply_torque = int(np.clip(-limited_torque, -1.0, 1.0) * self.params.STEER_MAX)

    # Send CAN commands
    can_sends = []

    # tester present - w/ no response (keeps radar disabled)
    if self.CP.flags & HondaFlags.BOSCH and not (self.CP.flags & HondaFlags.BOSCH_RADARLESS) and self.CP.openpilotLongitudinalControl:
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
    elif self.CP.flags & HondaFlags.NIDEC_ALT_PCM_ACCEL:
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
      if self.frame % 2 == 0 and not (self.CP.flags & (HondaFlags.BOSCH_RADARLESS | HondaFlags.BOSCH_CANFD)):
        can_sends.append(hondacan.create_bosch_supplemental_1(self.packer, self.CAN))
      # If using stock ACC, spam cancel command to kill gas when OP disengages.
      if pcm_cancel_cmd:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.CANCEL, self.CP))
      elif CC.cruiseControl.resume:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.RES_ACCEL, self.CP))

    else:
      # Send gas and brake commands.
      if self.frame % 2 == 0:
        ts = self.frame * DT_CTRL

        if self.CP.flags & HondaFlags.BOSCH:
          self.accel = float(np.clip(accel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))
          self.gas = float(np.interp(accel, self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))
          stopping = actuators.longControlState == LongCtrlState.stopping
          gas_domain = None
          brake_domain = None
          if self.CP.carFingerprint == CAR.HONDA_ODYSSEY_5G_MMR:
            previous_gas = self.odyssey_gas_selected
            gas_accel = accel
            cap_fraction = 0.0
            if (odyssey_pitch_valid and CC.orientationNED[1] * self.odyssey_pitch.x > 0.0 and
                actuators.longControlState == LongCtrlState.pid and not CS.out.gasPressed):
              gas_accel, cap_fraction = odyssey_uphill_gas_accel_with_cap(accel, self.odyssey_pitch.x)
            gas_selected, brake_selected = odyssey_command_domains(accel, CS.out.vEgo,
                                                                    self.odyssey_brake_selected,
                                                                    previous_gas,
                                                                    self.odyssey_gas_bridge_active,
                                                                    gas_accel)
            self.odyssey_brake_selected = brake_selected
            self.odyssey_gas_selected = gas_selected
            if gas_selected and gas_accel != accel:
              self.gas = float(np.interp(gas_accel, self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))
            feedback_eligible = (CC.longActive and gas_selected and previous_gas and
                                 not self.odyssey_gas_bridge_active and odyssey_pitch_valid and
                                 actuators.longControlState == LongCtrlState.pid and
                                 not CS.out.gasPressed and not CS.out.brakePressed and
                                 CS.out.vEgo >= 8.0 and self.gas > 0.0)
            correction = self.odyssey_gas_response.update(accel, CS.out.aEgo, self.odyssey_pitch.x,
                                                            CS.out.vEgo, CS.out.gearShifter, feedback_eligible, cap_fraction)
            if feedback_eligible:
              self.gas = float(np.clip(self.gas + correction, 0.0, self.params.BOSCH_GAS_LOOKUP_V[-1]))
            if gas_selected and actuators.longControlState == LongCtrlState.pid and not CS.out.gasPressed:
              self.gas = odyssey_low_speed_gas_command(self.gas, accel, CS.out.vEgo)
            if (brake_selected and CS.out.vEgo >= ODYSSEY_LOW_SPEED_DOMAIN_VEGO and odyssey_pitch_valid and
                actuators.longControlState == LongCtrlState.pid and not CS.out.brakePressed):
              brake_accel = odyssey_brake_accel(accel, self.odyssey_pitch.x)
              self.accel = float(np.clip(brake_accel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))
            # The low-speed domain keeps every non-positive request on the brake side without
            # reshaping the controller command.
            self.gas, self.odyssey_gas_bridge_active = odyssey_gas_command(accel, self.gas, gas_selected,
                                                                            previous_gas, self.odyssey_gas_bridge_active)
            gas_domain = gas_selected
            brake_domain = brake_selected

          self.stopping_counter = self.stopping_counter + 1 if stopping else 0
          can_sends.extend(hondacan.create_acc_commands(self.packer, self.CAN, CC.enabled, CC.longActive, self.accel, self.gas,
                                                        self.stopping_counter, self.CP, gas_domain, brake_domain))
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
        if self.CP.flags & HondaFlags.BOSCH and not (self.CP.flags & HondaFlags.BOSCH_RADARLESS):
          can_sends.append(hondacan.create_radar_hud(self.packer, self.CAN.pt))
        if self.CP.carFingerprint == CAR.HONDA_CIVIC_BOSCH:
          can_sends.append(hondacan.create_legacy_brake_command(self.packer, self.CAN.pt))
        if not (self.CP.flags & HondaFlags.BOSCH):
          self.speed = pcm_speed
          self.gas = pcm_accel / self.params.NIDEC_GAS_MAX

    new_actuators = actuators.as_builder()
    new_actuators.speed = self.speed
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas
    new_actuators.brake = self.brake
    new_actuators.torque = self.last_torque
    new_actuators.torqueOutputCan = apply_torque

    self.frame += 1
    return new_actuators, can_sends
