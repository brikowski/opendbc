import math
import unittest
from types import SimpleNamespace

from opendbc.can import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, gen_empty_fingerprint, structs
from opendbc.car.honda.carcontroller import (ODYSSEY_BRAKE_GRADE_GAIN, ODYSSEY_GAS_BRIDGE_COMMAND, ODYSSEY_GRADE_GAIN,
                                             ODYSSEY_GRADE_RAMP_ACCEL, ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS,
                                             ODYSSEY_ZERO_GAS_GRADE_GAIN,
                                             ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL, ODYSSEY_RESPONSE_COUNTS_PER_ACCEL,
                                             ODYSSEY_RESPONSE_DELAY_FRAMES, ODYSSEY_RESPONSE_MAX_COUNTS, ODYSSEY_RESPONSE_SLEW_COUNTS,
                                             ODYSSEY_ROAD_BRAKE_ENTRY, ODYSSEY_STEER_ERROR_FRAMES,
                                             ODYSSEY_UPHILL_GAS_ACCEL_MAX, CarController, OdysseyBrakeRelease,
                                             OdysseySteeringAuthority,
                                             OdysseyCoastResponse, OdysseyGasCoastRelease, OdysseyGasResponse, odyssey_brake_accel,
                                             odyssey_command_domains, odyssey_gas_command, odyssey_grade_weight,
                                             odyssey_creep_brake_accel, odyssey_low_speed_gas_command,
                                             odyssey_uphill_gas_accel_with_cap)
from opendbc.car.honda.values import CAR, CarControllerParams, HondaFlags
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import DBC


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"


class TestOdysseySteeringAuthority(unittest.TestCase):
  def test_parameter_rescaling_preserves_ordinary_physical_map(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    params = CarControllerParams(CP)
    self.assertEqual(params.STEER_MAX, 3840)
    self.assertEqual(params.STEER_DELTA_UP * params.STEER_MAX, 3 * 2560)
    self.assertAlmostEqual(CP.lateralTuning.torque.latAccelFactor, 1.35, places=6)
    self.assertAlmostEqual(CP.lateralTuning.torque.friction, 0.2 / 1.5, places=6)
    for demand in (0.2, 0.5, 0.8):
      old_count = (demand + 0.2 * 0.9) / 0.9 * 2560
      new_count = (demand + CP.lateralTuning.torque.friction * CP.lateralTuning.torque.latAccelFactor) / CP.lateralTuning.torque.latAccelFactor * 3840
      self.assertAlmostEqual(old_count, new_count, delta=0.01)

  def test_extra_range_requires_persistent_same_direction_error(self):
    authority = OdysseySteeringAuthority(3840 / 1.35)
    curvature = 1.8 / 31.0**2
    current = curvature - 0.3 / 31.0**2
    for _ in range(ODYSSEY_STEER_ERROR_FRAMES - 1):
      self.assertEqual(authority.limit(3500, curvature, current, 31.0, True, 3840), 2560)
    self.assertGreater(authority.limit(3500, curvature, current, 31.0, True, 3840), 2560)
    self.assertEqual(authority.limit(3500, curvature, current, 14.9, True, 3840), 2560)
    lower_curvature = 1.8 / 22.0**2
    lower_current = lower_curvature - 0.3 / 22.0**2
    for _ in range(ODYSSEY_STEER_ERROR_FRAMES - 1):
      self.assertEqual(authority.limit(3500, lower_curvature, lower_current, 22.0, True, 3840), 2560)
    self.assertGreater(authority.limit(3500, lower_curvature, lower_current, 22.0, True, 3840), 2560)
    self.assertEqual(authority.limit(3500, curvature, current, 31.0, False, 3840), 2560)
    self.assertEqual(authority.limit(-3500, -curvature, -current, 31.0, True, 3840), -2560)
    self.assertEqual(authority.limit(3500, -curvature, -current, 31.0, True, 3840), 2560)
    self.assertEqual(authority.limit(3500, curvature, curvature, 31.0, True, 3840), 2560)
    self.assertEqual(authority.limit(2000, curvature, current, 31.0, True, 3840), 2000)

  def test_steering_authority_speed_scope(self):
    for direction in (-1, 1):
      for speed in (8.5, 14.9, 15.0, 16.5, 19.9, 20.0, 33.0, 33.1):
        with self.subTest(direction=direction, speed=speed):
          authority = OdysseySteeringAuthority(3840 / 1.35)
          curvature = direction * 1.5 / speed**2
          current = curvature - direction * 0.4 / speed**2
          for _ in range(ODYSSEY_STEER_ERROR_FRAMES):
            output = authority.limit(direction * 3500, curvature, current, speed, True, 3840)
          if 15.0 <= speed <= 33.0:
            self.assertGreater(abs(output), 2560)
          else:
            self.assertEqual(abs(output), 2560)

  def test_extra_range_does_not_withdraw_on_growing_turn_demand(self):
    for direction in (-1, 1):
      with self.subTest(direction=direction):
        authority = OdysseySteeringAuthority(3840 / 1.35)
        speed = 31.5

        def command(demand, error, direction=direction, speed=speed, authority=authority):
          curvature = direction * demand / speed**2
          current = curvature - direction * error / speed**2
          return authority.limit(direction * 3840, curvature, current, speed, True, 3840)

        for _ in range(ODYSSEY_STEER_ERROR_FRAMES):
          command(2.4, 0.7)
        self.assertEqual(command(2.4, 0.7), direction * 3840)
        self.assertEqual(command(2.6, 0.7), direction * 3840)
        self.assertEqual(command(3.1, 1.0), direction * 3840)
        self.assertEqual(command(3.1, 0.0), direction * 2560)

  def test_steering_wire_preserves_persistent_request_through_ramp_and_turn_exit(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    for direction, alpha_long, bus in ((-1, True, 1), (1, True, 1), (-1, False, 0), (1, False, 0)):
      with self.subTest(direction=direction, alpha_long=alpha_long):
        CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], alpha_long, False, False)
        dbc = DBC[CP.carFingerprint][Bus.pt]
        controller = CarController({Bus.pt: dbc}, CP)
        parser = CANParser(dbc, [('STEERING_CONTROL', 0)], bus)
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)
        control = structs.CarControl()
        control.enabled = control.latActive = True
        control.actuators.torque = -direction * 0.9
        state = structs.CarState()
        state.vEgo = 22.0
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_engine_torque_estimate=math.nan, odyssey_car_gas=math.nan,
                             odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=0,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
        counts = []
        for frame in range(190):
          demand = 1.4 if frame < 100 else 1.25
          control.actuators.curvature = direction * demand / state.vEgo**2
          control.currentCurvature = direction * (demand - 0.4) / state.vEgo**2
          if frame >= 160:
            control.currentCurvature = control.actuators.curvature
          output, sends = controller.update(control.as_reader(), CS, frame * 10_000_000)
          steering = [(addr, dat, src) for addr, dat, src in sends if addr == 0xE4 and src == bus]
          self.assertEqual(len(steering), 1)
          for addr, dat, src in steering:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, src, dat)))
          parser.update([(frame * 10_000_000, steering)])
          counts.append(parser.vl['STEERING_CONTROL']['STEER_TORQUE'])
          self.assertEqual(counts[-1], output.torqueOutputCan)
          self.assertLessEqual(abs(counts[-1]), 3840)
          if 100 <= frame < 160:
            self.assertAlmostEqual(counts[-1], counts[99], delta=1)
          if frame >= 160:
            self.assertGreaterEqual(abs(counts[-1]), 2560)
        self.assertGreater(abs(counts[34]), 2560)
        self.assertGreater(abs(counts[99]), 3000)
        self.assertEqual(abs(counts[-1]), 2560)
        self.assertLessEqual(max(abs(b - a) for a, b in zip(counts, counts[1:], strict=False)), 77)

  def test_steering_wire_and_output_follow_bounded_authority_in_both_bus_modes(self):
    for alpha_long, bus, speed in ((True, 1, 31.0), (False, 0, 31.0), (True, 1, 22.0), (False, 0, 22.0),
                                  (True, 1, 16.5), (False, 0, 16.5)):
      with self.subTest(alpha_long=alpha_long, speed=speed):
        CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], alpha_long, False, False)
        dbc = DBC[CP.carFingerprint][Bus.pt]
        controller = CarController({Bus.pt: dbc}, CP)
        parser = CANParser(dbc, [('STEERING_CONTROL', 0)], bus)
        control = structs.CarControl()
        control.enabled = True
        control.latActive = True
        control.actuators.torque = -1.0
        control.actuators.curvature = 1.8 / speed**2
        control.currentCurvature = control.actuators.curvature - 0.3 / speed**2
        state = structs.CarState()
        state.vEgo = speed
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_engine_torque_estimate=math.nan, odyssey_car_gas=math.nan,
                             odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=0,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

        def step(frame, controller=controller, control=control, CS=CS, bus=bus, parser=parser):
          output, sends = controller.update(control.as_reader(), CS, frame * 10_000_000)
          steering = [(addr, dat, src) for addr, dat, src in sends if addr == 0xE4]
          self.assertEqual(len(steering), 1)
          self.assertEqual(steering[0][2], bus)
          parser.update([(frame * 10_000_000, steering)])
          self.assertEqual(parser.vl['STEERING_CONTROL']['STEER_TORQUE'], output.torqueOutputCan)
          self.assertAlmostEqual(output.torque, -output.torqueOutputCan / 3840, places=6)
          return output.torqueOutputCan

        counts = [step(frame) for frame in range(200)]
        self.assertLessEqual(max(abs(right - left) for left, right in zip(counts, counts[1:], strict=False)), 77)
        self.assertGreater(counts[-1], 2560)
        self.assertLessEqual(counts[-1], 3840)
        state.steerFaultTemporary = True
        fault_release = [step(frame) for frame in range(200, 215)]
        self.assertLessEqual(max(abs(right - left) for left, right in zip(fault_release, fault_release[1:], strict=False)), 77)
        self.assertEqual(fault_release[-1], 2560)
        state.steerFaultTemporary = False
        for frame in range(215, 250):
          step(frame)
        state.steeringPressed = True
        falling = [step(frame) for frame in range(250, 265)]
        self.assertLessEqual(max(abs(right - left) for left, right in zip(falling, falling[1:], strict=False)), 77)
        self.assertEqual(falling[-1], 2560)
        state.steeringPressed = False
        control.actuators.torque = 1.0
        control.actuators.curvature *= -1
        control.currentCurvature *= -1
        self.assertLessEqual(step(265), 2560)
        control.latActive = False
        self.assertEqual(step(266), 0)


class TestOdysseyLongitudinal(unittest.TestCase):
  def test_creep_braking_is_bounded_monotone_and_releases_with_request(self):
    for speed in (0.0, 0.5, 1.0, 1.5, 2.0, 5.0):
      requests = [-3.5 + i * 0.005 for i in range(741)]
      commands = [odyssey_creep_brake_accel(q, speed) for q in requests]
      self.assertTrue(all(left <= right + 1e-12 for left, right in zip(commands, commands[1:], strict=False)))
      for q, command in zip(requests, commands, strict=True):
        self.assertLessEqual(q - command, 0.25 + 1e-12)
        self.assertGreaterEqual(q - command, -1e-12)
        if q <= -0.8 or q >= 0.0 or speed >= 2.0:
          self.assertAlmostEqual(command, q)
    self.assertAlmostEqual(odyssey_creep_brake_accel(-0.3, 0.6), -0.55)
    self.assertAlmostEqual(odyssey_creep_brake_accel(-0.3, 1.5), -0.425)
    for speed in (-1.0, math.nan, math.inf):
      self.assertEqual(odyssey_creep_brake_accel(-0.3, speed), -0.3)
    for request in (-0.8, -0.3, 0.0):
      self.assertLess(abs(odyssey_creep_brake_accel(request - 1e-6, 0.5) -
                          odyssey_creep_brake_accel(request + 1e-6, 0.5)), 4e-6)

  def test_nonfinite_pitch_falls_back_to_raw_command_and_recovers(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request in (-0.6, 0.6):
      for invalid_pitch in (math.nan, math.inf, -math.inf):
        with self.subTest(request=request, invalid_pitch=invalid_pitch):
          controller = CarController({Bus.pt: dbc}, CP)
          parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
          control = structs.CarControl()
          control.enabled = True
          control.longActive = True
          control.actuators.accel = request
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
          state = structs.CarState()
          state.vEgo = 16.0
          CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                               odyssey_engine_torque_estimate=math.nan, odyssey_car_gas=math.nan,
                               odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=3,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

          def step(pitch, control=control, controller=controller, CS=CS, parser=parser):
            control.orientationNED = [0.0, pitch, 0.0]
            acc = []
            for _ in range(2):
              frame = controller.frame
              _, sends = controller.update(control.as_reader(), CS, frame * 10_000_000)
              acc.extend((addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1)
            self.assertEqual(len(acc), 1)
            parser.update([(frame * 10_000_000, acc)])
            return parser.vl['ACC_CONTROL']

          step(0.04)
          invalid = step(invalid_pitch)
          self.assertAlmostEqual(invalid['ACCEL_COMMAND'], request, delta=0.01)
          self.assertEqual(invalid['BRAKE_REQUEST'], request < 0.0)
          if request > 0.0:
            raw_gas = float((request - CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]) /
                            (CarControllerParams.BOSCH_GAS_LOOKUP_BP[-1] - CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]) *
                            CarControllerParams.BOSCH_GAS_LOOKUP_V[-1])
            self.assertAlmostEqual(invalid['GAS_COMMAND'], odyssey_low_speed_gas_command(raw_gas, request, state.vEgo), delta=1)
          restored = step(0.04)
          self.assertTrue(math.isfinite(controller.odyssey_pitch.x))
          self.assertTrue(math.isfinite(restored['ACCEL_COMMAND']))

  def test_received_engine_torque_is_parsed_from_odyssey_powertrain_bus(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    parser = CarState(CP).get_can_parsers(CP)[Bus.pt]
    dbc = DBC[CP.carFingerprint][Bus.pt]
    packer = CANPacker(dbc)
    now_nanos = 1_000_000_000
    parser.update([(now_nanos, [packer.make_can_msg('GAS_PEDAL_2', parser.bus,
                                                     {'ENGINE_TORQUE_ESTIMATE': -145, 'CAR_GAS': 0})])])
    self.assertEqual(parser.vl['GAS_PEDAL_2']['ENGINE_TORQUE_ESTIMATE'], -145)
    self.assertEqual(parser.vl['GAS_PEDAL_2']['CAR_GAS'], 0)
    self.assertEqual(parser.ts_nanos['GAS_PEDAL_2']['ENGINE_TORQUE_ESTIMATE'], now_nanos)

  def test_low_speed_positive_gas_trim_is_bounded_and_monotone(self):
    def gas(accel):
      return (accel + 0.2) / 2.2 * 2000.0
    self.assertEqual(ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS, 200.0)
    self.assertEqual(odyssey_low_speed_gas_command(gas(1.1), 1.1, 14.0), gas(1.1) - 200.0)
    for speed in (0.0, 8.0, 24.0, 32.0):
      self.assertEqual(odyssey_low_speed_gas_command(gas(1.1), 1.1, speed), gas(1.1))
    for accel in (0.4, 2.0):
      self.assertEqual(odyssey_low_speed_gas_command(gas(accel), accel, 14.0), gas(accel))
    self.assertEqual(odyssey_low_speed_gas_command(-60.0, -0.1, 14.0), -60.0)
    self.assertEqual(odyssey_low_speed_gas_command(0.0, 0.0, 14.0), 0.0)
    values = [odyssey_low_speed_gas_command(gas(accel), accel, 14.0) for accel in (i * 0.01 for i in range(201))]
    self.assertTrue(all(left <= right for left, right in zip(values, values[1:], strict=False)))
    for speed in (8.0, 12.0, 20.0, 24.0):
      self.assertLess(abs(odyssey_low_speed_gas_command(gas(1.1), 1.1, speed - 1e-4) -
                          odyssey_low_speed_gas_command(gas(1.1), 1.1, speed + 1e-4)), 0.01)

  def test_uphill_gas_load_is_continuous_through_negative_transition_and_bounded_at_high_request(self):
    pitch = 0.03
    grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY
    load = min(grade / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
    zero_grade = grade * ODYSSEY_ZERO_GAS_GRADE_GAIN * load * load
    self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(0.0, pitch)[0], zero_grade)
    self.assertGreater(odyssey_uphill_gas_accel_with_cap(-0.10, pitch)[0], -0.10)
    self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(-0.20, pitch)[0], -0.20 + zero_grade)
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(ODYSSEY_ROAD_BRAKE_ENTRY, pitch)[0], ODYSSEY_ROAD_BRAKE_ENTRY)
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(0.3, 0.0)[0], 0.3)
    downhill_grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * 0.6
    self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(0.3, -pitch)[0], 0.3 - downhill_grade)
    self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(0.01, pitch)[0] - 0.01, zero_grade, delta=0.001)
    expected = ODYSSEY_GRADE_RAMP_ACCEL + math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * 0.6
    self.assertEqual(ODYSSEY_GRADE_GAIN, 0.6)
    self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(ODYSSEY_GRADE_RAMP_ACCEL, pitch)[0], expected)
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(1.2, pitch)[0], 1.2)
    self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(1.2, -pitch)[0], 1.2 - downhill_grade)
    self.assertLessEqual(odyssey_uphill_gas_accel_with_cap(0.83, 0.072)[0], ODYSSEY_UPHILL_GAS_ACCEL_MAX)

  def test_negative_request_uphill_gas_scales_with_grade_without_reversing_request_order(self):
    self.assertEqual(ODYSSEY_ZERO_GAS_GRADE_GAIN, 0.25)
    mild = odyssey_uphill_gas_accel_with_cap(-0.20, 0.02)[0]
    steep = odyssey_uphill_gas_accel_with_cap(-0.20, 0.07)[0]
    self.assertLess(mild, -0.10)
    self.assertGreater(steep, -0.10)
    self.assertGreater(odyssey_uphill_gas_accel_with_cap(-0.10, 0.07)[0], 0.0)
    for pitch in (0.001, 0.02, 0.07, 0.20):
      requests = [ODYSSEY_ROAD_BRAKE_ENTRY + i * 0.001 for i in range(501)]
      mapped = [odyssey_uphill_gas_accel_with_cap(request, pitch)[0] for request in requests]
      self.assertTrue(all(left <= right + 1e-10 for left, right in zip(mapped, mapped[1:], strict=False)))
      self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(-0.20 - 1e-6, pitch)[0],
                             odyssey_uphill_gas_accel_with_cap(-0.20 + 1e-6, pitch)[0], delta=5e-6)
      self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(-1e-6, pitch)[0],
                             odyssey_uphill_gas_accel_with_cap(0.0, pitch)[0], delta=5e-6)
      self.assertAlmostEqual(odyssey_uphill_gas_accel_with_cap(0.0, pitch)[0],
                             odyssey_uphill_gas_accel_with_cap(1e-6, pitch)[0], delta=5e-6)
    self.assertGreater(odyssey_uphill_gas_accel_with_cap(-0.10, 0.08)[0] - odyssey_uphill_gas_accel_with_cap(-0.20, 0.08)[0], 0.09)
    self.assertLessEqual(odyssey_uphill_gas_accel_with_cap(0.0, 0.50)[0], ODYSSEY_UPHILL_GAS_ACCEL_MAX)
    self.assertEqual(odyssey_grade_weight(0.0), 0.0)
    self.assertTrue(0.0 < odyssey_grade_weight(0.02) < 1.0)
    self.assertEqual(odyssey_grade_weight(0.05), 1.0)

  def test_uphill_gas_cap_feedback_weight_tracks_limited_grade_only(self):
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(-0.1, 0.03)[1], 0.0)
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(0.95, -0.03)[1], 0.0)
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(0.3, 0.03)[1], 0.0)
    request, pitch = 0.95, 0.04
    mapped, weight = odyssey_uphill_gas_accel_with_cap(request, pitch)
    grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * ODYSSEY_GRADE_GAIN
    load = min(grade / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
    load_weight = load * load * (3.0 - 2.0 * load)
    self.assertEqual(mapped, ODYSSEY_UPHILL_GAS_ACCEL_MAX)
    self.assertAlmostEqual(weight, (request + grade - mapped) / grade * load_weight)
    high_mapped, high_weight = odyssey_uphill_gas_accel_with_cap(1.2, pitch)
    self.assertEqual(high_mapped, 1.2)
    self.assertAlmostEqual(high_weight, load_weight)
    threshold = ODYSSEY_UPHILL_GAS_ACCEL_MAX - grade
    self.assertLess(odyssey_uphill_gas_accel_with_cap(threshold + 1e-5, pitch)[1], 0.001)
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(threshold - 1e-5, pitch)[1], 0.0)

  def test_uphill_gas_cap_feedback_fades_with_vanishing_grade(self):
    high_request = 1.2
    self.assertEqual(odyssey_uphill_gas_accel_with_cap(high_request, 0.0)[1], 0.0)
    self.assertLess(odyssey_uphill_gas_accel_with_cap(high_request, 1e-5)[1], 1e-3)
    self.assertGreater(odyssey_uphill_gas_accel_with_cap(high_request, 0.08)[1], 0.9)

  def test_gas_response_cap_weight_increases_only_bounded_feedback(self):
    self.assertGreater(ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL, ODYSSEY_RESPONSE_COUNTS_PER_ACCEL)
    baseline, capped = OdysseyGasResponse(), OdysseyGasResponse()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 20):
      base = baseline.update(0.95, 0.65, 0.04, 19.0, 1, True)
      previous = capped.correction
      extra = capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 1.0)
      self.assertLessEqual(abs(extra - previous), 10.0)
    self.assertAlmostEqual(base, 60.0)
    self.assertEqual(extra, ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertLessEqual(capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 0.0), extra)
    self.assertEqual(capped.update(0.95, 0.65, 0.04, 19.0, 1, False, 1.0), 0.0)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      self.assertEqual(capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 1.0), 0.0)
    self.assertGreater(capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 1.0), 0.0)

  def test_gas_response_waits_for_vehicle_delay_before_adding_gas(self):
    response = OdysseyGasResponse()

    def update(request=0.0, aego=-0.3, pitch=0.06, gear=1, eligible=True):
      return response.update(request, aego, pitch, 20.0, gear, eligible)

    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      self.assertEqual(update(), 0.0)
    self.assertGreater(update(), 0.0)
    for _ in range(100):
      update()
    self.assertGreater(response.correction, 0.0)
    self.assertLessEqual(response.correction, ODYSSEY_RESPONSE_MAX_COUNTS)

    # A stronger deceleration request must unwind the old positive correction.
    previous = response.correction
    self.assertLess(update(request=-0.2), previous)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 1):
      update(request=-0.2, aego=0.2)
    self.assertLess(response.correction, 0.0)
    self.assertEqual(update(eligible=False), 0.0)
    self.assertEqual(response.correction, 0.0)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      self.assertEqual(update(gear=2), 0.0)
    self.assertGreater(update(gear=2), 0.0)

  def test_gas_response_reduces_startup_overacceleration_without_adding_gas(self):
    response = OdysseyGasResponse()
    for _ in range(15):
      self.assertEqual(response.update(0.02, -0.15, -0.03, 30.0, 1, True), 0.0)
    for _ in range(5):
      previous = response.correction
      correction = response.update(-0.02, 0.15, -0.03, 30.0, 1, True)
      self.assertLess(correction, 0.0)
      self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
      self.assertLessEqual(abs(correction), ODYSSEY_RESPONSE_MAX_COUNTS)
    previous = response.correction
    self.assertGreater(response.update(0.04, 1.0, -0.03, 30.0, 1, True), previous)
    self.assertEqual(response.update(-0.02, 0.15, -0.03, 30.0, 1, False), 0.0)
    self.assertFalse(response.requests)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES - 1):
      previous = response.correction
      correction = response.update(0.02, 1.0, -0.03, 30.0, 1, True, cap_weight=1.0)
      self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
      self.assertLessEqual(abs(correction), ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertEqual(correction, -ODYSSEY_RESPONSE_MAX_COUNTS)

  def test_gas_response_rejects_falling_grade_and_stale_lead_request(self):
    response = OdysseyGasResponse()
    for frame in range(ODYSSEY_RESPONSE_DELAY_FRAMES * 2):
      assert response.update(0.0, -0.3, 0.06 - frame * 0.0005, 20.0, 1, True) == 0.0
    response.reset()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      assert response.update(0.3, -0.3, 0.06, 20.0, 1, True) == 0.0
    assert response.update(-0.3, -0.3, 0.06, 20.0, 1, True) == 0.0

  def test_gas_feedback_uses_fresh_received_target_gear_through_drive_shifts(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    packer = CANPacker(dbc)
    for blocked in (None, 'missing', 'stale', 'future', 'invalid_can', 'neutral', 'reverse'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP)
        CS = CarState(CP)
        parsers = CS.get_can_parsers(CP)
        output_parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = .2
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        for frame in range(82):
          now = 1_000_000_000 + frame * 10_000_000
          target = 6 if frame < 80 else 7
          stamp = now - 5_000_000
          if frame >= 80:
            target = 0 if blocked == 'neutral' else 13 if blocked == 'reverse' else target
            stamp = now - 60_000_000 if blocked == 'stale' else now + 1 if blocked == 'future' else stamp
          parsers[Bus.pt].update([(stamp, [packer.make_can_msg('GEARBOX_AUTO', 1,
                                     {'TRANS_TARGET_GEAR': target, 'GEAR_SHIFTER': 4})])])
          state = CS.update(parsers)
          state.vEgo = 20.
          state.aEgo = 0.
          state.canValid = not (frame >= 80 and blocked == 'invalid_can')
          CS.out = state
          if frame >= 80 and blocked == 'missing':
            CS.odyssey_target_gear_ts_nanos = 0
          _, sends = controller.update(control.as_reader(), CS, now)
          output_parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
          self.assertEqual(str(state.gearShifter), 'drive')
          self.assertAlmostEqual(output_parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], .2, delta=.01)
          self.assertEqual(output_parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
          self.assertGreater(output_parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
          if frame == 78:
            self.assertGreater(controller.odyssey_gas_response.correction, 0)
            self.assertEqual(len(controller.odyssey_gas_response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
          if frame >= 80:
            self.assertEqual(len(controller.odyssey_gas_response.requests), 1 if blocked is None else 0)
            if blocked is not None:
              self.assertEqual(controller.odyssey_gas_response.correction, 0)

  def test_gas_feedback_does_not_compare_across_received_transmission_activity(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for changed in (False, True):
      with self.subTest(changed=changed):
        packer = CANPacker(dbc)
        CS = CarState(CP)
        parsers = CS.get_can_parsers(CP)
        controller = CarController({Bus.pt: dbc}, CP)
        wire = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = .2
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        for frame in range(134):
          now = 1_000_000_000 + frame * 10_000_000
          activity = 110 if changed and frame >= 80 else 102
          parsers[Bus.pt].update([(now - 5_000_000, [packer.make_can_msg('GEARBOX_AUTO', 1,
                                     {'TRANS_TARGET_GEAR': 6, 'TRANS_SHIFT_ACTIVITY': activity, 'GEAR_SHIFTER': 4})])])
          state = CS.update(parsers)
          state.vEgo, state.aEgo, state.canValid = 20., 0., True
          CS.out = state
          if frame > 0:
            self.assertEqual(CS.odyssey_target_gear, 6)
            self.assertEqual(CS.odyssey_shift_activity, activity)
            self.assertEqual(CS.odyssey_target_gear_ts_nanos, now - 5_000_000)
          previous = controller.odyssey_gas_response.correction
          _, sends = controller.update(control.as_reader(), CS, now)
          if frame % 2 or frame == 0:
            continue
          wire.update([(now, sends)])
          self.assertAlmostEqual(wire.vl['ACC_CONTROL']['ACCEL_COMMAND'], .2, delta=.01)
          self.assertEqual(wire.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
          self.assertGreater(wire.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
          response = controller.odyssey_gas_response
          self.assertLessEqual(abs(response.correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          if frame == 78:
            self.assertAlmostEqual(response.correction, 40., delta=1e-5)
          if frame == 80:
            self.assertEqual(len(response.requests), 1 if changed else ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
            self.assertAlmostEqual(response.correction, 30. if changed else 40., delta=1e-5)
          if frame == 128 and changed:
            self.assertEqual(len(response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES)
            self.assertEqual(response.correction, 0.)
          if frame == 130:
            self.assertGreater(response.correction, 0.)

  def test_odyssey_low_speed_feedback_is_bounded_and_survives_speed_entry(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]

    def run(request, achieved, speed):
      controller = CarController({Bus.pt: dbc}, CP)
      parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
      control = structs.CarControl()
      control.enabled = True
      control.longActive = True
      control.actuators.accel = request
      control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
      control.orientationNED = [0.0, 0.0, 0.0]
      state = structs.CarState()
      state.vEgo = speed
      state.aEgo = achieved
      state.canValid = True
      CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                           odyssey_engine_torque_estimate=math.nan, odyssey_car_gas=math.nan,
                           odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=3,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

      def step(frame):
        CS.odyssey_target_gear_ts_nanos = frame * 10_000_000
        _, sends = controller.update(control.as_reader(), CS, frame * 10_000_000)
        messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
        if messages:
          parser.update([(frame * 10_000_000, messages)])
          self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], request, delta=0.01)
          self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
        return parser.vl['ACC_CONTROL']['GAS_COMMAND']

      values = [step(frame) for frame in range(100)]
      return controller, state, step, values

    _, _, _, below = run(0.75, 1.05, 4.9)
    moderate, state, step, values = run(0.75, 1.05, 7.0)
    self.assertLess(values[-1], below[-1] - 40)
    self.assertLessEqual(abs(moderate.odyssey_gas_response.correction), ODYSSEY_RESPONSE_MAX_COUNTS)
    state.vEgo = 8.0
    self.assertLessEqual(abs(step(100) - values[-1]), 10)

    high, _, _, high_values = run(1.95, 2.25, 7.0)
    self.assertEqual(high.odyssey_gas_response.correction, 0.0)
    self.assertFalse(high.odyssey_gas_response.requests)
    self.assertLess(high_values[-1], CarControllerParams(CP).BOSCH_GAS_LOOKUP_V[-1])

    response = OdysseyGasResponse()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 20):
      response.update(0.75, 1.05, 0.0, 7.0, 1, True)
    self.assertLess(response.correction, -40.0)
    for _ in range(20):
      previous = response.correction
      response.update(1.95, 2.25, 0.0, 7.0, 1, True, response_weight=0.0)
      self.assertLessEqual(abs(response.correction - previous), 10.0)
    self.assertEqual(response.correction, 0.0)
    self.assertFalse(response.requests)

  def test_negative_gas_bridge_only_enters_from_road_speed_coast(self):
    self.assertEqual(odyssey_command_domains(-0.10, 20.0), (True, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True, bridge_active=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True), (True, False))
    self.assertEqual(odyssey_command_domains(-0.05, 20.0, previous_brake=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.05, 4.0), (False, True))

  def test_passive_coast_forecast_is_prior_only_and_resets_on_ineligible_state(self):
    response = OdysseyCoastResponse()
    for frame in range(0, 50, 2):
      self.assertIsNone(response.update(frame, 7, 0.0, 9.0, True, True))
    self.assertFalse(response.warmup)
    for frame, accel in zip(range(50, 60, 2), [0.25, 0.24, 3.0, 0.26, 0.27], strict=True):
      self.assertIsNone(response.update(frame, 7, 0.0, accel, True, True))
    self.assertAlmostEqual(response.update(60, 7, 0.0, 0.28, True, True), 0.26)
    self.assertEqual(len(response.warmup[7]), 5)
    for frame in range(62, 80, 2):
      self.assertAlmostEqual(response.update(frame, 7, 0.0, 0.26 if frame == 70 else 9.0, True, True), 0.27)
    self.assertAlmostEqual(response.update(80, 7, 0.0, 0.25, True, True), 0.27)
    self.assertAlmostEqual(response.update(82, 7, 0.0, -3.0, False, True), 0.26)
    samples = list(response.warmup[7])
    self.assertAlmostEqual(response.update(90, 7, 0.0, -3.0, False, True), 0.26)
    self.assertEqual(list(response.warmup[7]), samples)
    self.assertIsNone(response.update(92, 8, 0.0, 0.0, False, True))
    self.assertIsNone(response.update(94, 7, 0.0, 0.0, False, False))
    self.assertFalse(response.drag)
    self.assertFalse(response.warmup)

  def test_passive_coast_forecast_tracks_changed_response_without_learning_an_outlier(self):
    response = OdysseyCoastResponse()
    for frame in range(0, 100, 2):
      response.update(frame, 7, 0.0, -0.15, True, True)
    self.assertAlmostEqual(response.update(100, 7, 0.0, 0.30, True, True), -0.15)
    for frame in range(102, 140, 2):
      response.update(frame, 7, 0.0, 0.30, True, True)
    self.assertAlmostEqual(response.update(140, 7, 0.0, 3.0, True, True), 0.30)
    self.assertAlmostEqual(response.update(142, 7, 0.0, 0.30, True, True), 0.30)
    for frame in range(144, 210, 2):
      self.assertAlmostEqual(response.update(frame, 7, 0.0, -3.0, False, True), 0.30)

  def test_downhill_gas_release_requires_response_and_preserves_brake_authority(self):
    response = OdysseyGasCoastRelease()
    for frame in range(0, 20, 2):
      self.assertFalse(response.update(frame, 0.10, 0.30, 25.0, -0.03, 9, 0.09, True, True))
    for frame in range(20, 40, 2):
      self.assertFalse(response.update(frame, -0.15, 0.30, 25.0, -0.03, 9, None, True, True))
    self.assertTrue(response.update(40, -0.15, 0.30, 25.0, -0.03, 9, 0.09, True, True))
    self.assertEqual(odyssey_command_domains(-0.15, 25.0, previous_gas=True, release_gas=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.40, 25.0, previous_gas=True, release_gas=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.15, 4.0, previous_gas=True, release_gas=True), (False, True))
    self.assertTrue(response.update(42, -0.15, 0.30, 25.0, -0.03, 9, 0.09, False, True))
    self.assertFalse(response.update(44, -0.15, -0.11, 25.0, -0.03, 9, 0.09, False, True))
    self.assertFalse(response.update(46, 0.10, 0.30, 25.0, -0.03, 9, 0.09, True, True))
    self.assertFalse(response.update(48, -0.15, 0.30, 25.0, -0.03, 9, 0.09, True, False))
    self.assertFalse(response.requests)

  def test_downhill_coast_requires_learned_response_or_prior_gas(self):
    response = OdysseyGasCoastRelease()
    for frame in range(0, 20, 2):
      response.update(frame, 0.10, 0.30, 25.0, -0.03, 9, 0.09, True, True)
    self.assertFalse(response.update(20, -0.15, 0.30, 25.0, 0.03, 9, 0.09, True, True))
    self.assertFalse(response.update(22, -0.15, 0.30, 25.0, -0.03, 9, None, False, True))
    self.assertTrue(response.update(23, -0.15, 0.30, 25.0, -0.03, 9, 0.09, False, True))
    self.assertFalse(response.update(24, -0.15, 0.30, 25.0, -0.03, 9, 0.09, True, False))

  def test_downhill_coast_follows_positive_demand_and_measured_response(self):
    for request, aego, passive in ((0.35, 0.60, 0.30), (0.05, 0.06, 0.30), (0.05, 0.60, 0.0)):
      response = OdysseyGasCoastRelease()
      self.assertTrue(response.update(0, 0.05, 0.60, 20.0, -0.04, 7, 0.30, True, True))
      self.assertEqual(odyssey_command_domains(0.05, 20.0, previous_gas=True, release_gas=True), (False, False))
      self.assertFalse(response.update(2, request, aego, 20.0, -0.04, 7, passive, False, True))
      self.assertEqual(odyssey_command_domains(request, 20.0, release_gas=response.active), (True, False))
    for passive in (None, float("inf"), float("-inf"), float("nan")):
      response = OdysseyGasCoastRelease()
      self.assertFalse(response.update(0, 0.05, 0.60, 20.0, -0.04, 7, passive, True, True, True))

  def test_learned_coast_does_not_pre_activate_gas_below_passive_acceleration(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    CI = CarInterface(CP.copy())
    CI.update([])
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP)
    for frame in range(0, 102, 2):
      controller.odyssey_coast_response.update(frame, 7, 0.0, -0.1, True, True)
    controller.frame = 102
    controller.odyssey_pitch.x = -0.04
    CI.CS.out = structs.CarState(vEgo=20.0, vEgoRaw=20.0, aEgo=0.30, canValid=True)
    CI.CS.odyssey_target_gear = 7
    CI.CS.odyssey_engine_torque_estimate = -100.0
    CI.CS.odyssey_car_gas = 0.0
    control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0.0, -0.04, 0.0],
                                 actuators=structs.CarControl.Actuators(accel=0.05, longControlState=structs.CarControl.Actuators.LongControlState.pid))
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    seen = 0
    for frame in range(6):
      now = 1_000_000_000 + frame * 10_000_000
      CI.CS.odyssey_engine_torque_ts_nanos = now - 5_000_000
      CI.CS.odyssey_target_gear_ts_nanos = now - 5_000_000
      _, sends = controller.update(control.as_reader(), CI.CS, now)
      if any(addr == 0x1DF for addr, _, _ in sends):
        parser.update([(now, sends)])
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], 0.05)
        seen += 1
    self.assertEqual(seen, 3)
    for frame in range(4):
      now = 1_100_000_000 + frame * 10_000_000
      CI.CS.out.canValid = frame >= 2
      CI.CS.odyssey_engine_torque_ts_nanos = now - 5_000_000
      CI.CS.odyssey_target_gear_ts_nanos = now - (60_000_000 if frame >= 2 else 5_000_000)
      _, sends = controller.update(control.as_reader(), CI.CS, now)
      if any(addr == 0x1DF for addr, _, _ in sends):
        parser.update([(now, sends)])
        self.assertGreater(parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], 0.05)

  def test_downhill_gas_floor_releases_without_coast_history_only_on_falling_request(self):
    def run(gas_floor, pitch=-0.04, aego=0.4, request=-0.12):
      response = OdysseyGasCoastRelease()
      for frame in range(0, 22, 2):
        response.update(frame, 0.02, aego, 21.0, pitch, 7, None, True, True, gas_floor)
      return response.update(22, request, aego, 21.0, pitch, 7, None, True, True, gas_floor)

    self.assertTrue(run(True))
    self.assertFalse(run(False))
    self.assertFalse(run(True, pitch=0.04))
    self.assertFalse(run(True, aego=-0.1))
    self.assertFalse(run(True, request=0.02))

  def test_exhausted_gas_command_releases_to_coast_without_requesting_brake(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    control = structs.CarControl()
    control.enabled = True
    control.orientationNED = [0.0, -0.04, 0.0]
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo = 21.0
    state.aEgo = 0.5
    state.canValid = True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_engine_torque_estimate=50.0, odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

    for frame in range(260):
      control.longActive = frame >= 100
      control.actuators.accel = 0.1 if frame < 220 else max(-0.14, 0.1 - (frame - 220) * 0.008)
      CS.odyssey_engine_torque_ts_nanos = frame * 10_000_000
      CS.odyssey_target_gear_ts_nanos = frame * 10_000_000
      _, sends = controller.update(control.as_reader(), CS, frame * 10_000_000)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if messages:
        parser.update([(frame * 10_000_000, messages)])
      if frame == 240:
        self.assertGreater(parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)

    self.assertFalse(controller.odyssey_coast_response.drag)
    self.assertTrue(controller.odyssey_gas_coast_release.active)
    self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
    self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
    self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], -0.14, delta=0.01)

  def test_settled_coast_overdeceleration_enters_mapped_gas_and_retains_brake_authority(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    safety = libsafety_py.libsafety
    config = CP.safetyConfigs[-1]
    self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
    safety.init_tests()
    safety.set_controls_allowed(True)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, 0.03, 0.0]
    control.actuators.accel = -0.18
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.aEgo = 21.0, -0.45
    state.canValid = True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

    def step():
      frame = controller.frame
      CS.odyssey_engine_torque_ts_nanos = frame * 10_000_000
      CS.odyssey_target_gear_ts_nanos = frame * 10_000_000
      CS.odyssey_computer_braking_ts_nanos = frame * 10_000_000
      _, sends = controller.update(control.as_reader(), CS, frame * 10_000_000)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if messages:
        for addr, dat, src in messages:
          self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, src, dat)))
        parser.update([(frame * 10_000_000, messages)])
      return parser.vl['ACC_CONTROL']

    for _ in range(60):
      signals = step()
      self.assertEqual(signals['GAS_COMMAND'], -30000)
    for frame in range(60, 130):
      signals = step()
      if frame >= 60:
        self.assertGreater(signals['GAS_COMMAND'], 0)
        self.assertEqual(signals['BRAKE_REQUEST'], 0)
        self.assertAlmostEqual(signals['ACCEL_COMMAND'], -0.18, delta=0.01)
        self.assertFalse(controller.odyssey_gas_bridge_active)
    control.actuators.accel = -0.6
    signals = step()
    self.assertEqual(signals['GAS_COMMAND'], -30000)
    self.assertEqual(signals['BRAKE_REQUEST'], 1)
    control.actuators.accel = 0.1
    step()
    signals = step()
    self.assertGreater(signals['GAS_COMMAND'], 0)
    self.assertEqual(signals['BRAKE_REQUEST'], 0)

  def test_coast_gas_entry_requires_settled_reliable_underresponse(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for blocked in ('settling', 'no_history', 'stale', 'pose', 'gas_pedal', 'brake_pedal', 'inactive',
                    'not_pid', 'low_speed', 'high_speed', 'gear', 'forecast', 'response',
                    'invalid_CAN', 'missing_gear', 'stale_gear', 'missing_VSA', 'stale_VSA', 'received_gas', 'received_brake'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP)
        controller.frame = 100
        controller.odyssey_pitch.x = 0.03
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.coast_start = 0
        controller.odyssey_coast_response.drag[7] = -0.45 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.03)
        control = structs.CarControl()
        control.enabled = control.longActive = True
        control.orientationNED = [0.0, 0.03, 0.0]
        control.actuators.accel = -0.18
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState()
        state.vEgo, state.aEgo = 21.0, -0.45
        state.canValid = True
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                             odyssey_engine_torque_ts_nanos=1_000_000_000, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=1_000_000_000,
                             odyssey_target_gear_ts_nanos=1_000_000_000)
        if blocked == 'settling':
          controller.odyssey_coast_response.coast_start = 98
        elif blocked == 'no_history':
          controller.odyssey_coast_response.drag.clear()
        elif blocked == 'stale':
          CS.odyssey_engine_torque_ts_nanos -= 80_000_000
        elif blocked == 'pose':
          control.orientationNED = []
        elif blocked == 'gas_pedal':
          state.gasPressed = True
        elif blocked == 'brake_pedal':
          state.brakePressed = True
        elif blocked == 'inactive':
          control.longActive = False
        elif blocked == 'not_pid':
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
        elif blocked == 'low_speed':
          state.vEgo = 4.0
        elif blocked == 'gear':
          CS.odyssey_target_gear = 8
          controller.odyssey_coast_response.drag[8] = controller.odyssey_coast_response.drag[7]
        elif blocked == 'forecast':
          controller.odyssey_coast_response.drag[7] = -0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.03)
        elif blocked == 'response':
          state.aEgo = -0.1
        elif blocked == 'high_speed':
          state.vEgo = 40.0
        elif blocked == 'invalid_CAN':
          state.canValid = False
        elif blocked == 'missing_gear':
          CS.odyssey_target_gear_ts_nanos = 0
        elif blocked == 'stale_gear':
          CS.odyssey_target_gear_ts_nanos -= 80_000_000
        elif blocked == 'missing_VSA':
          CS.odyssey_computer_braking_ts_nanos = 0
        elif blocked == 'stale_VSA':
          CS.odyssey_computer_braking_ts_nanos -= 80_000_000
        elif blocked == 'received_gas':
          CS.odyssey_car_gas = 1.0
          controller.odyssey_brake_selected = True
        elif blocked == 'received_brake':
          CS.odyssey_computer_braking = True
          controller.odyssey_brake_selected = True
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        _, sends = controller.update(control.as_reader(), CS, 1_000_000_000)
        parser.update([(1_000_000_000, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], blocked in ('low_speed', 'received_gas', 'received_brake'))

  def test_brake_entry_ignores_transient_passive_response_before_settling(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP)
    controller.odyssey_pitch.x = 0.04
    controller.odyssey_coast_response.gear = 7
    controller.odyssey_coast_response.drag[7] = -0.8 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.04)
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, 0.04, 0.0]
    control.actuators.accel = -0.4
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.canValid = 22.0, True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                         odyssey_shift_activity=0, odyssey_target_gear=7, odyssey_computer_braking=False)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    for frame in range(2 * ODYSSEY_RESPONSE_DELAY_FRAMES + 1):
      now = 2_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
      state.aEgo = -0.8 if 2 <= frame <= 6 or frame == 2 * ODYSSEY_RESPONSE_DELAY_FRAMES else -0.1
      _, sends = controller.update(control.as_reader(), CS, now)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if not messages:
        continue
      parser.update([(now, messages)])
      values = parser.vl['ACC_CONTROL']
      if frame < 2 * ODYSSEY_RESPONSE_DELAY_FRAMES:
        self.assertEqual(values['GAS_COMMAND'], -30000)
        self.assertEqual(values['BRAKE_REQUEST'], 1)
      else:
        self.assertGreaterEqual(values['GAS_COMMAND'], 0)
        self.assertEqual(values['BRAKE_REQUEST'], 0)

    for frame in range(51, 57):
      now = 2_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
      state.aEgo = -0.8 if frame >= 54 else -0.1
      control.actuators.accel = 0.1 if frame in (51, 56) else -0.4
      _, sends = controller.update(control.as_reader(), CS, now)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if not messages:
        continue
      parser.update([(now, messages)])
      values = parser.vl['ACC_CONTROL']
      self.assertEqual(values['BRAKE_REQUEST'], frame != 56)
      self.assertEqual(values['GAS_COMMAND'] == -30000, frame != 56)

  def test_passive_gas_floor_feedback_delay_continuity_and_brake_fallback(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for change in ('none', 'at_target', 'approaching_passive', 'excess', 'stronger'):
      with self.subTest(change=change):
        controller = CarController({Bus.pt: dbc}, CP)
        controller.frame = 100
        controller.odyssey_pitch.x = 0.04
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 0
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.drag[7] = -0.8 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.04)
        control = structs.CarControl()
        control.enabled = control.longActive = True
        control.orientationNED = [0.0, 0.04, 0.0]
        control.actuators.accel = -0.4
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState()
        state.vEgo, state.aEgo, state.canValid = 22.0, -0.8, True
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                             odyssey_shift_activity=0, odyssey_target_gear=7, odyssey_computer_braking=False)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)
        last_gas = 0
        first_positive = None
        for frame in range(100, 202):
          now = 2_000_000_000 + (frame - 100) * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = now - 10_000_000
          CS.odyssey_target_gear_ts_nanos = now - 10_000_000
          CS.odyssey_computer_braking_ts_nanos = now - 10_000_000
          if frame >= 160:
            if change == 'at_target':
              state.aEgo = -0.4
            elif change == 'approaching_passive':
              control.actuators.accel = -0.7
              state.aEgo = -0.7
            elif change == 'excess':
              state.aEgo = 0.0
            elif change == 'stronger':
              control.actuators.accel = -0.95
          output, sends = controller.update(control.as_reader(), CS, now)
          messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
          if not messages:
            continue
          for addr, dat, src in messages:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, src, dat)))
          parser.update([(now, messages)])
          values = parser.vl['ACC_CONTROL']
          gas, brake = values['GAS_COMMAND'], values['BRAKE_REQUEST']
          self.assertFalse(gas > -30000 and brake)
          self.assertAlmostEqual(output.accel, values['ACCEL_COMMAND'], delta=0.01)
          if frame >= 160 and change in ('excess', 'stronger'):
            self.assertEqual(gas, -30000)
            self.assertEqual(brake, 1)
          else:
            self.assertGreaterEqual(gas, 0)
            self.assertEqual(brake, 0)
            self.assertAlmostEqual(values['ACCEL_COMMAND'], control.actuators.accel, delta=0.01)
            self.assertFalse(controller.odyssey_gas_bridge_active)
            self.assertLessEqual(abs(gas - last_gas), ODYSSEY_RESPONSE_SLEW_COUNTS)
            self.assertLessEqual(gas, ODYSSEY_RESPONSE_MAX_COUNTS)
            if frame < 152:
              self.assertEqual(gas, 0)
            if gas > 0 and first_positive is None:
              first_positive = frame
          last_gas = gas
        self.assertEqual(first_positive, 100 + 2 * (ODYSSEY_RESPONSE_DELAY_FRAMES + 1))
        if change == 'none':
          self.assertGreater(last_gas, 0)

  def test_prior_coast_response_braking_cancels_gas_preactivation(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP)
    controller.frame = 100
    controller.odyssey_coast_response.gear = 7
    controller.odyssey_coast_response.coast_start = 98
    controller.odyssey_coast_response.drag[7] = 0.3
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, 0.0, 0.0]
    control.actuators.accel = -0.05
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.aEgo, state.canValid = 22.0, 0.3, True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=1_990_000_000, odyssey_shift_activity=0, odyssey_target_gear=7,
                         odyssey_target_gear_ts_nanos=1_990_000_000,
                         odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=1_990_000_000)
    _, sends = controller.update(control.as_reader(), CS, 2_000_000_000)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    parser.update([(2_000_000_000, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
    self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
    self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 1)
    self.assertFalse(controller.odyssey_gas_bridge_active)

  def test_settled_coast_deceleration_shortfall_enters_brakes_and_releases_on_positive_request(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP)
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, -0.03, 0.0]
    control.actuators.accel = -0.15
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.aEgo, state.canValid = 20.0, 0.3, True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                         odyssey_target_gear_ts_nanos=0,
                         odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)

    def step():
      now = 1_000_000_000 + controller.frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
      output, sends = controller.update(control.as_reader(), CS, now)
      parser.update([(now, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
      return output, parser.vl['ACC_CONTROL']

    for _ in range(50):
      _, signals = step()
      self.assertEqual(signals['BRAKE_REQUEST'], 0)
    for _ in range(50):
      output, signals = step()
    self.assertEqual(signals['BRAKE_REQUEST'], 1)
    self.assertEqual(signals['GAS_COMMAND'], -30000)
    self.assertAlmostEqual(signals['ACCEL_COMMAND'], odyssey_brake_accel(-0.15, controller.odyssey_pitch.x), delta=0.01)
    self.assertAlmostEqual(output.accel, signals['ACCEL_COMMAND'], delta=0.01)
    control.actuators.accel = 0.1
    state.aEgo = 0.0
    _, signals = step()
    self.assertEqual(signals['BRAKE_REQUEST'], 0)
    self.assertGreater(signals['GAS_COMMAND'], 0)

  def test_cold_coast_shortfall_enters_light_brake_and_retains_its_release_forecast(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    safety = libsafety_py.libsafety
    config = CP.safetyConfigs[-1]
    self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
    safety.init_tests()
    safety.set_controls_allowed(True)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP)
    control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0.0, -0.03, 0.0])
    control.actuators.accel = -0.20
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState(vEgo=20.0, aEgo=0.15, canValid=True)
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                         odyssey_target_gear_ts_nanos=0,
                         odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    for frame in range(76):
      now = 1_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
      if frame == 62:
        control.actuators.accel = -0.13
        state.aEgo = -0.5
        CS.odyssey_computer_braking = True
      elif frame == 74:
        control.actuators.accel = 0.1
      output, sends = controller.update(control.as_reader(), CS, now)
      if frame % 2:
        continue
      messages = [s for s in sends if s[0] == 0x1DF and s[2] == 1]
      for addr, dat, bus in messages:
        self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))
      parser.update([(now, messages)])
      signals = parser.vl['ACC_CONTROL']
      self.assertEqual(signals['BRAKE_REQUEST'], 60 <= frame < 74)
      if 60 <= frame < 74:
        self.assertEqual(signals['GAS_COMMAND'], -30000)
        self.assertAlmostEqual(signals['ACCEL_COMMAND'], odyssey_brake_accel(control.actuators.accel, controller.odyssey_pitch.x), delta=0.01)
        self.assertAlmostEqual(output.accel, signals['ACCEL_COMMAND'], delta=0.01)
        self.assertIsNotNone(controller.odyssey_coast_response.drag.get(7))
      if frame == 74:
        self.assertGreater(signals['GAS_COMMAND'], 0)

  def test_coast_brake_entry_requires_fresh_reliable_deceleration_shortfall(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for blocked in ('short_coast', 'no_history', 'stale', 'pose', 'gas_pedal', 'brake_pedal', 'inactive',
                    'not_pid', 'low_speed', 'high_speed', 'unknown_gear', 'forecast', 'response', 'positive', 'zero',
                    'invalid_can', 'stale_gear', 'missing_gear', 'future_gear', 'stale_brake', 'missing_brake',
                    'future_brake', 'received_gas', 'received_brake', 'previous_gas'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP)
        controller.frame = 100
        controller.odyssey_pitch.x = -0.03
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.coast_start = 0
        controller.odyssey_coast_response.drag[7] = 0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.03)
        control = structs.CarControl()
        control.enabled = control.longActive = True
        control.orientationNED = [0.0, -0.03, 0.0]
        control.actuators.accel = -0.15
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState()
        state.vEgo, state.aEgo, state.canValid = 20.0, 0.3, True
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_engine_torque_estimate=-100.0, odyssey_car_gas=0.0,
                             odyssey_engine_torque_ts_nanos=1_000_000_000, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_target_gear_ts_nanos=1_000_000_000,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=1_000_000_000)
        if blocked == 'short_coast':
          controller.odyssey_coast_response.coast_start = 98
        elif blocked == 'no_history':
          controller.odyssey_coast_response.drag.clear()
        elif blocked == 'stale':
          CS.odyssey_engine_torque_ts_nanos -= 80_000_000
        elif blocked == 'pose':
          control.orientationNED = []
        elif blocked == 'gas_pedal':
          state.gasPressed = True
        elif blocked == 'brake_pedal':
          state.brakePressed = True
        elif blocked == 'inactive':
          control.longActive = False
        elif blocked == 'not_pid':
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
        elif blocked == 'low_speed':
          state.vEgo = 4.0
        elif blocked == 'high_speed':
          state.vEgo = 36.0
        elif blocked == 'unknown_gear':
          CS.odyssey_target_gear = 8
        elif blocked == 'forecast':
          controller.odyssey_coast_response.drag[7] = -0.1 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.03)
        elif blocked == 'response':
          state.aEgo = -0.1
        elif blocked in ('positive', 'zero'):
          control.actuators.accel = 0.1 if blocked == 'positive' else 0.0
        elif blocked == 'invalid_can':
          state.canValid = False
        elif blocked in ('stale_gear', 'missing_gear', 'future_gear'):
          CS.odyssey_target_gear_ts_nanos = {
            'stale_gear': 920_000_000, 'missing_gear': 0, 'future_gear': 1_000_000_001,
          }[blocked]
        elif blocked in ('stale_brake', 'missing_brake', 'future_brake'):
          CS.odyssey_computer_braking_ts_nanos = {
            'stale_brake': 920_000_000, 'missing_brake': 0, 'future_brake': 1_000_000_001,
          }[blocked]
        elif blocked == 'received_gas':
          CS.odyssey_car_gas = 1.0
        elif blocked == 'received_brake':
          CS.odyssey_computer_braking = True
        elif blocked == 'previous_gas':
          controller.odyssey_gas_selected = True
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        _, sends = controller.update(control.as_reader(), CS, 1_000_000_000)
        parser.update([(1_000_000_000, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], blocked in ('low_speed', 'short_coast'))

  def test_brake_feedback_release_requires_easing_request_and_excess_deceleration(self):
    for blocked in (None, 'no_brake', 'stale', 'future', 'missing', 'gas', 'gear', 'ineligible',
                    'flat_request', 'response', 'strong', 'positive', 'gap', 'gear_change'):
      with self.subTest(blocked=blocked):
        response = OdysseyBrakeRelease()
        for frame in range(12):
          now = 1_000_000_000 + frame * 20_000_000
          request = -.28 + frame * .008
          if blocked == 'flat_request':
            request = -.2
          elif blocked == 'strong':
            request -= .2
          elif blocked == 'positive':
            request = .1
          timestamp = now - 5_000_000
          if blocked in ('stale', 'future', 'missing'):
            timestamp = {'stale': now - 60_000_000, 'future': now + 1, 'missing': 0}[blocked]
          if blocked == 'gap':
            now += frame * 40_000_000
            timestamp = now - 5_000_000
          gear = 0 if blocked == 'gear' else (6 + frame % 2 if blocked == 'gear_change' else 6)
          release = response.update(now, request, request - (.1 if blocked == 'response' else .3),
                                    blocked != 'no_brake', timestamp, 1. if blocked == 'gas' else 0.,
                                    gear, blocked != 'ineligible')
          self.assertEqual(release, blocked is None and frame >= 10)
        if blocked is None:
          self.assertEqual(odyssey_command_domains(request, 20., previous_brake=True, release_brake=release), (False, False))
          self.assertEqual(odyssey_command_domains(-.4, 20., previous_brake=True, release_brake=True), (False, True))
          self.assertEqual(odyssey_command_domains(request, 4., previous_brake=True, release_brake=True), (False, True))
          self.assertFalse(response.update(now + 20_000_000, request, request - .3, False,
                                           0, 0., 6, True))
          self.assertFalse(response.samples)

  def test_brake_release_uses_received_braking_with_flat_engine_torque(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    packer = CANPacker(dbc)
    for blocked in (None, 'no_brake', 'stale_brake', 'stale_gas', 'invalid_can', 'gas_pedal',
                    'brake_pedal', 'stopping', 'low_speed', 'strong', 'coast_shortfall',
                    'coast_sufficient', 'stale_gear', 'missing_gear', 'future_gear', 'invalid_pose'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP)
        CS = CarState(CP)
        parsers = CS.get_can_parsers(CP)
        CS.update(parsers)
        CS.out = structs.CarState(vEgo=7.5 if blocked == 'low_speed' else 20., canValid=blocked != 'invalid_can',
                                  gasPressed=blocked == 'gas_pedal', brakePressed=blocked == 'brake_pedal')
        control = structs.CarControl(longActive=True, enabled=True)
        forecast_case = blocked in ('coast_shortfall', 'coast_sufficient', 'stale_gear', 'missing_gear',
                                   'future_gear', 'invalid_pose')
        if forecast_case:
          control.orientationNED = [] if blocked == 'invalid_pose' else [0., -.04, 0.]
          controller.odyssey_pitch.x = -.04
          controller.odyssey_coast_response.gear = 6
          passive_accel = -.5 if blocked == 'coast_sufficient' else .3
          controller.odyssey_coast_response.drag[6] = passive_accel + ACCELERATION_DUE_TO_GRAVITY * math.sin(-.04)
        control.actuators.longControlState = (structs.CarControl.Actuators.LongControlState.stopping if blocked == 'stopping'
                                              else structs.CarControl.Actuators.LongControlState.pid)
        output_parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(24):
          now = 1_000_000_000 + frame * 10_000_000
          signals = [('VSA_STATUS', {'COMPUTER_BRAKING': blocked != 'no_brake'}),
                     ('GAS_PEDAL_2', {'ENGINE_TORQUE_ESTIMATE': -120, 'CAR_GAS': 0}),
                     ('GEARBOX_AUTO', {'TRANS_TARGET_GEAR': 6})]
          for name, values in signals:
            if frame > 0 and ((blocked == 'stale_brake' and name == 'VSA_STATUS') or
                              (blocked == 'stale_gas' and name == 'GAS_PEDAL_2') or
                              (blocked == 'stale_gear' and name == 'GEARBOX_AUTO')):
              continue
            parsers[Bus.pt].update([(now - 5_000_000, [packer.make_can_msg(name, 1, values)])])
          CS.update(parsers)
          if blocked in ('missing_gear', 'future_gear'):
            CS.odyssey_target_gear_ts_nanos = 0 if blocked == 'missing_gear' else now + 1
          control.actuators.accel = -.4 if blocked == 'strong' else -.28 + frame * .004
          CS.out.aEgo = control.actuators.accel - .3
          if frame == 0:
            controller.odyssey_brake_selected = True
          _, sends = controller.update(control.as_reader(), CS, now)
          output_parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        retain_brake = blocked not in (None, 'coast_sufficient', 'stale_gear', 'missing_gear', 'future_gear', 'invalid_pose')
        self.assertEqual(output_parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], retain_brake)
        self.assertEqual(output_parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        if blocked is None:
          self.assertAlmostEqual(output_parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], control.actuators.accel, delta=.01)
        if blocked == 'coast_shortfall':
          for request in (0., .1):
            control.actuators.accel = request
            for _ in range(2):
              now += 10_000_000
              _, sends = controller.update(control.as_reader(), CS, now)
              output_parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
            self.assertEqual(output_parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)

  def test_uphill_negative_gas_demand_delays_only_an_active_gas_release(self):
    gas_accel = odyssey_uphill_gas_accel_with_cap(-0.23, 0.03)[0]
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True, gas_accel=gas_accel), (True, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, gas_accel=gas_accel), (False, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True, gas_accel=-0.23), (False, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True,
                                              gas_accel=odyssey_uphill_gas_accel_with_cap(-0.23, 0.01)[0]), (False, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True,
                                              gas_accel=odyssey_uphill_gas_accel_with_cap(-0.23, 0.07)[0]), (True, False))

  def test_brake_grade_translation_tracks_net_acceleration(self):
    accel = -0.5
    pitch = 0.03
    grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * ODYSSEY_BRAKE_GRADE_GAIN
    self.assertEqual(ODYSSEY_BRAKE_GRADE_GAIN, 0.3)
    self.assertAlmostEqual(odyssey_brake_accel(accel, pitch), accel + grade)
    self.assertAlmostEqual(odyssey_brake_accel(accel, -pitch), accel - grade)
    self.assertEqual(odyssey_brake_accel(-0.1, 0.2), 0.0)
    self.assertEqual(odyssey_brake_accel(0.1, pitch), 0.1)

  def test_negative_gas_bridge_preserves_existing_domain_lifecycle(self):
    gas, active = odyssey_gas_command(-0.10, 91.0, True, False, False)
    self.assertEqual((gas, active), (ODYSSEY_GAS_BRIDGE_COMMAND, True))
    gas, active = odyssey_gas_command(-0.05, 136.0, True, True, active)
    self.assertEqual((gas, active), (ODYSSEY_GAS_BRIDGE_COMMAND, True))
    gas, active = odyssey_gas_command(0.0, 182.0, True, True, active)
    self.assertEqual((gas, active), (182.0, False))
    gas, active = odyssey_gas_command(-0.10, 91.0, True, True, False)
    self.assertEqual((gas, active), (91.0, False))
    gas, active = odyssey_gas_command(-0.15, 45.0, False, True, True)
    self.assertEqual((gas, active), (0.0, False))
    self.assertEqual(odyssey_gas_command(-0.05, 136.0, True, False, False, 0.5), (38.0, True))
    self.assertEqual(odyssey_gas_command(-0.05, 136.0, True, False, False, 1.0), (136.0, True))
    self.assertEqual(odyssey_gas_command(0.0, 182.0, True, True, True, 1.0), (182.0, False))
