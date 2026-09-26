import math
import unittest

from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, gen_empty_fingerprint
from opendbc.car.honda.carcontroller import (ODYSSEY_BRAKE_GRADE_GAIN, ODYSSEY_GAS_BRIDGE_COMMAND, ODYSSEY_GRADE_GAIN,
                                             ODYSSEY_GRADE_RAMP_ACCEL, ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS,
                                             ODYSSEY_NEGATIVE_GRADE_ACCEL_MAX,
                                             ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL, ODYSSEY_RESPONSE_COUNTS_PER_ACCEL,
                                             ODYSSEY_RESPONSE_DELAY_FRAMES, ODYSSEY_RESPONSE_MAX_COUNTS,
                                             ODYSSEY_ROAD_BRAKE_ENTRY, ODYSSEY_UPHILL_GAS_ACCEL_MAX, OdysseyBrakeRelease,
                                             OdysseyGasResponse, odyssey_brake_accel,
                                             odyssey_command_domains, odyssey_gas_command, odyssey_low_speed_gas_command,
                                             odyssey_uphill_gas_accel, odyssey_uphill_gas_accel_with_cap)
from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import DBC


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"


class TestOdysseyLongitudinal(unittest.TestCase):
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
    self.assertEqual(odyssey_uphill_gas_accel(0.0, pitch), 0.0)
    self.assertGreater(odyssey_uphill_gas_accel(-0.10, pitch), -0.10)
    self.assertAlmostEqual(odyssey_uphill_gas_accel(-0.20, pitch),
                           -0.20 + ODYSSEY_NEGATIVE_GRADE_ACCEL_MAX)
    self.assertEqual(odyssey_uphill_gas_accel(ODYSSEY_ROAD_BRAKE_ENTRY, pitch), ODYSSEY_ROAD_BRAKE_ENTRY)
    self.assertEqual(odyssey_uphill_gas_accel(0.3, 0.0), 0.3)
    downhill_grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * 0.6
    self.assertAlmostEqual(odyssey_uphill_gas_accel(0.3, -pitch), 0.3 - downhill_grade)
    self.assertLess(odyssey_uphill_gas_accel(0.01, pitch) - 0.01, 0.002)
    expected = ODYSSEY_GRADE_RAMP_ACCEL + math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * 0.6
    self.assertEqual(ODYSSEY_GRADE_GAIN, 0.6)
    self.assertAlmostEqual(odyssey_uphill_gas_accel(ODYSSEY_GRADE_RAMP_ACCEL, pitch), expected)
    self.assertEqual(odyssey_uphill_gas_accel(1.2, pitch), 1.2)
    self.assertAlmostEqual(odyssey_uphill_gas_accel(1.2, -pitch), 1.2 - downhill_grade)
    self.assertLessEqual(odyssey_uphill_gas_accel(0.83, 0.072), ODYSSEY_UPHILL_GAS_ACCEL_MAX)

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

  def test_gas_response_waits_for_vehicle_delay_and_corrects_both_directions(self):
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

  def test_gas_response_rejects_falling_grade_and_stale_lead_request(self):
    response = OdysseyGasResponse()
    for frame in range(ODYSSEY_RESPONSE_DELAY_FRAMES * 2):
      assert response.update(0.0, -0.3, 0.06 - frame * 0.0005, 20.0, 1, True) == 0.0
    response.reset()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      assert response.update(0.3, -0.3, 0.06, 20.0, 1, True) == 0.0
    assert response.update(-0.3, -0.3, 0.06, 20.0, 1, True) == 0.0

  def test_negative_gas_bridge_only_enters_from_road_speed_coast(self):
    self.assertEqual(odyssey_command_domains(-0.10, 20.0), (True, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True, bridge_active=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True), (True, False))
    self.assertEqual(odyssey_command_domains(-0.05, 20.0, previous_brake=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.05, 4.0), (False, True))

  def test_torque_qualified_brake_release_retains_strong_and_low_speed_authority(self):
    response = OdysseyBrakeRelease()
    for frame in range(11):
      now = 1_000_000_000 + frame * 20_000_000
      request = -.28 + frame * .008
      release = response.update(now, request, request - .3, -100. - frame * 5.,
                                now - 5_000_000, 0., 6, True)
      if frame < 10:
        self.assertFalse(release)
    self.assertTrue(release)
    self.assertEqual(odyssey_command_domains(request, 20., previous_brake=True, release_brake=release), (False, False))
    self.assertEqual(odyssey_command_domains(-.4, 20., previous_brake=True, release_brake=True), (False, True))
    self.assertEqual(odyssey_command_domains(request, 4., previous_brake=True, release_brake=True), (False, True))
    self.assertFalse(response.update(now + 20_000_000, request, request - .3, -160., now - 80_000_000, 0., 6, True))
    self.assertFalse(response.samples)
    for frame in range(11):
      now += 20_000_000
      release = response.update(now, request, request - .3, -200. - frame * 5., now - 5_000_000,
                                0., 7, True)
    self.assertFalse(release)  # no rising request, even with falling torque
    self.assertFalse(response.update(now + 20_000_000, request, request - .3, -260., now + 30_000_000, 0., 7, True))
    self.assertFalse(response.samples)

  def test_brake_release_rejects_flat_torque_despite_rising_request_and_overdeceleration(self):
    response = OdysseyBrakeRelease()
    for frame in range(12):
      now = 1_000_000_000 + frame * 20_000_000
      request = -.28 + frame * .008
      self.assertFalse(response.update(now, request, request - .3, -120., now - 5_000_000, 0., 6, True))
    self.assertFalse(response.update(now + 20_000_000, request + .008, request - .3, -175.,
                                     now + 15_000_000, 1., 6, True))
    self.assertFalse(response.samples)

  def test_uphill_negative_gas_demand_delays_only_an_active_gas_release(self):
    gas_accel = odyssey_uphill_gas_accel(-0.23, 0.03)
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True, gas_accel=gas_accel), (True, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, gas_accel=gas_accel), (False, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True, gas_accel=-0.23), (False, False))

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
