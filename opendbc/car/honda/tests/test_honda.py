import math
import unittest

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car.honda.carcontroller import (ODYSSEY_BRAKE_GRADE_GAIN, ODYSSEY_GAS_BRIDGE_COMMAND, ODYSSEY_GRADE_GAIN,
                                             ODYSSEY_GRADE_RAMP_ACCEL, ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS,
                                             ODYSSEY_NEGATIVE_GRADE_ACCEL_MAX,
                                             ODYSSEY_ROAD_BRAKE_ENTRY, ODYSSEY_UPHILL_GAS_ACCEL_MAX, odyssey_brake_accel,
                                             odyssey_command_domains, odyssey_gas_command, odyssey_low_speed_gas_command,
                                             odyssey_steep_nearzero_grade_accel, odyssey_uphill_gas_accel)
from opendbc.car.honda.values import CAR, HondaFlags


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"


class TestOdysseyLongitudinal(unittest.TestCase):
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

  def test_steep_uphill_near_zero_gas_load_is_local_and_smooth(self):
    pitch = 0.06
    self.assertAlmostEqual(odyssey_steep_nearzero_grade_accel(0.0, pitch, 20.0), 0.15)
    self.assertGreater(odyssey_steep_nearzero_grade_accel(-0.03, pitch, 20.0), 0.10)
    self.assertAlmostEqual(odyssey_steep_nearzero_grade_accel(0.0, pitch, 4.0), 0.0)
    self.assertGreater(odyssey_steep_nearzero_grade_accel(0.0, pitch, 6.5), 0.0)
    self.assertAlmostEqual(odyssey_steep_nearzero_grade_accel(0.0, 0.03, 20.0), 0.0)
    self.assertAlmostEqual(odyssey_steep_nearzero_grade_accel(0.0, -pitch, 20.0), 0.0)
    for request in (-0.10, 0.20):
      self.assertAlmostEqual(odyssey_steep_nearzero_grade_accel(request, pitch, 20.0), 0.0)
    for p in (0.03, 0.045, 0.06, 0.09):
      values = [odyssey_uphill_gas_accel(i * 0.001, p) +
                odyssey_steep_nearzero_grade_accel(i * 0.001, p, 20.0) for i in range(-300, 301)]
      self.assertTrue(all(left <= right + 1e-9 for left, right in zip(values, values[1:], strict=False)))
      for threshold in (-0.10, -0.05, 0.0, 0.20):
        self.assertLess(abs(odyssey_steep_nearzero_grade_accel(threshold - 1e-5, p, 20.0) -
                            odyssey_steep_nearzero_grade_accel(threshold + 1e-5, p, 20.0)), 1e-3)

  def test_negative_gas_bridge_only_enters_from_road_speed_coast(self):
    self.assertEqual(odyssey_command_domains(-0.10, 20.0), (True, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True, bridge_active=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True), (True, False))
    self.assertEqual(odyssey_command_domains(-0.05, 20.0, previous_brake=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.05, 4.0), (False, True))

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
