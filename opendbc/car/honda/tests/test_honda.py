import math
import unittest

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car.honda.carcontroller import (ODYSSEY_BRAKE_GRADE_GAIN, ODYSSEY_GAS_BRIDGE_COMMAND, ODYSSEY_GRADE_GAIN,
                                             ODYSSEY_GRADE_RAMP_ACCEL, ODYSSEY_UPHILL_GAS_ACCEL_MAX, odyssey_brake_accel,
                                             odyssey_command_domains, odyssey_gas_command, odyssey_uphill_gas_accel)
from opendbc.car.honda.values import CAR, HondaFlags


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"


class TestOdysseyLongitudinal(unittest.TestCase):
  def test_uphill_gas_load_is_zero_at_split_and_bounded_at_high_request(self):
    pitch = 0.03
    self.assertEqual(odyssey_uphill_gas_accel(0.0, pitch), 0.0)
    self.assertEqual(odyssey_uphill_gas_accel(-0.01, pitch), -0.01)
    self.assertEqual(odyssey_uphill_gas_accel(0.3, 0.0), 0.3)
    self.assertEqual(odyssey_uphill_gas_accel(0.3, -pitch), 0.3)
    self.assertLess(odyssey_uphill_gas_accel(0.01, pitch) - 0.01, 0.002)
    expected = ODYSSEY_GRADE_RAMP_ACCEL + math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * 0.7
    self.assertEqual(ODYSSEY_GRADE_GAIN, 0.7)
    self.assertAlmostEqual(odyssey_uphill_gas_accel(ODYSSEY_GRADE_RAMP_ACCEL, pitch), expected)
    self.assertEqual(odyssey_uphill_gas_accel(1.2, pitch), 1.2)
    self.assertLessEqual(odyssey_uphill_gas_accel(0.83, 0.072), ODYSSEY_UPHILL_GAS_ACCEL_MAX)

  def test_negative_gas_bridge_only_enters_from_road_speed_coast(self):
    self.assertEqual(odyssey_command_domains(-0.10, 20.0), (True, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True, bridge_active=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True), (True, False))
    self.assertEqual(odyssey_command_domains(-0.05, 20.0, previous_brake=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.05, 4.0), (False, True))

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
