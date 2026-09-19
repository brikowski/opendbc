import math
import unittest

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car.honda.carcontroller import (ODYSSEY_GAS_BRIDGE_COMMAND, ODYSSEY_UPHILL_GAS_ACCEL_MAX,
                                             odyssey_command_domains, odyssey_gas_command,
                                             odyssey_uphill_gas_accel)
from opendbc.car.honda.values import CAR, HondaFlags


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"


class TestOdysseyLongitudinal(unittest.TestCase):
  def test_uphill_gas_accel_preserves_raw_command_boundaries(self):
    accel = 0.3
    pitch = 0.05
    self.assertAlmostEqual(odyssey_uphill_gas_accel(accel, pitch),
                           accel + math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY)
    self.assertEqual(odyssey_uphill_gas_accel(accel, 0.0), accel)
    self.assertEqual(odyssey_uphill_gas_accel(accel, -pitch), accel)
    self.assertEqual(odyssey_uphill_gas_accel(0.0, pitch), 0.0)
    self.assertEqual(odyssey_uphill_gas_accel(-0.1, pitch), -0.1)
    self.assertEqual(odyssey_uphill_gas_accel(accel, -pitch, pitch), accel)
    self.assertEqual(odyssey_uphill_gas_accel(accel, math.pi / 2), ODYSSEY_UPHILL_GAS_ACCEL_MAX)
    self.assertEqual(odyssey_uphill_gas_accel(1.2, pitch), 1.2)

  def test_negative_gas_bridge_only_enters_from_road_speed_coast(self):
    self.assertEqual(odyssey_command_domains(-0.10, 20.0), (True, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True, bridge_active=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True), (True, False))
    self.assertEqual(odyssey_command_domains(-0.05, 20.0, previous_brake=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.05, 4.0), (False, True))

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
