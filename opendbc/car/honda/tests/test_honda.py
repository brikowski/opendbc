import unittest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, HondaFlags


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"


class TestHondaParams(unittest.TestCase):
  def test_odyssey_mmr_lateral_range(self):
    for alpha_long in (False, True):
      CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], alpha_long, False, False)
      assert list(CP.lateralParams.torqueBP) == [0, 2560]
      assert list(CP.lateralParams.torqueV) == [0, 2560]
      assert CP.openpilotLongitudinalControl == alpha_long
