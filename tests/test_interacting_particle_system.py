# tests/test_interacting_particle_system.py

import unittest
import numpy as np
from src.interacting_particle_system import IPS


class TestIPS(unittest.TestCase):
    def setUp(self) -> None:
        self.x_1d = np.random.rand(100)
        self.x_2d = np.random.rand(100, 2)
        self.ips_1d = IPS(self.x_1d, 0, lambda y: -40 * y * np.where(abs(y)<= 1, True, False))
        self.ips_2d = IPS(self.x_2d, 0, lambda y: -40 * y * np.where(abs(y) <= 1, True, False))

    def test_evolve(self):
        # 1d
        ips_generator = self.ips_1d.evolve(1e-3, 3, sigma = 0.005)
        res = np.stack([*ips_generator], axis=0)
        self.assertEqual(res.shape, (3001, 100))

        # 2d
        ips_generator = self.ips_2d.evolve(1e-3, 3, sigma = 0.005)
        res = np.stack([*ips_generator], axis=0)
        self.assertEqual(res.shape, (3001, 100, 2))



if __name__ == '__main__':
    unittest.main()
