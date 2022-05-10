# tests/test_particles.py
import unittest
import numpy as np

from src.particles import Particles


class TestParticles(unittest.TestCase):
    def setUp(self) -> None:
        self.pos_1d = np.random.rand(100)
        self.pos_2d = np.random.rand(100, 2)

    @unittest.skip
    def test_init(self):
        par = Particles(self.pos_1d, lower = 0.1, upper = 0.95)
        print(par)


if __name__ == '__main__':
    unittest.main()