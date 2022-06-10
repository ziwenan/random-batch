# tests/test_interacting_particle_system.py

import time
import unittest

import numpy as np
import pandas as pd

from src.interacting_particle_system import IPS


class TestIPS(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.n = 100
        self.x_1d = np.random.rand(self.n)
        self.x_2d = np.random.rand(self.n, 2)
        self.x_3d = np.random.rand(self.n, 3)
        self.ips_1d_first = IPS(self.x_1d, order='first', boundary_condition='absorbing', lower=0.1)
        self.ips_2d_first = IPS(self.x_2d, order='first', boundary_condition='absorbing', lower=[0.1] * 2)
        self.ips_3d_first = IPS(self.x_3d, order='first', boundary_condition='absorbing', lower=[0.1] * 3)
        self.ips_2d_second = IPS(self.x_2d, order='second', boundary_condition='absorbing', lower=[0.1] * 2)
        # print(f'ips_1d: {self.ips_1d}')
        # print(f'ips_1d: {self.ips_2d_first}')
        # print(f'ips_1d: {self.ips_3d}')
        # self.ips_1d = IPS(self.x_1d, 0, lambda y: -40 * y * np.where(abs(y)<= 1, True, False))
        # self.ips_2d_first = IPS(self.x_2d, 0, lambda y: -40 * y * np.where(abs(y) <= 1, True, False))

    def test_add_property(self):
        # 2D
        print('\n**test_add_property: 2D**')
        start = time.perf_counter()
        mass = np.ones(self.n)
        mass[50:] = 5
        self.ips_2d_first.add_property(mass, 'mass')
        self.ips_2d_first.add_property(0.01, 'radius')
        vel = np.zeros_like(self.ips_2d_first.velocity)
        vel[0] = vel[50] = [1, 1]
        self.ips_2d_first.velocity = vel
        gpe = 9.8 * mass * self.x_2d[:, 0]
        self.ips_2d_first.add_property(0, 'height', lambda position: np.abs(position[:, 0]))

        def compute_energy(_d, mass, speed):
            return 0.5 * np.sum(mass * (speed * speed))

        self.ips_2d_first.add_property(0, 'kinetic_energy', compute_energy)
        self.ips_2d_first.add_property(0, 'gravitational_potential_energy', lambda mass, height: 9.8 * mass * height)
        print(self.ips_2d_first)
        np.testing.assert_allclose(self.ips_2d_first.get_property('mass'), mass)
        self.assertEqual(self.ips_2d_first.get_property('radius'), 0.01)
        self.assertAlmostEqual(self.ips_2d_first.get_property('kinetic_energy'), 6)
        np.testing.assert_allclose(self.ips_2d_first.get_property('gravitational_potential_energy'), gpe)
        end = time.perf_counter()
        print(f'Time Elapsed for test_add_property: {end - start:.5f} sec')

    def test_shuffle(self):
        # print('\n**test_shuffle**')
        mass = np.ones(self.n)
        mass[50:] = 5
        np.random.seed(self.n)
        index = np.arange(self.n)
        np.random.shuffle(index)
        mass_shuffled = mass[index]
        self.ips_2d_first.add_property(mass, 'mass')
        self.ips_2d_first.shuffle(seed=self.n)
        np.testing.assert_allclose(self.ips_2d_first.get_property('mass'), mass_shuffled)

    def test_add_controller(self):
        # 2D
        print('\n**test_add_controller: 2D**')
        self.ips_2d_second.add_controller(9.8, type='external', name=None)
        self.ips_2d_second.add_controller(lambda y: -40 * y * np.where(abs(y) <= 1, True, False), type='interacting',
                                          name='attraction 1')
        self.ips_2d_second.add_controller(lambda x, alpha=-40: alpha * x * np.where(abs(x) <= 1, True, False),
                                          type='interacting', name='attraction 2')
        self.ips_2d_second.add_property(-40, 'alpha')
        self.ips_2d_second.add_controller(lambda x, alpha: alpha * x * np.where(abs(x) <= 1, True, False),
                                          type='interacting', name='attraction 3')
        print(self.ips_2d_second, '\n')

    def test_evolve(self):
        # 1d
        # ips_generator = self.ips_1d.evolve(1e-3, 3, sigma = 0.005)
        # res = np.stack([*ips_generator], axis=0)
        # self.assertEqual(res.shape, (3001, 100))

        # 2D
        print('\n**test_evolve: 2D, first order**')
        self.ips_2d_first.add_property(-10, 'alpha')
        self.ips_2d_first.add_controller(0, type='external', name='V')
        self.ips_2d_first.add_property(0, 'phi', fn_compute=lambda position: abs(position) <= 1)
        self.ips_2d_first.add_controller(lambda x, alpha, phi: alpha * phi * (-x), type='interacting', name='K')
        print(self.ips_2d_first)
        start = time.perf_counter()
        res = self.ips_2d_first.evolve(dt=1e-3, T=1, sigma=0.005)
        print(self.ips_2d_first)
        print(res)
        end = time.perf_counter()
        print(f'Time Elapsed: {end - start:.5f} sec')


        # start = time.perf_counter()
        # self.ips_2d_first.add_property(10, 'alpha')
        # self.ips_2d_first.add_controller(lambda alpha: alpha * phi(y) * (-y))
        # print(self.ips_2d_first)
        #
        # end = time.perf_counter()
        # print(f'Time Elapsed for test_add_property: {end - start:.5f} sec')
        # ips_generator = self.ips_2d_first.evolve(dt=1e-3, T=5, sigma=0.005)
        # res = np.stack([*ips_generator], axis=0)
        # self.assertEqual(res.shape, (3001, 100, 2))


if __name__ == '__main__':
    unittest.main()
