# tests/test_controller.py

import unittest
import numpy as np
from src.controller import Controller
from src.exceptions import InputError
from src.particle_system import Particles


class TestController(unittest.TestCase):
    def setUp(self) -> None:
        self.x_1d = np.random.rand(100)
        self.x_2d = np.random.rand(100, 2)


    def test_is_valid(self):

        # vectorised
        result = Controller.is_valid(lambda y: np.zeros(y.shape), self.x_1d)
        self.assertEqual(result[1], False)
        self.assertEqual(result[2], True)

        result = Controller.is_valid(lambda y: np.zeros(y.shape), self.x_2d)
        self.assertEqual(result[1], False)
        self.assertEqual(result[2], True)

        result = Controller.is_valid(lambda y: np.zeros(y.T.shape), self.x_1d)
        self.assertEqual(result[1], False)
        self.assertEqual(result[2], True)

        result = Controller.is_valid(lambda y: np.zeros(y.T.shape), self.x_2d)
        self.assertEqual(result[1], False)
        self.assertEqual(result[2], False)

        # fail in vectorisation test
        result = Controller.is_valid(lambda y: 0 if y>0 else 1, self.x_1d)
        self.assertEqual(result[1], False)
        self.assertEqual(result[2], False)

        with self.assertRaises(InputError):
            Controller.is_valid(lambda y: 0 if y > 0 else 1, self.x_2d)

        # non-vectorised
        result = Controller.is_valid(lambda y: 0, self.x_1d)
        self.assertEqual(result[1], False)
        self.assertEqual(result[2], False)

        result = Controller.is_valid(lambda y: [0, 0], self.x_2d)
        self.assertEqual(result[1], False)
        self.assertEqual(result[2], False)

        # result = Controller.is_valid(lambda y: np.zeros(100), self.x_1d)
        # self.assertEqual(result[1], False)
        # self.assertEqual(result[2], True)

        with self.assertRaises(InputError):
            Controller.is_valid(lambda y: np.zeros(100), self.x_2d)

        # scalar
        result = Controller.is_valid(0, self.x_1d)
        self.assertEqual(result[0].tolist(), np.zeros(100).tolist())
        self.assertEqual(result[1], True)
        self.assertEqual(result[2], False)

        result = Controller.is_valid(0, self.x_2d)
        self.assertEqual(result[0].tolist(), np.zeros((100, 2)).tolist())
        self.assertEqual(result[1], True)
        self.assertEqual(result[2], False)

        # array-like
        result = Controller.is_valid([0, 1], self.x_2d)
        self.assertEqual(result[0].tolist(), np.full([100,2], [0, 1]).tolist())
        self.assertEqual(result[1], True)
        self.assertEqual(result[2], False)


    def test_apply(self):
        # scalar
        V = Controller.init(0, self.x_1d)
        self.assertEqual(V.is_fixed, True)
        self.assertEqual(V.is_vectorised, False)
        self.assertEqual(V.apply(self.x_1d).shape, (100, ))

        V = Controller.init(0, self.x_2d)
        self.assertEqual(V.is_fixed, True)
        self.assertEqual(V.is_vectorised, False)
        self.assertEqual(V.apply(self.x_2d).shape, (100, 2))

        # array-like
        V = Controller.init([0, 1], self.x_2d)
        self.assertEqual(V.is_fixed, True)
        self.assertEqual(V.is_vectorised, False)
        self.assertEqual(V.apply(self.x_2d).shape, (100, 2))

        # non-vectorised function
        def fn(y):
            return 1 if y>=0 else 0
        V = Controller.init(fn, self.x_1d)
        self.assertEqual(V.is_fixed, False)
        self.assertEqual(V.is_vectorised, False)
        self.assertEqual(V.apply(self.x_1d).shape, (100, ))

        def fn(y):
            ret = []
            ret.append(np.sign(y[0]))
            ret.append(np.sign(y[1]))
            return ret
        V = Controller.init(fn, self.x_2d)
        # V = Controller.init(fn, np.random.rand(100, 2))
        # V.apply(np.random.rand(100, 2))
        # np.stack([*map(fn, np.random.rand(10, 2))], axis=0)
        self.assertEqual(V.is_fixed, False)
        self.assertEqual(V.is_vectorised, False)
        self.assertEqual(V.apply(self.x_2d).shape, (100, 2))

        # vectorised function
        def fn(y):
            return -y
        V = Controller.init(fn, self.x_1d)
        self.assertEqual(V.is_fixed, False)
        self.assertEqual(V.is_vectorised, True)
        self.assertEqual(V.apply(self.x_1d).shape, (100, ))

        def fn(y):
            return -y
        V = Controller.init(fn, self.x_2d)
        self.assertEqual(V.is_fixed, False)
        self.assertEqual(V.is_vectorised, True)
        self.assertEqual(V.apply(self.x_2d).shape, (100, 2))



if __name__ == '__main__':
    unittest.main()
