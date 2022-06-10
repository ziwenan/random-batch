# tests/test_controller.py

import unittest
from timeit import timeit

import numpy as np

from src.controller import Controller


class TestController(unittest.TestCase):
    def setUp(self) -> None:
        self.num_particles = 1000
        self.x_1d = np.random.rand(self.num_particles)
        self.x_2d = np.random.rand(self.num_particles, 2)
        self.res_1d = np.where(np.abs(self.x_1d) < 0.5, 1, 0)
        self.res_2d = np.where(np.abs(self.x_2d) < 0.5, 1, 0)

    def timer(self, fn):
        from time import time
        def inner(*args, **kwargs):
            t0 = time()
            res = fn(*args, **kwargs)
            t1 = time()
            print(f'{fn.__name__} time elapse: {1000 * (t1 - t0):2.4f} ms')
            return res

        return inner

    def test_check_validity(self):
        # scalar
        what, func, value, vectorised, numba_ = Controller.check_validity(0, self.x_1d)
        self.assertEqual(what, 'value')
        self.assertEqual(func, None)
        np.testing.assert_allclose(value, np.zeros(self.num_particles))
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

        what, func, value, vectorised, numba_ = Controller.check_validity(0, self.x_2d)
        self.assertEqual(what, 'value')
        self.assertEqual(func, None)
        np.testing.assert_allclose(value, np.zeros([self.num_particles, 2]))
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

        # array-like
        what, func, value, vectorised, numba_ = Controller.check_validity([0, 1], self.x_2d)
        self.assertEqual(what, 'value')
        self.assertEqual(func, None)
        np.testing.assert_allclose(value, np.full([self.num_particles, 2], [0, 1]))
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

        self.assertRaises(AssertionError, Controller.check_validity, [0, 1, 2], self.x_2d)

        # vectorised
        what, func, value, vectorised, numba_ = Controller.check_validity(lambda y: np.zeros((self.num_particles,)),
                                                                         self.x_1d)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_1d), np.zeros_like(self.x_1d))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, True)
        self.assertEqual(numba_, False)

        what, func, value, vectorised, numba_ = Controller.check_validity(
            func_or_value=lambda y: np.zeros((self.num_particles, 2)), x=np.ones([self.num_particles, 2]))
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_2d), np.zeros_like(self.x_2d))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, True)
        self.assertEqual(numba_, False)

        what, func, value, vectorised, numba_ = Controller.check_validity(lambda y: np.zeros(y.T.shape), self.x_1d)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_1d), np.zeros_like(self.x_1d))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, True)
        self.assertEqual(numba_, False)

        # non-vectorised, numba
        what, func, value, vectorised, numba_ = Controller.check_validity(lambda y: np.zeros(y.T.shape), self.x_2d)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_2d), np.zeros_like(self.x_2d))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, True)

        what, func, value, vectorised, numba_ = Controller.check_validity(lambda y: 0 if y > 0 else 1, self.x_1d)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_1d), np.zeros_like(self.x_1d))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, True)

        what, func, value, vectorised, numba_ = Controller.check_validity(lambda y: [0,0] if y[0]>0 else [1,1], self.x_2d)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_2d), np.zeros_like(self.x_2d))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, True)

        what, func, value, vectorised, numba_ = Controller.check_validity(lambda x, threshold: 0 if x > threshold else 1, self.x_1d, threshold=0, suppress_warnings=True)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_1d, threshold=0), np.zeros_like(self.x_1d))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

        self.assertRaises(ValueError, Controller.check_validity, lambda y: 0 if y > 0 else 1, self.x_2d)

        # non-vectorised, !numba
        what, func, value, vectorised, numba_ = Controller.check_validity(lambda y: y+1, self.x_1d, use_numba=False, force_nonvectorise=True)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_1d), self.x_1d + 1)
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

        what, func, value, vectorised, numba_ = Controller.check_validity(lambda y: [0, np.mean(y, -1)], self.x_2d, suppress_warnings=True)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_2d), np.column_stack((np.zeros(self.num_particles),np.mean(self.x_2d, -1))))
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

        what, func, value, vectorised, numba_ = Controller.check_validity(lambda x, a: x + a, self.x_1d, a=1, use_numba=False, force_nonvectorise=True)
        self.assertEqual(what, 'function')
        np.testing.assert_allclose(func(self.x_1d, a=1), self.x_1d+1)
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

        what, func, value, vectorised, numba_ = Controller.check_validity(lambda x, a: [0, np.mean(x, a)], self.x_2d, a=-1, suppress_warnings=True)
        self.assertEqual(what, 'function')
        self.assertEqual(value, None)
        self.assertEqual(vectorised, False)
        self.assertEqual(numba_, False)

    # @unittest.skip
    def test_apply(self):
        # scalar
        C = Controller.init(0, self.x_1d)
        np.testing.assert_allclose(C.apply_to(self.x_1d), np.zeros(self.num_particles))

        C = Controller.init(0, self.x_2d)
        np.testing.assert_allclose(C.apply_to(self.x_2d), np.zeros((self.num_particles, 2)))

        # array-like
        C = Controller.init([0, 1], self.x_2d)
        np.testing.assert_allclose(C.apply_to(self.x_2d), np.full((self.num_particles, 2), [0, 1]))
        t_scalar = timeit('C.apply_to(self.x_2d)', number=1, globals=locals())

        # non-vectorised function
        def fn(x):
            return 1 if np.abs(x) < 0.5 else 0

        C = Controller.init(fn, self.x_1d)
        np.testing.assert_allclose(C.apply_to(self.x_1d), self.res_1d)

        def fn(x):
            ret = []
            for i in x:
                ret.append(1 if np.abs(i) < 0.5 else 0)
            return ret

        C = Controller.init(fn, self.x_2d, use_numba=False)
        self.assertEqual(C.is_vectorised, False)
        self.assertEqual(C.use_numba, False)
        res = C.apply_to(self.x_2d)
        np.testing.assert_allclose(res, self.res_2d)
        np.testing.assert_allclose(res, self.res_2d)
        t_nonvec = timeit('C.apply_to(self.x_2d)', number=1, globals=locals())

        C = Controller.init(fn, self.x_2d)
        self.assertEqual(C.is_vectorised, False)
        self.assertEqual(C.use_numba, True)
        res = C.apply_to(self.x_2d)
        np.testing.assert_allclose(res, self.res_2d)
        np.testing.assert_allclose(res, self.res_2d)
        t_numba = timeit('C.apply_to(self.x_2d)', number=1, globals=locals())

        # vectorised function
        def fn(y):
            return np.where(np.abs(y) < 0.5, 1, 0)

        C = Controller.init(fn, self.x_1d)
        self.assertEqual(C.is_vectorised, True)
        self.assertEqual(C.use_numba, False)
        np.testing.assert_allclose(C.apply_to(self.x_1d), self.res_1d)

        def fn(y):
            return np.where(np.abs(y) < 0.5, 1, 0)

        C = Controller.init(fn, self.x_2d)
        self.assertEqual(C.is_vectorised, True)
        self.assertEqual(C.use_numba, False)
        res = C.apply_to(self.x_2d)
        np.testing.assert_allclose(res, self.res_2d)
        np.testing.assert_allclose(res, self.res_2d)
        t_vec = timeit('C.apply_to(self.x_2d)', number=1, globals=locals())

        print([t_scalar, t_nonvec, t_numba, t_vec])
        import matplotlib.pyplot as plt
        plt.style.use('seaborn')
        fig, ax = plt.subplots()
        ax.plot(np.array([t_scalar, t_nonvec, t_numba, t_vec]) / t_scalar, marker='D', linewidth=4,
                markerfacecolor='darkblue', markersize=10)
        ax.set_title(f'Performance of Controller.apply_to to {self.num_particles} particles')
        ax.set_ylabel('Relative Computation Time')
        ax.set_xticks([0, 1, 2, 3], labels=['scalar', 'non-vect', 'numba_vect', 'vect'])
        ax.set_yscale('log', base=2)
        # plt.pause(10)
        plt.waitforbuttonpress()
        fig.show()


if __name__ == '__main__':
    unittest.main()
