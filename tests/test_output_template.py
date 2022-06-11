# tests/test_interacting_particle_system.py

import dataclasses
import unittest

import numpy as np

from src.output_template import OutputTemplate


class TestOutputTemplate(unittest.TestCase):
    # @unittest.skip
    def test_init(self):
        # empty
        # ot = OutputTemplate()
        # self.assertIsNone(ot.position)
        # self.assertIsNone(ot.index)

        # init with num, d
        ot = OutputTemplate(num=5, d=2)
        self.assertEqual(ot.position.shape, (0, 5, 2))
        self.assertEqual(ot.index.shape, (0, 5))

    def test_new_attr(self):
        # add attr
        def my_init(self):
            OutputTemplate.__post_init__(self)
            if self.velocity is None and self.num is not None and self.d is not None:
                if self.d == 1:
                    self.velocity = np.empty((0, self.num,))
                else:
                    self.velocity = np.empty((0, self.num, self.d))
            self.attr_names.append('velocity')

        Output = dataclasses.make_dataclass('Output',
                                            [('velocity', np.ndarray, dataclasses.field(default=None))],
                                            bases=(OutputTemplate,),
                                            namespace={'__post_init__': my_init})
        ot = Output(num=5, d=2)
        self.assertEqual(ot.position.shape, (0, 5, 2))
        self.assertEqual(ot.velocity.shape, (0, 5, 2))
        self.assertEqual(ot.index.shape, (0, 5))

        # append
        def my_init(self):
            OutputTemplate.__post_init__(self)
            if self.velocity is None and self.num is not None and self.d is not None:
                if self.d == 1:
                    self.velocity = np.empty((0, self.num,))
                else:
                    self.velocity = np.empty((0, self.num, self.d))
            if self.t is None and self.num is not None and self.d is not None:
                self.t = []
            self.attr_names.append('velocity')
            self.attr_names.append('t')

        Output = dataclasses.make_dataclass('Output',
                                            [('velocity', np.ndarray, dataclasses.field(default=None)),
                                             ('t', float, dataclasses.field(default=None))],
                                            bases=(OutputTemplate,),
                                            namespace={'__post_init__': my_init})
        ot = Output(num=5, d=2)
        # print(ot)
        to_append = np.empty((5, 2))
        ot.append(dict(position=to_append, velocity=to_append, index=np.empty((5,)), t=1))
        ot.append(dict(position=to_append, velocity=to_append, index=np.empty((5,)), t=2))
        # print(ot)
        self.assertEqual(ot.position.shape, (2, 5, 2))
        self.assertEqual(ot.velocity.shape, (2, 5, 2))
        self.assertEqual(ot.index.shape, (2, 5))
        self.assertListEqual(ot.t, [1, 2])

    def test_render(self):
        import time
        from src.interacting_particle_system import IPS
        n = 100
        ips = IPS(np.random.rand(n, 2), order='first')
        ips.add_property(-10, 'alpha')
        ips.velocity = np.ones_like(ips.velocity)
        ips.add_property(0, 'average_speed', fn_compute=lambda speed: np.mean(speed))
        # print(f'as0: {ips.get_property("average_speed"):.4f}')
        ips.add_controller(0, type='external', name='V')
        ips.add_controller(lambda x, alpha: alpha * np.where(abs(x) <= 0.3, 1, 0) * (-x), type='interacting', name='K')
        print(ips)
        start = time.perf_counter()
        res = ips.evolve(dt=1e-3, T=3, sigma=0.005, thinning=100, options=['velocity'],
                         early_stopping=dict(property='average_speed', value=0.0001))
        end = time.perf_counter()
        print(f'Time Elapsed: {end - start:.5f} sec')
        print(f'as1: {ips.get_property("average_speed"):.4f}')
        print(ips)
        self.assertEqual(isinstance(res, OutputTemplate), True)
        res.render()


if __name__ == '__main__':
    unittest.main()
