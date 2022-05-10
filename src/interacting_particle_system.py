# src/interacting_particle_system.py
import numpy as np
from src.particles import Particles
from src.controller import Controller


class IPS:
    '''
    IPS (interacting particle system) class holds the information of a set of particles for a given period of time.
    '''
    particles: Particles
    external_force: Controller
    interacting_force: Controller

    # initial state of the particle system
    def __init__(self, x, V, K, lower=None, upper=None):
        '''
        Initialise an interacting particle system.

        operators:
            x: An array-like object of dimension num by dim. If the data is one-dimensional, x is flattened.
        '''
        self.particles = Particles(x, lower=lower, upper=upper)
        self.external_force = Controller.init(V, x)
        self.interacting_force = Controller.init(K, self.particles.pairwise_diff())

    # update
    def update(self, dt, sigma):
        self.particles.shuffle()
        self.particles.velocity = -self.external_force.apply(self.particles.position) + self.interacting_force.apply(
            self.particles.pairwise_diff())
        if self.particles.d == 1:
            self.particles.position += self.particles.velocity * dt + sigma * np.sqrt(dt) * np.random.randn(self.particles.num)
        else:
            self.particles.position += self.particles.velocity * dt + sigma * np.sqrt(dt) * np.random.randn(self.particles.num, self.particles.d)
        self.particles.apply_boundary_condition()

    # evolve
    def evolve(self, dt, T, sigma=0, method='random_batch', velocity = False, id = False, thinning: int = None):
        t = 0
        count = 0
        while t < T:
            if method == 'random_batch':
                self.update(dt, sigma)
                t += dt
                count += 1
                
                if thinning is None or count % thinning == 0:
                    if not velocity and not id:
                        yield self.particles.position
                    else:
                        ret = {'position': self.particles.position}
                        if velocity:
                            ret['velocity'] = self.particles.velocity
                        if id:
                            ret['id'] = self.particles.id
                        yield ret

    def __str__(self):
        return f'IPS: {self.particles.__str__()}, external force: {self.external_force.__str__()}, interacting force: {self.interacting_force.__str__()}'


if __name__ == '__main__':
    x = np.random.rand(100) * 10  # (random) different opinions at t = 0
    ips = IPS(x, 0, lambda y: -40 * y * np.where(abs(y)<= 1, True, False))
    print(ips)
    # ips.particles.id

    # ips.update(1e-3, 0)

    ips_generator = ips.evolve(1e-3, 3)
    # n1 = next(ips_generator)
    res = np.stack(ips_generator, axis=0)
