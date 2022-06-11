# src/interacting_particle_system.py
import dataclasses
import inspect
import math
import warnings
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.controller import Controller
from src.output_template import OutputTemplate
from src.particles import Particles


class IPS(Particles):
    '''
    IPS (interacting particle system).

    IPS's are continuous-time Markov jump processes describing the collective behavior of stochastically interacting
    particles. An IPS class object has two different types of components, properties and controllers. Properties are
    used to store the information of current states of an IPS. Controllers are used to modify positions and velocities
    of particles. Properties can be used as input arguments of controller functions. The "update" method updates
    properties for a short time interval dt. The "evolve" method updates properties iteratively until time T.

    Properties, including built-in and custom ones, of the IPS are maintained as class attributes. Built-in properties
    are initialised upon construction, and will be updated while running the "update" or "evolve" method. There are
    chiefly two purposes of a property, to get track of the current state, and to be supplied as an input variable of
    another function. Both public attributes (e.g., position, velocity, etc.) and private attributes with a getter
    method (d, num, index) are built-in properties. Custom properties are supplements to built-in properties. They are
    user-defined properties in order to provide additional features of the IPS. Custom properties can be added/accessed
    with the "add_property/get_property" method. See examples for more hints.

    Attributes:
        position: 1D (particles) or 2D (particles by axes) numpy array, position of the particles.
        velocity: 1D (particles) or 2D (particles by axes) numpy array, velocity of the particles.
        acceleration: 1D (particles) or 2D (particles by axes) numpy array, acceleration of the particles.
        lower (list[float]): Lower bound(s) of the particle space.
        upper (list[float]): Upper bound(s) of the particle space.
        boundary_condition (str): Boundary condition of the particle space.
        speed: Numpy array, speed of the particles.
        t (float): Current time of the IPS.
        order (str): Order of the IPS, "first" or "second". In a first order IPS, controllers modifies particle
                     positions. In a second order IPS, controllers modifies particle velocities.
        custom_properties (dict['wrapper_fn_compute':Callable, 'eager_evaluation':bool]): Dict.
        controllers (dict): Dict.
        controller_objects (dict): Dict.
        column_names (List[str]): List of column names.
    '''

    def __init__(self, position, order='first', t=0, velocity=None, acceleration=None, lower=None, upper=None,
                 boundary_condition=None, index=None, column_names=None):
        '''
        Initialise an IPS class object.

        Args:
            position: 1D (particles) or 2D (particles by axes) array-like object, position of the particles.
            order (str): Order of the interacting particle system, "first" or "second".
            t (float): Current time, a non-negative float.
            velocity: 1D (particles) or 2D (particles by axes) Array-like object, velocity of the particles.
            lower (list[float]): Lower bound(s) of the particle space. Length should match the dimension of the space.
            upper (list[float]): Upper bound(s) of the particle space. Length should match the dimension of the space.
            boundary_condition (str): Boundary condition of the particle space, None or "absorbing" or "elastic" or
                                      "periodic".
            column_names (List[str]): List of column names.
            index: Array, index of the particles.
        '''
        if order not in ['first', 'second']:
            raise ValueError(f'"order" must be "first" or "second".')
        super().__init__(position, velocity=velocity, acceleration=acceleration, lower=lower, upper=upper,
                         boundary_condition=boundary_condition, index=index, column_names=column_names)
        self.calc_speed()
        self.order = order
        self.t = t if t > 0 else 0
        self.custom_properties = dict()
        self.controllers = dict()
        self.controller_objects = dict()
        self._unnamed_controller_counter = 0

    def add_property(self, value, name, fn_compute=None, eager_evaluation=False):
        '''
        Adds a custom property to the IPS.

        Custom properties of IPS are treated as read-only attributes. Writing to custom properties is viable only
        through the computing function given by the "fn_compute" argument. Properties can be used as input arguments 
        of a controller function or the computing function for another property.

        By default, if "fn_compute" is None, the custom property cannot be modified. Conversely, if a callable function
        is assigned to "fn_compute", the custom property will be handled as a computed attribute. It will be
        updated/written upon calling or while running the "update" and "evolve" method.

        Args:
            value: Initial value of the custom property.
            name (str): Name of the custom property.
            fn_compute: If provided, it must be a callable that will be called to modify the custom property. Input of 
                        the callable should be keyword argument(s), where a keyword must have the name with an existing
                        property. Output is the computed property, which should preserve data type.
            eager_evaluation (bool): Only used when "fn_compute" is provided. If True, "fn_compute" will be handled as 
                                     an eager function (i.e., evaluate whenever "update" or "evolve" or "get_property" 
                                     is called), or a lazy one (i.e., evaluate only when "get_property" is called).
        '''
        if getattr(self, name, None) is not None:
            warnings.warn(f"Overwriting attribute {name}")
        if fn_compute is None:
            property_ = {'wrapper_fn_compute': None, 'eager_evaluation': False}
            setattr(self, name, value)
        else:
            setattr(self, name, value)
            try:
                valid = self._check_property_validity(fn_compute)
                assert (valid is not None), f'"{name}"\'s computing function is not callable.'
                property_ = {'wrapper_fn_compute': valid, 'eager_evaluation': eager_evaluation}
                setattr(self, name, property_)
            except Exception as exc:
                delattr(self, name)
                raise ValueError(
                    f'Fail to add property "{name}". The following error occurred while running \
                    "{fn_compute.__name__}": {str(exc)}') from None

        if name not in self.custom_properties:
            self.custom_properties[name] = property_

    def _wrapper_compute_property(self, fn_compute):
        '''
        Creates a wrapper function to compute custom property.

        Args:
            fn_compute: A function used to update the custom property. Input arguments should use the same keywords with
                        existing properties. Output should be the new value of the custom property.

        Returns:
            Wrapper function for computing property.
        '''

        # TODO: use numba
        fn_signatures_keys = inspect.signature(fn_compute).parameters.keys()

        def inner():
            # fn_signatures_keys = inspect.signature(fn_compute).parameters.keys()
            kwargs = {}
            for key in fn_signatures_keys:
                if getattr(self, key, None) is not None:
                    kwargs[key] = self.get_property(key)
            return fn_compute(**kwargs)

        return inner

    def _check_property_validity(self, fn_compute):
        if not callable(fn_compute):
            return None
        wrapper_fn_compute = self._wrapper_compute_property(fn_compute)
        wrapper_fn_compute()
        return wrapper_fn_compute

    def get_property(self, name):
        '''
        Getter method for custom property, update if possible.

        Args:
            name: Name of the property.

        Returns:
            Value of the property.
        '''
        if name in self.custom_properties:
            wrapper = self.custom_properties[name].get('wrapper_fn_compute')
            if wrapper is not None:  # update
                new_value = wrapper()
                setattr(self, name, new_value)
            return getattr(self, name)
        else:
            return getattr(self, name)

    def add_controller(self, func_or_value, type='external', name=None, use_numba=True, force_nonvectorise=False,
                       verbose=False, suppress_warnings=False):
        '''
        Adds a controller to the IPS.

        Args:
            func_or_value: The following three types of "func_or_value" argument are acceptable,
                          (1) a non-vectorised function (operates on one particle per execution),
                          (2) a vectorised function (operates on a set of particles per execution),
                          (3) (a) fixed value(s) (scalar or array-like).
                          If "func_or_value" is a callable function and the number of arguments of the function is more
                          than two, the first argument must be named as "x", representing the target
                          (positions/velocities or binary interactions of them in case of an "interacting" controller)
                          of the calculation. The rest must be correctly named to match existing properties.
            type (str): Type of the controller. An "external" controller is a global effect influencing all the
                        particles in the system. An "interacting" controller affects binary pairs of particles depending
                        on their relative locations. In case of a non-vectorised function, the input variable of an
                        "external" controller function is the current location/velocity of a particle, whereas the input
                        of an "interacting" one is the difference of location/velocity.
            name (str): Name of the controller. Avoid using preoccupied names such as "x" and "all".
            use_numba (bool): Only used when "func_or_value" is a function, use numba just-in-time compiler to
                              optimise function, works only if the function takes one argument as numba does not support 
                              **kwargs arguments.
            force_nonvectorise (bool): Only used when "func_or_value" is a function, force using non-vectorised mode
                                       if True to prevent unintended automatic vectorisation.
            verbose (bool):  Verbose messages.
            suppress_warnings (bool): Suppress warning messages.
        '''
        if type not in ['external', 'interacting']:
            raise ValueError(f'Controller "type" must be "external" or "interaction".')
        if type == 'external':
            target = self.position
        else:  # type == 'interacting'
            target = self.pairwise_diff()
        kwargs = {}
        if callable(func_or_value):
            fn_parameters = dict(inspect.signature(func_or_value).parameters)
            if len(fn_parameters.keys()) > 1:
                if 'x' not in fn_parameters.keys():
                    raise KeyError(f'"x" must be given as a keyword argument in function {func_or_value.__name__}')
                fn_parameters.pop('x', None)
                for key, value in fn_parameters.items():
                    if value.default is value.empty:  # no default is given
                        if getattr(self, key, None) is None:
                            raise AttributeError(
                                f'Argument {key} is required by function {func_or_value.__name__}, \
                                but no attribute was found.')
                        else:
                            kwargs[key] = self.get_property(key)
                    else:
                        kwargs[key] = value.default
                        self.add_property(value.default, key)

        C = Controller.init(func_or_value, target, use_numba=use_numba,
                            force_nonvectorise=force_nonvectorise,
                            verbose=verbose, suppress_warnings=suppress_warnings, **kwargs)

        def wrapper(x):
            wrapper_kwargs = {}
            for key in kwargs.keys():
                wrapper_kwargs[key] = self.get_property(key)
            return C.apply_to(x, **wrapper_kwargs)

        if name is None:
            self._unnamed_controller_counter += 1
            name = f'unnamed controller {self._unnamed_controller_counter}'
        controller = dict(name=name, type=type, properties=set(kwargs.keys()), wrapper=wrapper, what=C.what,
                          function=C.function, value=C.value, is_vectorised=C.is_vectorised, use_numba=C.use_numba)
        self.controllers[name] = controller
        self.controller_objects[name] = C

    def reset_states(self):
        '''Resets particle velocities if order is one, accelerations if order is two.'''
        if self.order == 'first':
            self.velocity = np.zeros_like(self.velocity)
        else:  # order == 'second'
            self.acceleration = np.zeros_like(self.acceleration)

    def apply_controllers(self, which='all'):
        '''
        Applies external and interacting controllers to the particles.

        Args:
            which (str): "all" to apply all controllers, or specify the name of a controller to apply.
        '''
        if which not in ['all', self.controllers.keys()]:
            raise ValueError(f'"which" must be "all" or {", ".join(s for s in self.controllers.keys())}')
        increment = 0
        if which == 'all':
            for controller in self.controllers.values():
                if controller['type'] == 'external':
                    target = self.position
                else:  # controller['type'] == 'interacting'
                    target = self.pairwise_diff()
                controller_wrapper = controller['wrapper']
                increment = controller_wrapper(target)
            if self.order == 'first':
                self.velocity += increment
            else:  # order == 'second'
                self.acceleration += increment
        else:
            controller = self.controllers[which]
            if controller['type'] == 'external':
                target = self.position
            else:  # controller['type'] == 'interacting'
                target = self.pairwise_diff()
            controller_wrapper = controller['wrapper']
            increment = controller_wrapper(target)
            if self.order == 'first':
                self.velocity += increment
            else:  # order == 'second'
                self.acceleration += increment

    def apply_stochastic_terms(self, dt, sigma, gamma):
        '''Adds stochsticity to particle positions.'''
        if self.order == 'first':
            if self._d == 1:
                self.position += sigma * np.sqrt(dt) * np.random.randn(self._num)
            else:
                self.position += sigma * np.sqrt(dt) * np.random.randn(self._num, self._d)
        else:  # order == 'second'
            if self._d == 1:
                self.acceleration -= gamma * self.velocity
                self.velocity += sigma * np.sqrt(dt) * np.random.randn(self._num)
            else:
                self.acceleration -= gamma * self.velocity
                self.velocity += sigma * np.sqrt(dt) * np.random.randn(self._num, self._d)

    def update_states(self, dt):
        '''
        Updates states of particles.
        
        Arg:
            dt (float): Time interval.
        '''
        if self.order == 'first':
            self.position += self.velocity * dt
        else:  # order == 'second'
            self.velocity += self.acceleration * dt
            self.position += self.velocity * dt

    def shuffle(self, seed=None, return_index=False):
        '''
        Shuffles particle positions, velocities, accelerations. Also shuffles custom properties if they are of size
        self.num.
        '''
        index = super().shuffle(seed=seed, return_index=True)
        if len(self.custom_properties):
            for property_name in self.custom_properties:
                property_ = self.get_property(property_name)
                if getattr(property_, '__len__', None) is not None and len(property_) == self._num:
                    setattr(self, property_name, property_[index])
        if return_index:
            return index
        else:
            return None

    def update_rbm(self, dt, sigma=None, gamma=None):
        if len(self.controllers):
            self.shuffle()
            self.reset_states()
            self.apply_controllers()
            if sigma is not None or gamma is not None:
                self.apply_stochastic_terms(dt, sigma, gamma)
            self.update_states(dt)
            if self.boundary_condition is not None:
                self.apply_boundary_condition()
        if len(self.custom_properties):
            for property_name in self.custom_properties:
                wrapper = self.custom_properties[property_name].get('wrapper_fn_compute')
                if wrapper is not None:  # update
                    new_value = wrapper()
                    setattr(self, property_name, new_value)

    def update_rbm_r(self, dt, sigma, gamma):
        raise NotImplementedError

    def update(self, dt, method='rbm', sigma=None, gamma=None):
        if method not in ['rbm', 'rbm-r']:
            raise ValueError(f'"method" must be "rbm" or "rbm-r".')
        if method == 'rbm':
            self.update_rbm(dt=dt, sigma=sigma)
        else:  # method == 'rbm-r'
            self.update_rbm_r(dt=dt, sigma=sigma, gamma=gamma)

    def evolve_generator(self, dt, T, sigma=None, gamma=None, method='rbm', options=None, thinning=None,
                         early_stopping=None):
        '''
        Evolves the IPS using random batch method.

        Create a generator of evolved result. More flexible than the "evolve" method.

        Args:
            dt (float): Time interval.
            T (float): Time when evolution ends.
            sigma (Optional[float]): Stochastic term, see equation.
            gamma (Optional[float]): Stochastic term, see equation.
            method (Optional[str]): Method to evolve IPS, "rbm" (default) for the standard random batch method, "rbm_r"
                                    for random batch with replacement.
            options (List[str]): Extra properties to be reported.
            thinning (Optional[int]): Thinning of the output results.
            early_stopping (Optional[dict]): Dictionary dict(property=<property name>, value=<value by which to stop>).
        Returns:

        '''
        if method not in ['rbm', 'rbm-r']:
            raise ValueError(f'"method" must be "rbm" or "rbm-r".')
        if self.order == 'first' and gamma is not None:
            warnings.warn('"gamma" is ignored because "order" is "first".')

        # define early stopping rules
        if early_stopping is None:
            early_stopping = dict(property=None, value=None)
        early_stopping_property = early_stopping['property']
        if early_stopping_property is None:
            def stops_if():
                return False
        else:
            early_stopping_value = early_stopping['value']

            def is_curr_ge_target() -> bool:
                curr = self.get_property(early_stopping_property)
                return curr >= early_stopping_value

            if is_curr_ge_target():
                def stops_if() -> bool:
                    return not is_curr_ge_target()
            else:
                def stops_if() -> bool:
                    return is_curr_ge_target()

        # options
        output = ['position', 'index']
        if options is not None:
            properties = ['position', 'velocity', 'acceleration', 'index', 'speed', *self.custom_properties]
            for item in options:
                if item in properties:
                    output.append(item)
                else:
                    raise ValueError(
                        f'Property "{item}" not found. "options" must be {", ".join([p for p in properties])}')
        output = set(output)

        # main loop
        count = 0
        while self.t < T:
            self.update(dt, method=method, sigma=sigma, gamma=gamma)
            self.t += dt
            count += 1

            results = dict()
            if thinning is None or count % thinning == 0:
                for item in output:
                    results[item] = self.get_property(item)
                yield results

                if stops_if():
                    print(
                        f'Trigger early stopping as "{early_stopping_property}" \
                        is {self.get_property(early_stopping_property)}.')
                    break

    def evolve(self, dt, T, sigma=None, gamma=None, method='rbm', options=None, thinning=None,
               early_stopping=None):
        # generator of results
        generator = self.evolve_generator(dt=dt, T=T, sigma=sigma, gamma=gamma, method=method, options=options,
                                          thinning=thinning, early_stopping=early_stopping)

        # create a dataclass to store results
        if options is None:
            Output = dataclasses.make_dataclass('Output', [], bases=(OutputTemplate,))
        else:
            names_output = []
            output = dict()
            if options is not None:
                properties = ['velocity', 'acceleration', 'speed', *self.custom_properties]
                for item in options:
                    if item in properties:
                        names_output.append(item)
                    else:
                        raise ValueError(
                            f'Property "{item}" not found. "options" must be {", ".join([p for p in properties])}')
            names_output = set(names_output)
            for name_output in names_output:
                property_ = self.get_property(name_output)
                try:
                    curr = np.asarray(property_)
                    output[name_output] = dict(type='ndarray', shape=curr.shape, dtype=curr.dtype)
                except:
                    output[name_output] = dict(type='list', shape=None, dtype=None)

            # __post_init__ method for dataclasses.make_dataclass
            def my_init(self):
                OutputTemplate.__post_init__(self)
                for key, value in output.items():
                    if getattr(self, key) is None and self.num is not None and self.d is not None:
                        if value['type'] == 'ndarray':
                            setattr(self, key, np.empty((0, *value['shape']), dtype=value['dtype']))
                        else:
                            setattr(self, key, [])
                    self.attr_names.append(key)

            Output = dataclasses.make_dataclass('Output',
                                                [(key, np.ndarray if value['type'] == 'ndarray' else list,
                                                  dataclasses.field(default=None)) for key, value in output.items()],
                                                bases=(OutputTemplate,),
                                                namespace={'__post_init__': my_init},
                                                kw_only=True)
        results = Output(num=self._num, d=self._d)

        total = math.ceil((T - self.t) / dt) if thinning is None else math.ceil((T - self.t) / dt / thinning)
        for item in tqdm(generator, total=total):
            results.append(item)
        return results

    def __str__(self):
        self.calc_speed()
        string = []
        string.append(f'{self.order}-order IPS @t={self.t}, average speed {np.average(self.get_property("speed")):.2f}')
        string.append(super().__str__())
        if len(self.custom_properties):
            string.append(f'custom properties: {", ".join([str(s) for s in self.custom_properties])}')
        if len(self.controllers):
            for controller in self.controllers.values():
                name = controller['name']
                args = ', '.join(str(x) for x in set('x').union(controller["properties"]))
                string.append(f'{name}: {self.controller_objects[name].__str__(args)}')
        return '\n'.join(s for s in string if s)


if __name__ == '__main__':
    import time
    n = 100
    ips = IPS(np.random.rand(n, 2), order='first')
    ips.add_property(-10, 'alpha')
    ips.add_controller(0, type='external', name='V')
    ips.add_property(0, 'phi', fn_compute=lambda position: abs(position) <= 1)
    ips.velocity = np.ones_like(ips.velocity)
    ips.add_property(0, 'average_speed', fn_compute=lambda speed: np.mean(speed))
    ips.add_controller(lambda x, alpha: alpha * np.where(abs(x) <= 0.3, 1, 0) * (-x), type='interacting', name='K')
    print(ips)
    start = time.perf_counter()
    res = ips.evolve(dt=1e-3, T=10, sigma=0.005, thinning=100,
                     early_stopping=dict(property='average_speed', value=0.01))
    # res = ips.evolve_generator(dt=1e-3, T=1, sigma=0.005, early_stopping=dict(property='average_speed', value=0.0001))
    # for _ in res:
    #     pass
    # print(ips)
    # print(res)
    end = time.perf_counter()
    print(f'Time Elapsed: {end - start:.5f} sec')
    res.render()
