# src/output_template.py

from dataclasses import dataclass, field

import numpy as np
import pygame


@dataclass(kw_only=True)
class OutputTemplate:
    """Class for storing output from the IPS.evolve method."""
    position: np.ndarray = field(default=None)
    index: np.ndarray = field(default=None)
    attr_names: list[str] = field(default_factory=lambda: ['position', 'index'])
    length: int = 0
    num: int = None
    d: int = None

    def __post_init__(self):
        if self.position is None and self.num is not None and self.d is not None:
            if self.d == 1:
                self.position = np.empty((self.length, self.num,))
            else:
                self.position = np.empty((self.length, self.num, self.d))
        if self.index is None and self.num is not None and self.d is not None:
            self.index = np.empty((self.length, self.num,))

    def append(self, value):
        # slow
        for attr_name in self.attr_names:
            curr = getattr(self, attr_name)
            to_append = value[attr_name]
            if isinstance(curr, np.ndarray):
                setattr(self, attr_name, np.append(curr, to_append[np.newaxis, :], axis=0))
            else:  # should be list in most of the time
                getattr(self, attr_name, curr.append(to_append))
        self.length += 1

    def plot(self, which='position'):
        target = getattr(self, which)
        if isinstance(target, np.ndarray):
            raise TypeError(f'The object to plot, "{which}", must be a numpy array.')
        if self.d == 1:
            if target.shape != (self.length, self.num):
                raise ValueError(f'The object to plot, "{which}", must be a ({self.length}, {self.num}) numpy array.')
        else:
            if target.shape != (self.length, self.num, self.d):
                raise ValueError(
                    f'The object to plot, "{which}", must be a ({self.length}, {self.num}, {self.d}) numpy array.')
        raise NotImplementedError

    def render(self, color=None, lower=None, upper=None, width=1366, height=768, loop=True, dumpFrames=False,
               backend=None, caption='IPS simulator', bg=(255, 255, 255)):
        '''
        Render animations of the particle system.

        Args:
            color: A color (Color or int or tuple(int, int, int, [int])) or a list of colors or an attribute name.
            lower (list[float]): A list of minimum value for each axis. Inherits from "boundary_condition"
            upper (list[float]): A list of maximum value for each axis.
            width (int): Window width.
            height (int): Window height.
            loop (bool): Loop animations.
            dumpFrames (bool): Dump frames to local depository.
            backend (str): Image rendering backend, "pygame" (default) or "panda3D" or "matplotlib".

        Returns:
            Any.
        '''
        assert self.d <= 3, f'Rendering is not supported for {self.d}D space.'

        # setup colors
        if color is None:
            if getattr(self, 'velocity', None) is not None:
                pass # use velocity to define color
            color = pygame.Color('blue')
        if isinstance(color, list):
            assert len(color) == self.num, f'Length of "color" list must be {self.num}.'
        else:
            color = [color for _ in range(self.num)]
        bg = pygame.Color(bg)
        box_color = pygame.Color(255 - bg.r, 255 - bg.g, 255 - bg.b)

        # setup window size and region of plotting
        if lower is None:
            lower = self.position.min(axis=(0, 1))
        if upper is None:
            upper = self.position.max(axis=(0, 1))
        padx = int(width * 0.05)
        pady = int(height * 0.05)
        scalex = 0.9 * width / (upper[0] - lower[0])
        scaley = 0.9 * height / (upper[1] - lower[1])

        def pos_to_pixel(pos):
            x = padx + int((pos[0] - lower[0]) * scalex)
            y = pady + int((pos[1] - lower[1]) * scaley)
            return x, y

        topleft = pos_to_pixel((lower[0], lower[1]))
        topright = pos_to_pixel((upper[0], lower[1]))
        bottomleft = pos_to_pixel((lower[0], upper[1]))
        bottomright = pos_to_pixel((upper[0], upper[1]))

        # dump frames and animation backend
        if dumpFrames:
            raise NotImplementedError
        if backend is None:
            backend = 'pygame'
        if backend not in ['pygame', 'panda3D', 'matplotlib']:
            raise ValueError(f'"backend" must be "pygame" or "panda3D" or "matplotlib"')
        if backend in ['panda3D', 'matplotlib']:
            raise NotImplementedError

        pygame.init()
        pygame.display.set_caption(caption)
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        fps = 10

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            for k in range(self.length):
                clock.tick(fps)
                screen.fill(bg)

                # box
                pygame.draw.line(screen, box_color, bottomleft, bottomright, width=3)
                pygame.draw.line(screen, box_color, bottomleft, topleft, width=3)
                pygame.draw.line(screen, box_color, topleft, topright, width=3)
                pygame.draw.line(screen, box_color, topright, bottomright, width=3)

                for i, point in enumerate(self.position[k]):
                    if point[0] < lower[0] or point[0] > upper[0] or point[1] < lower[1] or point[1] > upper[1]:
                        continue
                    if self.d == 1:
                        pixel = pos_to_pixel((point[0], (lower[1] + upper[1]) / 2))
                    elif self.d == 2:
                        pixel = pos_to_pixel(point)
                    else:  # self == 3
                        raise NotImplementedError
                    x = pixel[0]
                    y = pixel[1]
                    pygame.draw.circle(screen, color[i], (x, y), 3)
                pygame.display.update()
            if not loop:
                done = True

        pygame.display.quit()
        pygame.quit()
