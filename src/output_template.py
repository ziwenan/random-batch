# src/output_template.py

from dataclasses import dataclass, field

import numpy as np
import pygame


@dataclass(kw_only=True)
class OutputTemplate:
    """Class for storing output from evolving an interacting particle system."""
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

    def render(self, lower=None, upper=None, width=1366, height=768, loop=True):
        if lower is None:
            lower = self.position.min(axis=(0,1))
        if upper is None:
            upper = self.position.max(axis=(0,1))
        padx = int(width * 0.05)
        pady = int(height * 0.05)
        scalex = 0.9 * width / (upper[0] - lower[0])
        scaley = 0.9 * height / (upper[1] - lower[1])

        black, white, blue = (20, 20, 20), (230, 230, 230), (0, 154, 255)

        pygame.init()
        pygame.display.set_caption("3D cube Projection")
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        fps = 10

        while True:
            pygame.event.get()
            for i in range(self.length):
                clock.tick(fps)
                screen.fill(white)

                for point in self.position[i]:
                    x = padx + int((point[0] - lower[0]) * scalex)
                    y = pady + int((point[1] - lower[1]) * scaley)
                    pygame.draw.circle(screen, blue, (x, y), 3)
                pygame.display.update()
            if not loop:
                break

        pygame.display.quit()
        pygame.quit()

