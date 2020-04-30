import numpy as np
from variables import*

class GridWorld(object):
    def __init__(self):
        self.cols = cols
        self.rows = rows
        self.i = start[0]
        self.j = start[1]

    def current_state(self):
        return (self.i, self.j)

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def game_over(self, s):
        return (s in [death, goal])

    def move(self, action):
        #move upward
        if action == 'U':
            k = self.i
            self.i = max(k-1, 0)

        if action == 'D':
            k = self.i
            self.i = min(k+1, self.rows - 1)

        if action == 'L':
            k = self.j
            self.j = max(k-1, 0)

        if action == 'R':
            k = self.j
            self.j = min(k+1, self.cols - 1)
