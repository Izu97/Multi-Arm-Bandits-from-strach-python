import numpy as np
from grid_world import GridWorld
from variables import*

class IterativePolicyEvaluation(object):
    def __init__(self):
        self.grid = GridWorld()
        self.rewards = rewards
        self.actions = actions

    def initialize_V(self):
        V = {}
        S = []
        for i in range(self.grid.rows):
            for j in range(self.grid.cols):
                V[(i,j)] = 0
                if (i, j) not in [death, goal]:
                    S.append((i,j))
        self.V = V
        self.S = S
        self.dynamic_p = 1.0 / len(S)

    def value_step(self):
        diff = 0
        old_V = self.V
        for s in self.S:
            new_v = 0
            old_v = old_V[s]
            for a in self.actions:
                self.grid.set_state(s)
                self.grid.move(a)
                s_new = self.grid.current_state()
                r = self.rewards.get(s_new,0)
                if self.grid.game_over(s_new):
                    new_v += self.dynamic_p * r
                    break
                else:
                    new_v += self.dynamic_p * (r + gamma * self.V[s_new])
            self.V[s] = new_v
            diff = max(diff, np.abs(old_v - new_v))
        return diff

    def policy_evaluation(self):
        self.initialize_V()
        step = 0
        while True:
            diff = self.value_step()
            if step % verbose == 0:
                print(step)
            if diff < delta:
                self.print_values()
                break
            step += 1

    def print_values(self):
        for i in range(self.grid.rows):
            print("------------------------")
            for j in range(self.grid.cols):
                v = self.V.get((i,j), 0)
                if v >= 0:
                    print(" %.2f|" % v, end="")
                else:
                    print("%.2f|" % v, end="")
            print("")
        print("------------------------")

if __name__ == "__main__":
    rl_model = IterativePolicyEvaluation()
    rl_model.policy_evaluation()
