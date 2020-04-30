import numpy as np
from grid_world import GridWorld
from variables import*

class PolicyIteration(object):
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

    def initialize_P(self):
        P = {}
        for i in range(self.grid.rows):
            for j in range(self.grid.cols):
                if (i, j) not in [death, goal]:
                    P[(i,j)] = np.random.choice(self.actions)
        self.P = P


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
                new_v += self.dynamic_p * (r + gamma * self.V[s_new])
            self.V[s] = new_v
            diff = max(diff, np.abs(old_v - new_v))
        return diff

    def policy_evaluation(self):
        while True:
            diff = self.value_step()
            if diff < delta:
                return None

    def improvement_step(self):
        policy_stable = True
        for s in self.S:
            old_action = self.P[s]
            action_values = []
            for a in self.actions:
                self.grid.set_state(s)
                self.grid.move(a)
                s_new = self.grid.current_state()
                r = self.rewards.get(s_new,0)
                new_v = r + gamma * self.V[s_new]
                action_values.append(new_v)

            idx = np.argmax(action_values)
            new_action = self.actions[idx]
            self.P[s] = new_action
            if old_action != new_action:
               policy_stable = False

        return policy_stable

    def policy_iteration(self):
        self.initialize_V()
        self.initialize_P()
        print("Initial Policy:")
        self.print_policy()
        while True:
            self.policy_evaluation()
            if self.improvement_step():

                print("Final Policy:")
                self.print_policy()

                print("Final Values:")
                self.print_values()
                return None

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

    def print_policy(self):
        for i in range(self.grid.rows):
            print("---------------------------")
            for j in range(self.grid.cols):
                a = self.P.get((i,j), '#')
                print("  %s  |" % a, end="")
            print("")
        print("------------------------")

if __name__ == "__main__":
    rl_model = PolicyIteration()
    rl_model.policy_iteration()