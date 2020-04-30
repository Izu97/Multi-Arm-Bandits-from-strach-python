# Grid Parameters
cols = 4
rows = 3
start = (3,1)
goal = (0,3)
death = (0,2)
delta = 1e-4
gamma = 0.9
rewards = {
        goal : 10,
        death : -5
          }
actions = ['U', 'D', 'R', 'L']