import matplotlib.pyplot as plt
import numpy as np


velocity = 10

distance = np.linspace(0, 100)

epsilon = 0.5
penalty = velocity / (distance**2 + epsilon)

plt.plot(distance, penalty)
plt.show()
