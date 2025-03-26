import matplotlib.pyplot as plt
import numpy as np


velocities = np.linspace(0, 10)

penalty = (velocities/4) ** 4

plt.plot(velocities, penalty)
plt.show()
