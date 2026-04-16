import numpy as np


def signed_dist(x, theta, theta_0):
    return (np.dot(x.T, theta) + theta_0)/np.linalg.norm(theta)


theta = np.array([3,4])
theta_0 = 5
p = np.array([4, -1/2])

print(signed_dist(p, theta, theta_0))
