from scipy.interpolate import RegularGridInterpolator
import numpy as np
import time

def f(x, y, z):

    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)

y = np.linspace(4, 7, 22)

z = np.linspace(7, 9, 33)

xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

data = f(xg, yg, zg)

t0 = time.time()

interp = RegularGridInterpolator((x, y, z), data)

t1 = time.time()

pts = np.array([[2.1, 6.2, 8.3],

                [3.3, 5.2, 7.1]])

print(t1-t0)

interp(pts)

t2 = time.time()

print(t2-t1)
