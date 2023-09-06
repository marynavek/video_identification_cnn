import numpy as np
print("Hello world")


a = np.array([[1, 2, 3], [3,2,1]])

b = np.array([[4, 5, 6], [6,5,4]])

vs = np.vstack((a,b))
# print(vs)

ds = np.dstack((a,b))
# print(ds)

c = np.array([[7, 8, 9], [9,8,7]])

vs1 = np.vstack((vs,c))
print(vs1)

ds1 = np.dstack((ds,c))
print(ds1)

stackX = ds1.reshape((ds1.shape[0], ds1.shape[1]*ds1.shape[2]))

print(stackX)