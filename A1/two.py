import random
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def sgn(x):
  if x > 0:
    return 1
  else:
    return 0

def adaline(weights, inputs):
  return sgn(np.dot(weights, inputs))

def madaline(inputs):

  # Add bias weight of 1
  input_a1 = inputs + [1]
  input_a2 = inputs + [1]

  sgn_1 = adaline(np.array([-1, 1, 0.5]), np.array(input_a1))
  sgn_2 = adaline(np.array([-1, 1, -0.5]), np.array(input_a2))

  weights = np.array([2, -2, -1])
  inputs = np.array([sgn_1, sgn_2, 1])

  return adaline(weights, inputs)

if __name__ == "__main__":
  print(madaline([0,0]))
  print(madaline([0,1]))
  print(madaline([1,0]))
  print(madaline([1,1]))


