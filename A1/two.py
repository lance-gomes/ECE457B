import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np

def sgn(x):
  if x > 0:
    return 1
  else:
    return -1

def adaline(weights, inputs):
  return sgn(np.dot(weights, inputs))

def madaline(inputs):

  # Add bias weight of 1
  input_a1 = inputs + [1]
  input_a2 = inputs + [1]

  sgn_1 = adaline(np.array([-1, 1, 0.5]), np.array(input_a1))
  sgn_2 = adaline(np.array([-1, 1, -0.5]), np.array(input_a2))

  weights = np.array([1, -1, -1])
  inputs = np.array([sgn_1, sgn_2, 1])

  return adaline(weights, inputs)

def graph():
  x = [i for i in range(-1, 3)]
  madaline_x = [i for i in range(0, 2)]
  y_1 = [x_val - 0.5 for x_val in x]
  y_2 = [x_val + 0.5 for x_val in x]
  y_3 = [-x_val + 1 for x_val in madaline_x]

  plt.plot(x, y_1, label="y = x - 0.5")
  plt.plot(x, y_2, label="y = x + 0.5")
  plt.plot(madaline_x, y_3, label="y = -x + 1")

  plt.plot(0,0, "ro")
  plt.plot(1,0, "ro")
  plt.plot(0,1, "ro")
  plt.plot(1,1, "ro")

  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('X vs Y')
  plt.legend()
  plt.show()


if __name__ == "__main__":
  print(madaline([-1,-1]))
  print(madaline([-1,1]))
  print(madaline([1,-1]))
  print(madaline([1,1]))
  graph()


