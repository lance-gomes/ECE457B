import random
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

C1 = [[-1, 0.8, 0.7, 1.2], [-1, -0.8,-0.7, 0.2], [-1, -0.5,0.3,-0.2], [-1, -2.8, -0.1, -2]]
C2 = [[-1, 1.2,-1.7, 2.2] , [-1, -0.8,-2, 0.5], [-1, -0.5,-2.7,-1.2], [-1, 2.8, -1.4, 2.1]]
learning_rate = 0.6

def step(x):
  if x <= 0:
    return 0
  else:
    return 1

def sigmoid(x):
  return (1 / (1 + math.exp(-x)))

def perceptron(x):
  training_data, target = x
  weights = [random.uniform(-1, 1) for i in range(len(training_data[0]))]

  training_finished = False

  while not training_finished:
    training_finished = True

    for i, t_set in enumerate(training_data):
      output = 0

      for j in range(len(t_set)):
        output += t_set[j] * weights[j]
      o = step(output)

      for j in range(len(t_set)):
        delta_weight = learning_rate * (target[i] - o) * t_set[j]
        weights[j] += delta_weight

        if delta_weight != 0:
          training_finished = False

  print("Perceptron Weights : ", weights)
  return weights

def adaline(x):
  training_data, target = x
  weights = [random.uniform(-1, 1) for i in range(len(training_data[0]))]

  training_finished = False

  while not training_finished:
    training_finished = True

    for i, t_set in enumerate(training_data):
      output = 0

      for j in range(len(t_set)):
        output += t_set[j] * weights[j]

      s = sigmoid(output)

      for j in range(len(t_set)):
        delta_weight = learning_rate * (target[i] - s) * (s*s * math.exp(-output)) * t_set[j]
        weights[j] += delta_weight

        if delta_weight > 0.000001:
          training_finished = False

  print()
  print("Adaline Weights: ", weights)
  return weights


def graph():
  points_x_c1 = [x[1] for x in C1]
  points_x_c2 = [x[1] for x in C2]

  points_y_c1 = [x[2] for x in C1]
  points_y_c2 = [x[2] for x in C2]

  points_z_c1 = [x[3] for x in C1]
  points_z_c2 = [x[3] for x in C2]

  fig = plt.figure(0)
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  ax.scatter(points_x_c1, points_y_c1, points_z_c1, c="b", marker="o", depthshade=False)
  ax.scatter(points_x_c2, points_y_c2, points_z_c2, c="r", marker="x", depthshade=False)

  weights_perceptron = perceptron(get_data())
  l_x1, l_y1, l_z1 = get_plane_points(weights_perceptron)
  ax.plot_surface(l_x1, l_y1, l_z1)
  ax.legend()

  fig2 = plt.figure(1)
  ax2 = fig2.add_subplot(111, projection='3d')
  ax2.set_xlabel('x')
  ax2.set_ylabel('y')
  ax2.set_zlabel('z')

  ax2.scatter(points_x_c1, points_y_c1, points_z_c1, c="b", marker="o", depthshade=False)
  ax2.scatter(points_x_c2, points_y_c2, points_z_c2, c="r", marker="x", depthshade=False)

  weights_adaline = adaline(get_data())
  l_x2, l_y2, l_z2 = get_plane_points(weights_adaline)
  ax2.plot_surface(l_x2, l_y2, l_z2)

  ax.scatter([-1.4], [-1.5], [2], c="g", marker="o", depthshade=False)
  ax2.scatter([-1.4], [-1.5], [2], c="g", marker="o", depthshade=False)

  plt.show()

def get_plane_points(weights):
  x, y = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
  z = (weights[0] - weights[1]*x - weights[2]*y) / weights[3]

  return x, y, z

def get_data():
  training_data = C1 + C2
  training_targets = []
  for _ in range(len(C1)):
    training_targets.append(0) #C1 is 0
  for _ in range(len(C2)):
    training_targets.append(1) #C2 is 1

  return training_data, training_targets

def main():
  graph()


if __name__ == "__main__":
  main()