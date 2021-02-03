import random
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

def perceptron():
  training_data, target = get_data()
  weights = [0.5, 0.5, 0.5]
  bias = 0.5

  training_finished = False

  while not training_finished:
    training_finished = True

    for i, t_set in enumerate(training_data):
      output = - bias
      for j in range(1, len(t_set)):
        output += t_set[j] * weights[j-1]
      o = step(output)
      for j in range(1, len(t_set)):
        delta_weight = learning_rate * (target[i] - o) * t_set[j]
        weights[j-1] += delta_weight

        if delta_weight != 0:
          training_finished = False
      delta_bias = -learning_rate * (target[i] - o)
      bias += delta_bias
  return weights, bias

def adaline():
  training_data, target = get_data()
  weights = [random.uniform(-1, 1) for i in range(len(training_data[0]) - 1)]
  bias = training_data[0][0]

  training_finished = False

  while not training_finished:
    training_finished = True

    for i, t_set in enumerate(training_data):
      output = - bias
      for j in range(1, len(t_set)):
        output += t_set[j] * weights[j-1]
      s = sigmoid(output)
      for j in range(1, len(t_set)):
        delta_weight = learning_rate * (target[i] - s) * (s*s * math.exp(-(output + bias))) * t_set[j]
        weights[j-1] += delta_weight

        if delta_weight > 0.0000001:
          training_finished = False
      delta_bias = -learning_rate * (target[i] - s)
      bias += delta_bias

  return weights, bias


def graph():
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  points_x_c1 = [x[1] for x in C1]
  points_x_c2 = [x[1] for x in C2]

  points_y_c1 = [x[2] for x in C1]
  points_y_c2 = [x[2] for x in C2]

  points_z_c1 = [x[3] for x in C1]
  points_z_c2 = [x[3] for x in C2]

  weights, bias = perceptron()
  l_x1, l_y1, l_z1 = get_line_points(weights, bias)

  weights, bias = adaline()
  l_x2, l_y2, l_z2 = get_line_points(weights, bias)

  ax.scatter(points_x_c1, points_y_c1, points_z_c1, c="b", marker="o", depthshade=False)
  ax.scatter(points_x_c2, points_y_c2, points_z_c2, c="r", marker="x", depthshade=False)
  ax.scatter([-1.4], [-1.5], [2], c="b", marker="o", depthshade=False)
  ax.plot(l_x1,l_y1,l_z1, label='Perceptron')
  ax.plot(l_x2,l_y2,l_z2, label='Adaline')
  ax.legend()
  plt.show()

def get_line_points(weights, bias):
  x = [i for i in range(-3, 3)]
  y = [i for i in range(-3, 3)]
  z = [((bias - weights[0]*x[i] - weights[1]*y[i]) / weights[2])  for i in range(len(x))]

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