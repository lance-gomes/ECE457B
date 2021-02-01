from math import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

C1 = [[-1, 0.8, 0.7, 1.2], [-1, -0.8,-0.7, 0.2], [-1, -0.5,0.3,-0.2], [-1, -2.8, -0.1, -2]]
C2 = [[-1, 1.2,-1.7, 2.2] , [-1, -0.8,-2, 0.5], [-1, -0.5,-2.7,-1.2], [-1, 2.8, -1.4, 2.1]]

def adaline():
  pass

def perceptron():
  pass

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

  ax.scatter(points_x_c1, points_y_c1, points_z_c1, c="r", marker="o", depthshade=False)
  ax.scatter(points_x_c2, points_y_c2, points_z_c2, c="b", marker="x", depthshade=False)
  plt.show()

def main():
  graph()


if __name__ == "__main__":
  main()