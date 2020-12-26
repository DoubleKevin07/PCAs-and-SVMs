import numpy as np
import sys


def load_data(fname):
  f = open(fname, 'r')
  ctr = 0
  y_str = ''
  for line in f:
    line = line.strip().split(';')
    if ctr == 0:
      x_str = line
    else:
      y_str = line
    ctr+=1
  f.close()
  X = []
  Y = []
  for item in x_str:
    temp = [float(x) for x in item.split(',')]
    X.append(temp)
  if len(y_str)>0:
    for item in y_str:
      temp = int(item)
      Y.append(temp)
  X = np.array(X)
  Y = np.array(Y)
  return X, Y


def generate_training_data_binary(num):
  if num == 1:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, -1]
    
  elif num == 2:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [0, i-5, 1]
      data[i+5] = [0, i+1, -1]

  elif num == 3:
    data = np.zeros((10,3))
    data[0] = [3, 2, 1]
    data[1] = [6, 2, 1]
    data[2] = [3, 6, 1]
    data[3] = [4, 4, 1]
    data[4] = [5, 4, 1]
    data[5] = [-1, -2, -1]
    data[6] = [-2, -4, -1]
    data[7] = [-3, -3, -1]
    data[8] = [-4, -2, -1]
    data[9] = [-4, -4, -1]
  elif num == 4:
    data = np.zeros((10,3))
    data[0] = [-1, 1, 1]
    data[1] = [-2, 2, 1]
    data[2] = [-3, 5, 1]
    data[3] = [-3, -1, 1]
    data[4] = [-2, 1, 1]
    data[5] = [3, -6, -1]
    data[6] = [0, -2, -1]
    data[7] = [-1, -7, -1]
    data[8] = [1, -10, -1]
    data[9] = [0, -8, -1]

  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return data

def generate_training_data_multi(num):
  if num == 1:
    data = np.zeros((20,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, 2]
      data[i+10] = [0, i-5, 3]
      data[i+15] = [0, i+1, 4]
    Y = 4

  elif num == 2:
    data = np.zeros((15,3))
    data[0] = [-5, -5, 1]
    data[1] = [-3, -2, 1]
    data[2] = [-5, -3, 1]
    data[3] = [-5, -4, 1]
    data[4] = [-2, -9, 1]
    data[5] = [0, 6, 2]
    data[6] = [-1, 3, 2]
    data[7] = [-2, 1, 2]
    data[8] = [1, 7, 2]
    data[9] = [1, 5, 2]
    data[10] = [6, 3, 3]
    data[11] = [9, 2, 3]
    data[12] = [10, 4, 3]
    data[13] = [8, 1, 3]
    data[14] = [9, 0, 3]
    Y = 3

  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return [data, Y]

