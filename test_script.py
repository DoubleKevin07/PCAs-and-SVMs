import numpy as np
import matplotlib.pyplot as plt
import helpers
import sys
#This should run outside of your Project3 Directory
sys.path.insert(0, 'Project3')

#TEST PCA FIRST
import pca

print("PCA Test 1:")

print("Z:")
X,Y = helpers.load_data("data_1.txt")
Z = pca.compute_Z(X)

COV = pca.compute_covariance_matrix(Z) 
print(COV)

L, PCS = pca.find_pcs(COV) 
Z_star = pca.project_data(Z, PCS, L, 1, 0)

print("L:")
print(L)

print("PCS:")
print(PCS)
print("Zstar:")
print(Z_star)

print(" ")
print("PCA Test 2:")
X,Y = helpers.load_data("data_2.txt")
Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z) 
L, PCS = pca.find_pcs(COV) 
print(COV)
print("L:")
print(L)
print("PCS:")
print(PCS)
Z_star = pca.project_data(Z, PCS, L, 0, 1)
print("Zstar:")
print(Z_star)

#TEST SVM
import binary_classification as bc

print(" ")
print("SVM Binary Classification Test 1:")
data = helpers.generate_training_data_binary(1)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)

print(" ")
print("SVM Binary Classification Test 2:")
data = helpers.generate_training_data_binary(2)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)

print(" ")
print("SVM Binary Classification Test 3:")
data = helpers.generate_training_data_binary(3)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)

print(" ")
print("SVM Binary Classification Test 4:")
data = helpers.generate_training_data_binary(4)
[w,b,S] = bc.svm_train_brute(data)
print(w,b,S)


print(" ")
print("SVM Multi-Class Classification Test:")
#622 students only. Should run and print the following errors for 422 students.
try:
  import multiclass_classification as mc
except:
  print("Could not import multiclass_classification")
[data,Y] = helpers.generate_training_data_multi(1)
try:
  [W,B] = mc.svm_train_multiclass([data,Y])
  print(W,B)
except:
  print("Could not test multiclass_classification")
