# PCAs-and-SVMs
# Usage
run `python test_script.py`

The code will then show output for PCA and SVM calculations given imported data (data_1.txt, data_2.txt).

# Function descriptions:
## PCAS:

### compute_Z(X, centering=True, scaling=False):
- This function works by manually subtracting the average of each feature set, from each feature.
- Dividing by the standard deviation is also an option.
- We do this by going through each feature and calculating the average, for said feature.
- From there, we subtract the average from each point with that feature, assuming centering is set to true.
- We repeat a similar process for standard deviation, if scaling is true.

compute_covariance_matrix(Z):
- This is a rather simple function.
- It works by multiplying the transpose of the input, Z, by Z. (Z^T * Z)

### find_pcs(COV)
- This function works by calculating the eigenvalues/eigenvectors using np.linalg.eig.
- From there, we use the "argsort()" function to get the indexes for what a sorted version of our Eigenvalues array would look like.
- We then apply the "reverse" of those indexes to both the eigenvalues and eigenvectors array, in order to sort them from greatoest to least.
- Lastly, we return the eigenvalues and eigenvectors.

### project_data(Z, PCS, L, k, var)
- This function projects the samples using k dimensions.
- We use k eigenvectors to project the data, starting from the greatest eigenvector (using eigenvalues).
- If k is 0, then we decide what k should be by using the desired variance. 
  - We calculate the variance by running the function for each possible value of k, and then using `np.var()` to get the variance from the returned data.
  - We use the value of k that results in the variance of the output being closest to target variance 'var'.
- In the end, we return the data projected by multiplying each sample by the desired PCS to use. (ZStar)

# PCA Sample output:
```
PCA Test 1:
[4. 4.]
[[0. 1.]
 [1. 0.]]
[[-1.]
 [ 1.]
 [-1.]
 [ 1.]]

PCA Test 2:
[11.55624941  0.44175059]
[[-0.6778734  -0.73517866]
 [-0.73517866  0.6778734 ]]
[[-0.82797019]
 [ 1.77758033]
 [-0.99219749]
 [-0.27421042]
 [-1.67580142]
 [-0.9129491 ]
 [ 0.09910944]
 [ 1.14457216]
 [ 0.43804614]
 [ 1.22382056]]
```

## SVMS
### distance_point_to_hyperplane(pt, w, b)
- This function computes the distance from a point, to the hyperplane, using a distance equation (https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line)
- For instance, for a line ax + by + c = 0, we treat each sample in w as corresponding to the coefficient in front of x or y. ie, w[0] is the "a" in ax. "b" is the "c" in ax + by + c = 0.
- We return the absolute value of the distance.

### compute_margin(data, w, b)
- This function computes the lowest distance from a data point, to the hyperplane.
- The function will also test to make sure the hyperplane correctly classifies each point. If it doesn't, we return a bad margin (-1) so the program knows not to consider it.

### svm_train_brute(training_data)
- This function returns the optimal w, b, and support vectors, that lead to the greatest margin.
- We do this by going through each possible combination of points, including 1 negative and 1 positive point, 2 negative and 1 positive point, and 1 negative and 2 positive points.
- When we have selected our group of points (each iteration), we will compute weight and bias for these points using compute_wb(). Then, we will compute the margin.
  - From there, we perform a quick test. If the margin is less than the distance from one of the points to the hyperplane, then we ditch this point set and move on. 
	  - This way, we're making sure we get points that are on the edge of the margin.
	- Otherwise, we see if the margin we calculated is greater than the one we had before. If it is, then we update the optimal weight, bias, support vector, and margin variables.
- Finally, we return the optimal weight, bias, and support vectors that are correlated with the most optimal margin.

### compute_wb(support_vectors)
This is a function I coded to be a helper for svm_train_brute()
- This function works by computing the weight and bias, using the second method described in the SVM video, and in the practice worksheets.
- We do this by covering each test case (1 neg + 1 pos, 2 neg + 1 pos, 1 neg + 2 pos) for support vectors.
- In first case (1 neg + 1 pos), we get the direction by subtracting both vectors.
- In the second case (2 neg + 1 pos), we used complex vector math to compute the distance.
	- I'm a little bit excited about how this works, so I'm going to share it here:
	  - We compute the "line" between the two negative points, by subtracting one point from the other.
		- We then get the vector projection of the positive point onto this line.
		- Next, we add this line onto the initial line between the two negative points.
		- Lastly, we subtract the newly added line from the positive point, to get the distance/direction from the negative point line to the positive point
	  - We're essentially doing what was shown in the video, by getting the height of the triangle.
- In the third case, we do roughly the same as the second case, except instead of subtracting the projected line from the positive point, we subtract the negative point from the projected line, so that our result is pointing towards the positive values (if that makes sense).
	

- From there, we compute the magnitude of the direction we calculated earlier, and then the unit direction vector.
	- NOTE: If our direction vector is a 0 vector, then we just return the 0 vector.
- Next, we compute gamma by halving the direction magnitude (aka the distance).
- We get the magnitude of w by dividing 1 by gamma.
- We compute W by multiplying the magnitude of w by the unit direction vector.
- Next, we compute b by plugging in w, and one of the support vector points, into the y(wx + b) = 1 equation.
- Lastly, we return w and b.

### svm_test_brute(w,b,x):
- This function works by simply running the data through the equation wx + 1 > 0.
- If it's > 0, we return 1. Otherwise, we return -1.

# SVM Sample output:
```
SVM Binary Classification Test 1:
[-1.0, 0.0] 0.0 [[ 1.  0. -1.]
 [-1.  0.  1.]]

SVM Binary Classification Test 2:
[0.0, -1.0] 0.0 [[ 0.  1. -1.]
 [ 0. -1.  1.]]

SVM Binary Classification Test 3:
[0.0, 0.5] 0.0 [[-1. -2. -1.]
 [ 3.  2.  1.]
 [ 6.  2.  1.]]

SVM Binary Classification Test 4:
[-0.5000000000000001, 0.5] 0.0 [[ 0. -2. -1.]
 [-1.  1.  1.]
 [-3. -1.  1.]]
 ```
