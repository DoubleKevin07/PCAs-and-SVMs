import numpy as np

def compute_Z(X, centering=True, scaling=False):

    # Base value for Z
    Z = []
    for sample in X:
        Z.append(sample)

    # Globals we'll use for calculations.
    feature_count = len(X[0])
    sample_count = len(X)

    # Calculate averages if we need to.
    averages = []
    if centering is True or scaling is True:

        # Initialize each average to 0.
        
        for feature in range(feature_count):
            averages.append(0)

        # Go through feature list, and calculate averages.
        for row in X:

            # For every row, go through its data.
            for num in range(feature_count):
                averages[num] += row[num]
            
        # Calculate average for each feature.
        for num in range(feature_count):
            averages[num] = averages[num] / sample_count

    # If we want to do centering
    if centering is True:

        # Center values by subtracting the average from each one.
        row_index = 0


        for row in Z:
            col_index = 0
            for col in row:
                Z[row_index][col_index] -= averages[col_index]
                col_index += 1
            
            row_index += 1

    if scaling is True:

        # Compute the standard deviation.
        standard_deviations = []
        for feature in range(feature_count):
            standard_deviations.append(0)

        # Iterate through each row and for each feature, add the sum of the val of the feature - the average, and then square the result.
        # (Numerator)
        for row in X:
            for num in range(feature_count):
                standard_deviations[num] += (row[num] - averages[num]) ** 2 
        
        # Divide by N (Denominator), then square root.
        # RFE: Maybe we should divide by N - 1.
        for num in range(feature_count):
            standard_deviations[num] = (standard_deviations[num] / (sample_count - 1) ) ** 0.5

        # Now put this all in Z.
        row_index = 0
        for row in Z:
            col_index = 0
            for col in row:
                Z[row_index][col_index] /= standard_deviations[col_index]
                col_index += 1
            row_index += 1
    
    return Z

def compute_covariance_matrix(Z):
    ZT = np.transpose(Z)
    return np.matmul(ZT, Z)

def find_pcs(COV):
    eigenvalues, eigenvectors = np.linalg.eig(COV) # Compute eigenvalues, eigenvectors.
    # Get data for how we would sort the eigenvalues.
    indexes = eigenvalues.argsort()
    eigenvectors = np.transpose(eigenvectors) # np.linalg.eig returns eigenvectors in rows. We want to return it as columns.
    eigenvalues = eigenvalues[indexes[::-1]] # Sort greatest to least.
    eigenvectors = eigenvectors[indexes[::-1]]

    return eigenvalues, eigenvectors # Return them.

def project_data(Z, PCS, L, k, var):
    pcas_to_use = []
    
    # use K
    if var == 0:
        for num in range(k):
            pcas_to_use.append(PCS[num])
    else:
        # Otherwise, determine the k that would return the desired variance.
        num_dimensions = len(Z[0])
        # print("num dimeions = " + str(num_dimensions))
        best_k = -1
        best_dist = 9999
        target = var
        for dimension in range(1,num_dimensions+1):
            # print("test k = " + str(dimension))
            variance = np.var(project_data(Z, PCS, L, dimension, 0))
            dist = abs(variance - target)
            if dist < best_dist:
                best_k = dimension
                best_dist = dist
            
        
        return project_data(Z, PCS, L, best_k, 0)

    # We don't need to transpose our PCS since we already did. At this point, each row is an eigenvector.

    Zstar = []
    # Go through all the points.
    for sample in Z:
        Zstar.append(np.matmul(pcas_to_use, sample))
    return np.array(Zstar)