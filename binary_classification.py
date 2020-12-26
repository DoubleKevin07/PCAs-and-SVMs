import  numpy as np

def distance_point_to_hyperplane(pt, w, b):

    # Throw these out
    if(np.linalg.norm(w) == 0):
        return 0
        
    dist = abs(pt[0] * w[0] + pt[1] * w[1] + b) / np.linalg.norm(w)
    return dist


# Find minimal point to hyperplane
def compute_margin(data, w, b):
    dist = -1
    
    # print("LEN OF W: " + str(len(w)))
    for sample in data:
        point = sample[0:2]

        d = distance_point_to_hyperplane(point, w, b)
        if dist == -1 or d < dist:
            dist = d
        
        # We need to make sure all the points are correctly classified by these weights.
        if svm_test_brute(w, b, sample) != sample[2]:
            return -1 # Otherwise, margin not valid.

    #return 1 / np.linalg.norm(w)
    # print(dist)

    # This distance should theoretically be equal to the distance between the support vectors and the hyperplane.
    return dist

def svm_train_brute(training_data):
    # Store optimal weights.
    optimal_weights = []
    optimal_bias = 99
    optimal_support_vecs = []
    optimal_margin = 0

    # Cycle through all data points, Must be a set of 2 negative points and 1 positive point, 1 negative point and 2 positive points, or 1 negative point and 1 positive point.

    # All 1-negative point, 1-positive point possibilities:
    for point_a in training_data:
        if point_a[2] == -1: # If point is negative...
           
            # Now, cycle through the rest.

            # 1 negative point, 1 positive point
            for point_b in training_data:

                # If it's the same point, or if the label is negative, continue, because we're looking for a positive point.
                if (point_a == point_b).all() or point_b[2] == -1:
                    continue

                # print("POINT A: " + str(point_a))
                # Compute hyperplane
                # ===================
                w,b = compute_wb(np.array([point_a, point_b]))


                # Compute margin
                # ====================
                margin = compute_margin(training_data, w, b)

                dist_a = distance_point_to_hyperplane(point_a, w, b)
                
                if margin != dist_a:
                    continue

                # See if optimal. If so, update!
                if margin >= optimal_margin:
                    optimal_margin = margin

                    # Add all the weights
                    optimal_weights = []
                    for weight in w:
                        optimal_weights.append(weight)

                    optimal_bias = b

                    optimal_support_vecs = np.array([point_a, point_b])

            # 1 negative point, 2 positive points
            for point_b in training_data:
                if (point_a == point_b).all() or point_b[2] == -1:
                    continue

                # Grabbed one positive point.

                for point_c in training_data:
                    if (point_c == point_a).all()  or (point_c == point_b).all()  or point_c[2] == -1:
                        continue          

                    # Grabbed another positive point.          

                    # Compute hyperplane
                    # ===================

                    w, b = compute_wb(np.array([point_a, point_b, point_c]))

                    # Compute margin
                    # ====================
                    
                    margin = compute_margin(training_data, w, b)

                    # print("MARIGN: " + str(margin) + " " + "DIST_A: " + str(distance_point_to_hyperplane(point_a,w,b)))
                    
                    dist_a = distance_point_to_hyperplane(point_a, w, b)
                    # dist_b = distance_point_to_hyperplane(point_b, w, b)
                    # dist_c = distance_point_to_hyperplane(point_c, w, b)
                    # print(" [DIST_A] [DIST_B] [DIST_C] [MARGIN] " + str(dist_a) + " " + str(dist_b) + " " + str(dist_c) + " " + str(margin))
                    
                    if margin != dist_a:
                        continue
                    # See if optimal. If so, update!
                    if margin > optimal_margin:
                        optimal_margin = margin

                        # Add all the weights
                        optimal_weights = []
                        for weight in w:
                            optimal_weights.append(weight)

                        optimal_bias = b

                        optimal_support_vecs = np.array([point_a, point_b, point_c])

            # 2 negative points, 1 positive points
            for point_b in training_data:
                if (point_a == point_b).all() or point_b[2] == 1:
                    continue

                # Grabbed another negative point.

                for point_c in training_data:
                    if (point_c == point_a).all()  or (point_c == point_b).all()  or point_c[2] == -1:
                        continue          

                    # Grabbed a positive point.          

                    # Compute hyperplane
                    # ===================

                    w, b = compute_wb(np.array([point_a, point_b, point_c]))

                    # Compute margin
                    # ====================
                    margin = compute_margin(training_data, w, b)

                    dist_a = distance_point_to_hyperplane(point_a, w, b)
                    
                    if margin != dist_a:
                        continue
                    # See if optimal. If so, update!
                    if margin > optimal_margin:
                        optimal_margin = margin

                        # Add all the weights
                        optimal_weights = []
                        for weight in w:
                            optimal_weights.append(weight)

                        optimal_bias = b

                        optimal_support_vecs = np.array([point_a, point_b, point_c])


    # print(svm_test_brute(optimal_weights, optimal_bias, np.array([3, 2])))
    return optimal_weights, optimal_bias, optimal_support_vecs

def compute_wb(support_vectors):

    """
    Ignore this - this is my attempt at solving with system of linear equations.

    left_equations = []
    right_equations = []
    for vec in support_vectors:
        left = []
        left.append(vec[2] * vec[0] )
        left.append(vec[2] * vec[1] )
        left.append(vec[2] * 1)
        left_equations.append(left)
        right_equations.append(1)

    if len(left_equations) == 2:
        left_equations.append([0,0,0])
        right_equations.append(0)

    print(left_equations)
    print(right_equations)
    answers = np.linalg.solve(left_equations,right_equations)
    return np.array([answers[0:2]]), answers[2]
    """
    # print("\n")
    # print("Computing w, b ==============")
    # 3 possibilities.

    direction = []
    dist = []

    # 1 negative, 1 positive.
    if len(support_vectors) == 2:
        
        point_a = support_vectors[0][0:2]
        point_b = support_vectors[1][0:2]

        # print("Point a: " + str(point_a))
        # get direction:
        direction = np.subtract(point_b, point_a)

        # Compute distance
        # print("Case 1.")

    elif len(support_vectors) == 3:
        point_a = support_vectors[0][0:2]
        point_b = support_vectors[1][0:2]
        point_c = support_vectors[2][0:2]

        # 2 negative, 1 positive
        if support_vectors[1][2] == -1:
            
            # Compute distance. C is positive point.

            # Find the vector from point 
            # direction = np.cross(point_b-point_a,point_c-point_a)/np.linalg.norm(point_b - point_a)

            # This gets the direction from positive to negative, so we need to flip it.
            # direction = np.flip(direction)
            line = np.subtract(point_a, point_b)
            c_projection = line * np.dot(point_c - point_b, line) / (np.linalg.norm(line)**2)
            c_on_b = point_b + c_projection
            # # print("PROJECTION: " + str(c_projection))
            direction = point_c - c_on_b
            # # print("Case 2.")

        # 1 negative, 2 positive
        if support_vectors[1][2] == 1:
            
            # Compute distance. B and C are positive.
            # print("Point a: " + str(point_a))
            # print("Point b: " + str(point_b))
            # print("Point c: " + str(point_c))

            # Find the vector from point 
            # direction = np.cross(point_c-point_b,point_b-point_a)/np.linalg.norm(point_c - point_b)

            # We can calculate this by projecting A onto BC, then subbing that from A.
            line = np.subtract(point_b, point_c)
            a_projection = line * np.dot(point_a - point_c, line) / (np.linalg.norm(line)**2)
            a_on_c = point_c + a_projection
            direction = a_on_c - point_a # point_a is negative, so we need to point towards the line (positive)

            # print("DIRECTION: " + str(direction))
            # #print("Case 3.")

    if (direction == 0).all():
        # print("Got a 0 direction.")
        # print("vectors: " + str(support_vectors))
        return [0, 0],0
    # Compute w
    # print("Direction: " + str(direction))
    direction_mag = np.linalg.norm(direction)
    direction_unit = direction / direction_mag
    
    gamma = direction_mag / 2 # Gamma is distance divided by 2.

    mag_w = 1 / gamma

    w = mag_w * direction_unit #np.multiply(mag_w, direction_unit)

    # Solve for b
    # y(w dot x + b) = 1
    # (1/y) = w dot x + b
    # (1/y) - w dot x = b
    # We will use the first point.
    y = support_vectors[0][2]
    x = support_vectors[0][0:2]
    b = (1/y) - np.dot(w,x)

    # print("==========================")
    # print("\n")
    
    return w, b


def svm_test_brute(w,b,x):

    # y(wx + b) = 1
    # 1 / (w dot x + b) = y

    if np.dot(w,x[0:2]) + b > 0:
        return 1
    else:
        return -1