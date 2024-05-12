import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian

def distance_metric(transform_params, points_A, points_B):
    
    # Extract rotation and translation parameters
    rotation_params = transform_params[:3]
    translation_params = transform_params[3:]
    
    # Create rotation matrix with rotation paramaters
    R = rotation_matrix(*rotation_params)

    # Apply transformation to points_A
    transformed_points_A = np.dot(points_A,R.T) +translation_params

    # Set initial total distance to zero
    total_distance = 0 
    # Loop through the reference points
    for i in range(len(transformed_points_A)):
        # Calculate distance between reference point
        point_distance = np.linalg.norm(transformed_points_A[i] - points_B[i])
        # Add distance to the cumulative distances
        total_distance += point_distance
        
    return total_distance


def rotation_matrix(rx, ry, rz):
    
    # Convert angles to radians
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)

    # Compute rotation matrix
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz.dot(Ry.dot(Rx))  

# Function that returns Jacobian matrix 
def fun_der(x, points_A,points_B):
    return Jacobian(lambda x: distance_metric(x, points_A,points_B))(x).ravel()

def fun_hess(x, points_A,points_B):
    return Hessian(lambda x: distance_metric(x, points_A,points_B))(x)

def find_transformation_first_two_points(points_A, points_B):

    # Initial guess for rotation and translation parameters
    initial_guess = np.zeros(6)
    # Minimize the distance metric
    result = minimize(distance_metric, initial_guess, args=(points_A, points_B), method='BFGS',jac=fun_der)
    # Extract optimal transformation parameters
    optimal_params = result.x
    optimal_rotation_params = optimal_params[:3]
    optimal_translation_params = optimal_params[3:]
    distances = distance_metric(optimal_params,points_A,points_B)
    # Construct rotation matrix
    optimal_rotation_matrix = rotation_matrix(*optimal_rotation_params)

    return optimal_rotation_matrix, optimal_translation_params,distances

def rotate_point(angle,point_trans,point_fix, axis_point1, axis_point2 ):
    """
    Rotate a point cloud around an axis defined by two fixed points.

    Parameters:
        point_cloud (np.array): Numpy array of shape (N, 3) representing the point cloud.
        axis_point1 (tuple): Tuple (x, y, z) representing the first fixed point.
        axis_point2 (tuple): Tuple (x, y, z) representing the second fixed point.
        angle (float): Angle in degrees for rotation.

    Returns:
        np.array: Rotated point cloud.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Create rotation matrix
    u = np.array(axis_point2) - np.array(axis_point1)
    u = u.astype(float)  # Ensure u is of float type
    u /= np.linalg.norm(u)
    ux, uy, uz = u
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

    # Translate to the origin
    translated_point = point_trans - np.array(axis_point1)

    # Rotate
    rotated_point = np.dot(translated_point, rotation_matrix.T)
    rotated_point += np.array(axis_point1)
    # Translate back
    distance = np.linalg.norm(rotated_point-point_fix)
    print('angle',angle)
    print('distance',distance)
    return distance

def find_transformation_third_point(point_trans,point_fix, axis_point1, axis_point2):

    # Initial guess for rotation and translation parameters
    initial_guess = 0

    # Find the optimal transformation of the angle by minimizing the distance between point_trans and point_fix
    result = minimize(rotate_point, initial_guess, args=(point_trans,point_fix, axis_point1, axis_point2), method='BFGS')

    # Extract angle from the result
    opt_angle = result.x
    
    return opt_angle

def rotate_point_cloud(point_cloud, axis_point1, axis_point2, angle ):
    """
    Rotate a point cloud around an axis defined by two fixed points.

    Parameters:
        point_cloud (np.array): Numpy array of shape (N, 3) representing the point cloud.
        axis_point1 (tuple): Tuple (x, y, z) representing the first fixed point.
        axis_point2 (tuple): Tuple (x, y, z) representing the second fixed point.
        angle (float): Angle in degrees for rotation.

    Returns:
        np.array: Rotated point cloud.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Create rotation matrix
    u = np.array(axis_point2) - np.array(axis_point1)
    u = u.astype(float)  # Ensure u is of float type
    u /= np.linalg.norm(u)
    ux, uy, uz = u
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

    # Translate to the origin
    translated_point_cloud = point_cloud - np.array(axis_point1)

    # Rotate
    rotated_point_cloud = np.dot(translated_point_cloud, rotation_matrix.T)
    rotated_point_cloud += np.array(axis_point1)
    
    return rotated_point_cloud
