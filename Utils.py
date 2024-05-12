import open3d as o3d
import numpy as np
import os

def check_folder(path):
    """
    Checks if a given path corresponds to a directory that contains files.

    Parameters:
    - path (str): The path to the directory to be checked.

    Returns:
    - bool: True if the directory exists, is not empty, and contains files, False otherwise.
    """
    # Check if the path exists
    if not os.path.exists(path):
        print("Error: The specified path does not exist.")
        return False
    # Check if the path is a directory
    elif not os.path.isdir(path):
        print("Error: The specified path is not a directory.")
        return False
    # Check if the directory is empty
    elif not os.listdir(path):
        print("Error: The specified directory is empty.")
        return False
    # If all checks pass, return True indicating that the directory contains files
    else:
        print("The directory contains files. Proceeding...")
        return True

def create_and_draw_pointcloud(points,colors,draw):
    """
    Creates a point cloud and draws or does not draw it.
    Parameters:
        points: np.array of x,y,z points
        colors: np.array of RGB channel values
        draw: boolean. When true visualize the point cloud
    Returns:
        pcd: o3d point cloud object that contains points with RGB values.
    """
    # Create an open3d pointcloud and visualize point cloud when draw = True
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if draw:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def trim(pixels,axis,pixels_to_trim):

    """
    Trims the columns or rows of an array of pixels by to the amount of pixels_to_trim.
    This only trims the first values of the columns or rows.
    So when pixels_to_trim = 3: The first 3 elements of every column or row are deleted, not yet the last three elements.
    Parameters:
        pixels: Array of pixel coordinates of the mask segmented by the YOLO model
        axis: axis = 0 means row will be trimmed, axis = 1 means column will be trimmed
        pixels_to_trim: Amount of pixels that will be removed/trimmed from the array
    Returns:
        final_array: Trimmed array
    """
    # Extract the first column
    first_column = pixels[:, axis]

    # Dictionary to keep track of counts for each unique value
    count_dict = {}

    # List to store the indices to be removed
    indices_to_remove = []

    # Loop through the first column
    for i, value in enumerate(first_column):
        if value in count_dict:
            count_dict[value] += 1
        else:
            count_dict[value] = 1
        
        # If the count exceeds 2, add the index to indices_to_remove
        if count_dict[value] <= pixels_to_trim:
            indices_to_remove.append(i)

    # Remove the corresponding rows from the original array
    pixels_trimmed = np.delete(pixels, indices_to_remove, axis=0)

    return pixels_trimmed

def trim_array(segmented_coordinates,pixels_to_trim):

    """
    Trims an array of coordinates by removing x amount of pixels from each side of the array: Left, right, top, bottom
    Parameters:
        segmented_coordinates: Array of coordinates of the mask segmented by the YOLO model
        pixels_to_trim: Amount of pixels that will be removed/trimmed from the array
    Returns:
        final_array: Trimmed array
    """
    #Remove pixels from left and top side
    trim_row = trim(segmented_coordinates,0,pixels_to_trim)
    trim_column = trim(trim_row,1,pixels_to_trim)

    # Reverse the first column using slicing
    reversed_trim_column = trim_column[:][::-1]

    # Them remove pixels from right and bottom side
    trim_row_end = trim(reversed_trim_column,0,pixels_to_trim)
    trim_column_end = trim(trim_row_end,1,pixels_to_trim)

    #Reverse array back to original form
    final_array = trim_column_end[:][::-1]

    return final_array