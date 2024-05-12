import numpy as np
import open3d as o3d
from ultralytics import YOLO
from PIL import Image
from Utils import *
from Utils_reference_points import *
from Calculations import calculate_point_cloud_volume
import pyrealsense2 as rs2

# Obtain frames from RealSense camera
from Realsense_aligned_frames import DepthCamera
dc = DepthCamera()

# Initiate YOLO model
model = YOLO('yolov8n-seg.pt')

def make_pcds(): 

    """
    Creates point clouds of three objects and the apple and creates an array with the reference points.
    RGBD images were taken with RealSense.

    Returns:
        pcd_apple: Point cloud of the apple
        mean_points: Array with reference points
    
    """
    detections = False   
    # While all reference objects are not correctly detected and segmented, stay in loop
    while not detections:

        # Set the reference objects to False (= not Detected)
        orange = False
        stop = False
        clock = False
        apple = False
        # Get the depth and RGB frame, depth camera intrinsics and depth scale
        depth_image,color_image,depth_intrin,depth_scale = dc.get_frame()
        # Apply YOLO model to RGB image
        results = model(source = color_image,show=True, conf=0.1,imgsz=640)

        if results !=None: # Check if YOLO model has detected an object
            
            for cls in results[0].boxes.cls: # Loop through all detections
                # Check if all items are detected
                if cls.item() == 11: # A stop_sign is detected
                    stop = True
                if cls.item() == 47: # An apple is detected
                    apple = True
                if cls.item() == 49: # An orange is detected
                    orange = True
                if cls.item() == 74: # A clock is detected
                    clock = True
                
            if orange and apple and stop and clock:
                # If all items are detected, check is segmentation is done okay. 
                # It is possible that the YOLO model detects and segments half the object.
                if input('Segmentation ok? : y/n') == 'y':
                    detections = True

    if results !=None: # Check if YOLO model has detected an object

        # Detection integer, needed for obtaining correct mask for each detection
        det_int = -1
 
        for cls in results[0].boxes.cls: # Loop through detections

            # Create empty points and colors array
            points = []
            colors = []
            # Add one to get the next detection
            det_int+=1

            if cls.item() == 11 or cls.item() == 47  or cls.item() == 49 or cls.item() == 74:
                
                # Get the mask and coordinates for the current detected object
                result = results[0]
                masks = result.masks
                mask1 = masks[det_int]
                mask = mask1.data[0].numpy()
                mask_img = Image.fromarray(mask,"I")
                mask_array = np.array(mask_img)
                segmented_coordinates = np.argwhere(mask_array != 0)

                #Trim of the noise by removing 3 pixels from each side
                pixels_to_trim = 3
                trimmed_coords = trim_array(segmented_coordinates,pixels_to_trim)

                # Get the x and y values from the trimmed mask coordinates
                pixel_x_coords = trimmed_coords[:,0]
                pixel_y_coords = trimmed_coords[:,1]

                for coord in range(len(trimmed_coords)-1):

                    # Get x and y value of current pixel
                    x = pixel_x_coords[coord]
                    y = pixel_y_coords[coord]

                    # Calculate depth for current pixel
                    depth = depth_image[x,y] * depth_scale

                    # Only calculate x,y,z point for depth smaller than 2 and larger than 0.1
                    # Values higher than 2 meter are not needed and creates less computing

                    if 0.1 < depth < 2:
                        
                        # Calculate the x,y,z point from the current pixel
                        point = rs2.rs2_deproject_pixel_to_point(depth_intrin,[y,x],depth)
                        # Add the current point to array
                        points.append(point)
                        # Order of color channel is in BGR and is flipped to RGB
                        color = np.asarray(color_image[x, y][::-1]) / 255.0  
                        # Add the current color value to array
                        colors.append(color) 
                
                # Get the reference points and point clouds for all objects
                if cls.item() == 11: # Stop_sign
                    ref_point_A1 = np.mean(points, axis=0)
                    pcd_A1 = create_and_draw_pointcloud(points,colors,True)
                if cls.item() == 47: 
                    ref_point_apple= np.mean(points, axis=0)
                    pcd_apple = create_and_draw_pointcloud(points,colors,True)
                if cls.item() == 49: # orange
                    ref_point_C1 = np.mean(points, axis=0)
                    pcd_C1 = create_and_draw_pointcloud(points,colors,True)
                if cls.item() == 74: # Clock
                    ref_point_B1 = np.mean(points, axis=0)
                    pcd_B1 = create_and_draw_pointcloud(points,colors,True)
                
    # Stack the reference points in one array
    mean_points = np.vstack((ref_point_A1,ref_point_B1,ref_point_C1))
    
    # OPTIONAL: Visualize the point clouds of all segmentations
    o3d.visualization.draw_geometries([pcd_apple,pcd_A1,pcd_B1,pcd_C1])

    return pcd_apple,pcd_A1,pcd_B1,pcd_C1,mean_points

def merge_pcds(pcd_apple1,pcd_A1,pcd_B1,pcd_C1,org_mean_points):

    """
    Merges point clouds using reference points.
    Parameters:
        org_mean_points: reference points of the first/original point cloud
    Returns:
        pcd_transformed: Transformed point cloud of apple
    
    """
    # Make new point clouds of the apple and three objects and make array with reference points
    pcd_apple_trans,pcd_A2,pcd_B2,pcd_C2,mean_points = make_pcds()
    points = np.asarray(pcd_apple_trans.points)
    curr_points = mean_points

    # Initial sum_distance is big value, this sum_distances will be close to zero when reference points are merged correctly
    sum_distances = 1000
  
    # This while loop is made, because the optimal transformation value for the first two reference points is not always guaranteed firs try.
    # When the sum of distances for the two reference points is equal to the previous sum of distances for the two reference points
    # five times (i=5) in a row the optimal transformation value is found. Therefore initial value for i=0
    i=0
    while i<5:
        # Update prev_sum_distances
        prev_sum_distances = sum_distances
        # Find optimal rotation matrix and translation vector and the sum_distances for transformation of first two reference points
        optimal_rotation_matrix, optimal_translation_params,sum_distances = find_transformation_first_two_points(curr_points[1:],org_mean_points[1:])
        # Update the current two reference points to curr transformation
        curr_points = np.dot(curr_points, optimal_rotation_matrix.T) + optimal_translation_params
   
        print('distances two points',sum_distances)
        # If there is no improvement in sum_distances and thus transformation value add 1 to i
        if (sum_distances-0.0001)<prev_sum_distances < (sum_distances+0.0001):
            i+= 1
        else:
            i=0

    # Find angle for the transformation of the third reference point
    angle = find_transformation_third_point(curr_points[0],org_mean_points[0],org_mean_points[1],org_mean_points[2])
    # Rotate the point cloud by the angle found around a fixed axis between the first two reference points
    rot_pcd = rotate_point_cloud(curr_points,org_mean_points[1],org_mean_points[2],angle)
    rot_pcd = np.squeeze(rot_pcd)

    # Create an Open3d point cloud
    pcd_transformed = create_and_draw_pointcloud(rot_pcd,pcd_apple_trans.colors,False)

    return pcd_transformed

if __name__ == "__main__":
    
    # Create an empty open3d point cloud object
    merged_pcd = o3d.geometry.PointCloud()
    # Make new point clouds of the apple and three objects and make array with reference points
    pcd_apple1,pcd_A1,pcd_B1,pcd_C1,org_mean_points = make_pcds() 
    # Add first point cloud of apple to merged_pcd
    merged_pcd+=pcd_apple1
    
    while not finished:
        
        # Use the reference point merging method to create a transformed point cloud
        pcd_ac = merge_pcds(pcd_apple1,pcd_A1,pcd_B1,pcd_C1,org_mean_points)

        # Visualize the point cloud to see if the merging was succesfull
        o3d.visualization.draw_geometries([pcd_apple1,pcd_ac])

        # If merging was successful press y to add curr pcd to total
        merge_okay = input("Is the merge ok? y/n (n to cancel current merge, y to add current pcd to total pcds)")
        if merge_okay == "y":
            merged_pcd+=pcd_ac

        # Visualize the total merged_pcd
        o3d.visualization.draw_geometries([merged_pcd])

        # When the merged_pcd looks good and a from a few angles RGBD images are taken, 
        # you can stop the program here and calculate volume
        done = input("Finished capturing RGBD images? y/n (n to continue, y to stop and calculate volume)")
        if done == "y":
            finished = True

    # Calculate volume of the merged_pcd
    volume = calculate_point_cloud_volume(np.asanyarray(merged_pcd.points))
    print('Volume of merged Point cloud = ',volume)





