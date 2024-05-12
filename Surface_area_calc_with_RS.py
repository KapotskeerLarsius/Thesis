import numpy as np
from ultralytics import YOLO
from PIL import Image
from Utils import *
from Calculations import calc_surface_area
import pyrealsense2 as rs2
# Obtain frames from RealSense camera
from Realsense_aligned_frames import DepthCamera
dc = DepthCamera()

"""
    This python script calculates surface area with the Intel RealSense camera
    Ideal setup is to set the camera at the same height as the centre of the object, so that the object is in the middle of the frame

    First the current distance of the camera to the object is asked
    After that, assuming the setup and distance camera to object is correct, 
    the surface area is calculated fifty times and the average surface area of those is used for results.
"""

# Current distance/depth of the camera to the object
# This should be the distance you want to do measurements of
curr_distance = input('What is the current distance in cm(!) of the camera to the object?: ')

# Initiate YOLO model
model = YOLO('yolov8n-seg.pt')

# Array for surface area
surface_area_set=[]

for i in range(1,50): # Fifty measurements are done and after the mean of these is calculated

    # Get the depth and RGB frame, depth camera intrinsics and depth scale
    depth_image,color_image,depth_intrin,depth_scale = dc.get_frame()
    # Apply YOLO model to RGB image
    results = model(source = color_image,show=True, conf=0.6,imgsz=640)

    if results !=None: # Check if YOLO model has detected an object

        # Detection integer, needed for obtaining correct mask for each detection
        det_int = -1
 
        for cls in results[0].boxes.cls: # Go through detections

            # Create empty points and colors array
            points = []
            colors = []
            # Add one to get the next detection
            det_int+=1
        
            if cls.item() == 74:
                
                # Get the mask and coordinates for the current detected object
                result = results[0]
                masks = result.masks
                mask1 = masks[det_int]
                mask = mask1.data[0].numpy()
                mask_img = Image.fromarray(mask,"I")
                mask_array = np.array(mask_img)
                segmented_coordinates = np.argwhere(mask_array != 0)

                # Get the x and y values from the mask coordinates
                pixel_x_coords = segmented_coordinates[:,0]
                pixel_y_coords = segmented_coordinates[:,1]

                for coord in range(len(segmented_coordinates)-1): # Loop through all the pixels of the mask
                    
                    # Get x and y value of current pixel
                    x = pixel_x_coords[coord]
                    y = pixel_y_coords[coord]

                    # Calculate depth for current pixel
                    depth = depth_image[x,y] * depth_scale
                    
                    # This step depends on the setup. If there are points or other objects behind the current object
                    # then it is possible to have points in the mask that do not belong the object due to wrong segmentation or alignment.
                    # This if statement is to filter the points are too far or too close and therefore not belong the detected object.
                    # It is therefore important to set the camera at almost exactly the curr_distance or vice versa
                    if (float(curr_distance)/100 - 0.05) < depth < (float(curr_distance)/100 + 0.05):

                        # Calculate the x,y,z point from the current pixel
                        point = rs2.rs2_deproject_pixel_to_point(depth_intrin,[y,x],depth)
                        # Add the current point to array
                        points.append(point)
                        # Order of color channel is in BGR and is flipped to RGB
                        color = np.asarray(color_image[x, y][::-1]) / 255.0  
                        # Add the current color value to array
                        colors.append(color) 

                # Create point cloud of segmented object
                pcd = create_and_draw_pointcloud(points,colors,False)
                # Calculate surface area using vtk library. In calc_surface_area the triangulated mesh is visualized.
                surface_area = calc_surface_area(np.array(pcd.points))
                print('savtk', surface_area)

                # When surface area without voxel size is not withing 100-300 cm^2 (Real value is 179cm^2) something is wrong with the calculation or images.
                # Surface area is too high when wrong points are segmented
                # Usually surface area is too low or zero when the distance to the object is not correct
                if 0.01 < surface_area < 0.03:  

                    # Append the surface areas to the arrays
                    surface_area_set.append(surface_area)

# Calculate the mean of the set
average_surface_area = np.mean(np.array(surface_area_set))

# Print the average surface area for each voxel size
print('average_surface_area',average_surface_area)


