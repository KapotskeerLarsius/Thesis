import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from Utils import *
from Calculations import calc_surface_area
import pyrealsense2 as rs2
import os

"""
    This python script calculates surface area from the Measurements Folder.
    Fifty RGB and Depth images for each different set up/ conditions were shot with RealSense and saved in a folder
    This code uses the COCO dataset of the ultralytics library. The object used for the report was a clock.
    In line 82, cls.item()== 74 means that the YOLO model from ultralytics detected object 74 (a clock) from the COCO dataset.
    To calculate another surface area of another object that is in the COCO dataset, the number 74 has to be changed. For instance, 47 is an apple.  
    Ideal setup is to set the camera at the same height as the centre of the object, so that the object is in the middle of the frame
    RGBD image were collected at four distances to camera, two different lightings and two different angles
    To chose from which folder the surface area will be calculated, the distance, lighting and angle is asked.
    After the correct folder is entered, the script will calculate the surface area of the fifty measurements and calculate the average surface area.

    """

# Intrinsics values RealSense
intrin = rs2.intrinsics()
intrin.width = 640
intrin.height = 480
intrin.ppx = 318.808 
intrin.ppy = 245.411
intrin.fx = 615.828
intrin.fy = 616.214

# Depth scale RealSense, with depth_scale converts from mm to meters
depth_scale = 0.001

# Chose what folder(s) to use to calculate surface area
curr_distance = input("What distance value do you want to check? Values can be: 30 , 60, 90, 120: ")
lux = input("What lux value do you want to check? Values can be: 50, high: ")
angle = input("What angle value do you want to check? Values can be: angle, no_angle: ")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the Measurements folder
measurements_folder = os.path.join(current_dir,"Measurements")

# Path to folders with depth and RGB data
depth_np_folder= measurements_folder+"\\Surface_area\\"+angle+"\\distance_"+curr_distance+"\\lux_"+lux+"\\depth_np_no_seg\\"
color_folder = measurements_folder+"\\Surface_area\\"+angle+"\\distance_"+curr_distance+"\\lux_"+lux+"\\color_no_seg\\"
folder_paths = [color_folder,depth_np_folder]
check_folder(color_folder)
check_folder(depth_np_folder)
filenames = [os.listdir(folder_path) for folder_path in folder_paths]

# Initiate YOLO model
model = YOLO('yolov8n-seg.pt')

# Array for surface area
surface_area_set=[]

# Loop through files in the folders simultaneously
for color_image, depth_np in zip(*filenames):

    # Load
    color_image = cv2.imread(color_folder+color_image)
    depth_image = np.load(depth_np_folder+depth_np)

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

                # Loop through all of the segmented coordinates
                for coord in range(len(segmented_coordinates)-1): 
                    
                    # Get x and y value of current pixel
                    x = pixel_x_coords[coord]
                    y = pixel_y_coords[coord]

                    # Calculate depth for current pixel
                    depth = depth_image[x,y] * depth_scale
                    
                    # This step depends on the setup. If there are points or other objects behind the current object
                    # then it is possible to have points in the mask that do not belong the object due to wrong segmentation or alignment.
                    # This if statement is to filter the points are too far or too close and therefore not belong the detected object.
                    # It is therefore important to set the camera at almost exactly the curr_distance or vice versa
                    if (float(curr_distance)/100 - 0.07) < depth < (float(curr_distance)/100 + 0.07):

                        # Calculate the x,y,z point from the current pixel
                        point = rs2.rs2_deproject_pixel_to_point(intrin,[y,x],depth)
                        # Add the current point to array
                        points.append(point)
                        # Order of color channel is in BGR and is flipped to RGB
                        color = np.asarray(color_image[x, y][::-1]) / 255.0  
                        # Add the current color value to array
                        colors.append(color) 

                # Create point cloud of segmented object
                pcd = create_and_draw_pointcloud(points,colors,False)
                surface_area = calc_surface_area(np.array(pcd.points))
                print('savtk', surface_area)

                # Append the surface area to the array
                surface_area_set.append(surface_area)

# Calculate the mean of the set
average_surface_area = np.mean(np.array(surface_area_set))

# Print the average surface area for each voxel size
print('average_surface_area',average_surface_area)


