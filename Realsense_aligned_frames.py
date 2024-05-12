import pyrealsense2 as rs
import numpy as np
import cv2

class DepthCamera:

    def __init__(self):

        """
            Initialization of the Intel RealSense camera
            It configures the RGB/color and depth frames, the resolution used is 640 in width and 480 in height.
            It also extract the depth scale and intrinsic values of the camera.          
            Finally the alignment is initiated
        """
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color,640,480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth,640,480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        
        # Intrinsic values Realsense
        self.depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # Depth scale RealSense
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # Initiate alignment
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
    def get_frame(self):

        """
            Extracts and aligns the RGBD frames from the Intel RealSense.

            Returns:
                aligned_depth_image: array with distance in mm
                aligned_color_image: array with RGB channels
                self.depth_intrin: intrinsic values of RealSense
                self.depth_scale: depth_scale of Realsense

        """
        # Initiate frames Realsense
        frames = self.pipeline.wait_for_frames()

        # Align frames
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        # Get RGB and Depth images
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())

        # Show Aligned RGBD frames
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_color_image = np.hstack((aligned_color_image, depth_colormap))
        cv2.imshow('RGB image and depth image aligned', depth_color_image)

        return aligned_depth_image, aligned_color_image,self.depth_intrin,self.depth_scale

    def release(self):

        self.pipeline.stop()

