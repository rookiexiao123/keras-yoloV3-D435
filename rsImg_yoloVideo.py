
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
'''
            # Get the depth frame's dimensions
            width = depth_frame.get_width()
            width = int(width / 2)
            height = depth_frame.get_height()
            height = int(height / 2)

            # Query the distance from the camera to the object in the center of the image
            #相机到图像中心的距离
            dist_to_center = depth_frame.get_distance(width, height)
            print(dist_to_center)
'''
def detect_img(yolo):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipe_profile = pipeline.start(config)
    
    align_to = rs.stream.color
    align = rs.align(align_to)	

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
	
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            img = Image.fromarray(color_image)
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', images)
              
            detections = list()
            r_image = yolo.detect_imageout(img, single_image=False, output=detections)
            #r_image = yolo.detect_image(img)
            #r_image.show()
            print(detections)
            #center s: 0.97 l: 294 t: 333 r: 314 b: 376
            #pallet s: 0.98 l: 129 t: 332 r: 479 b: 378
            #4个角
            #(left, top), (right, bottom)
            #(left, bottom), (right, bottom) 
            if detections:
                #print(len(detections)) 打印有几个检测框
                for i in range(len(detections)):
                    #检测出是center
                    if detections[i][0] == 'c':
                        score = detections[i][10] + detections[i][11] + detections[i][12] + detections[i][13]
                        print(score)
                        for j in range(len(detections[i])):
                            if detections[i][j] == 'l' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     centerLeft = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     centerLeft = detections[i][j+3] + detections[i][j+4]
                                elif detections[i][j+6] == ' ':
                                     centerLeft = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                centerLeft = int(centerLeft)
                                print('centerLeft: ', centerLeft)

                            elif detections[i][j] == 't' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     centerTop = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     centerTop = detections[i][j+3] + detections[i][j+4]
                                elif detections[i][j+6] == ' ':
                                     centerTop = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                centerTop = int(centerTop)
                                print('centerTop: ', centerTop)

                            elif detections[i][j] == 'r' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     centerRight = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     centerRight = detections[i][j+3] + detections[i][j+4]
                                elif detections[i][j+6] == ' ':
                                     centerRight = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                centerRight = int(centerRight)
                                print('centerRight: ', centerRight)

                            elif detections[i][j] == 'b' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     centerBottom = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     centerBottom = detections[i][j+3] + detections[i][j+4]
                                else:
                                     centerBottom = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                centerBottom = int(centerBottom)
                                print('centerBottom: ', centerBottom)
                    #检测出是pallet                        
                    if detections[i][0] == 'p':
                        score = detections[i][10] + detections[i][11] + detections[i][12] + detections[i][13]
                        print(score)
                        for j in range(len(detections[i])):
                            if detections[i][j] == 'l' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     palletLeft = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     palletLeft = detections[i][j+3] + detections[i][j+4]
                                elif detections[i][j+6] == ' ':
                                     palletLeft = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                palletLeft = int(palletLeft)
                                print('palletLeft: ', palletLeft)

                            elif detections[i][j] == 't' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     palletTop = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     palletTop = detections[i][j+3] + detections[i][j+4]
                                elif detections[i][j+6] == ' ':
                                     palletTop = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                palletTop = int(palletTop)
                                print('palletTop: ', palletTop)

                            elif detections[i][j] == 'r' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     palletRight = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     palletRight = detections[i][j+3] + detections[i][j+4]
                                elif detections[i][j+6] == ' ':
                                     palletRight = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                palletRight = int(palletRight)
                                print('palletRight: ', palletRight)

                            elif detections[i][j] == 'b' and detections[i][j+1] == ':':
                                if detections[i][j+4] == ' ':
                                     palletBottom = detections[i][j+3]
                                elif detections[i][j+5] == ' ':
                                     palletBottom = detections[i][j+3] + detections[i][j+4]
                                else:
                                     palletBottom = detections[i][j+3] + detections[i][j+4] + detections[i][j+5]
                                palletBottom = int(palletBottom)
                                print('palletBottom: ', palletBottom)
                        
                        #pallet左下角 离相机的距离
                        width1 = palletLeft
                        height1 = palletBottom
                        if width1 == 640:
                            width1 = 639
                        if height1 == 480:
                            height1 = 479

                        dist_to_center1 = depth_frame.get_distance(width1, height1)
                        print('distance1: ', dist_to_center1)

                        #pallet右下角 离相机的距离
                        width2 = palletRight
                        height2 = palletBottom
                        if width2 == 640:
                            width2 = 639
                        if height2 == 480:
                            height2 = 479

                        dist_to_center2 = depth_frame.get_distance(width2, height2)
                        print('distance2: ', dist_to_center2)

                        #pallet左上角 离相机的距离
                        width3 = palletLeft
                        height3 = palletTop
                        if width3 == 640:
                            width3 = 639
                        if height3 == 480:
                            height3 = 479

                        dist_to_center3 = depth_frame.get_distance(width3, height3)
                        print('distance3: ', dist_to_center3)

                        #pallet右上角 离相机的距离
                        width4 = palletRight
                        height4 = palletTop
                        if width4 == 640:
                            width4 = 639
                        if height4 == 480:
                            height4 = 479

                        dist_to_center4 = depth_frame.get_distance(width4, height4)
                        print('distance4: ', dist_to_center4)

                        distance_one = abs(dist_to_center1 - dist_to_center2)
                        #distance_one = distance_one / () + ()
                        if distance_one < 1 :
                            corner1 = math.asin(distance_one)
                            print('corner1: ', corner1)

                        distance_two = abs(dist_to_center3 - dist_to_center4)
                        if distance_two < 1 :
                            corner2 = math.asin(distance_two)
                            print('corner2: ', corner2)
                     
            #time.sleep(10) 
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        yolo.close_session()
        #Stop streaming
        pipeline.stop()
        

FLAGS = None

if __name__ == '__main__':

    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
