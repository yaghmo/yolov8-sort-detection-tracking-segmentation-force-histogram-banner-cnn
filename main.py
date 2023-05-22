'''
NOTE: This code is made on may 2023 by BELMIR YAGHMORACEN
it reads the video and split it into frames then calling the module `object_detection_to_masks.py` if its wanted
do make it happen uncomment the block comment sectiom where it imports the module `object_detection_to_masks`
'''


import numpy as np
import cv2
from tqdm import tqdm
import os
import argparse


'''
Functionality:

This is the main function of the project, starting by setting:
    - Path to dataset
    - Video name

NOTE: If you willing to change the virable here instead of passing them through arguments, make sure to specify the exact path of the folder 
that contains the video, the name of the video along with extension.
'''


# ****************** #
#  adding arguments  #
# ****************** #
parser = argparse.ArgumentParser(description='Exctacting the frams of a given video')
parser.add_argument('video_path', nargs='?', type=str,
                    help='The path to the video')
args = parser.parse_args()


# video name and path #
if args.video_path:
    if args.video_path:
        video = os.path.basename(args.video_path)
        path = args.video_path[:-len(video)]
else:
    # ******************************************** #
    # To change when running the code with no args #
    # ******************************************** #
    path = 'Data/'
    video = 'trim1.mp4'


cap = cv2.VideoCapture(path + video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
_, dot = os.path.splitext(video)

# lenght of the extension #
dot = len(dot)

if total_frames > 0:
    # making a file for the frames
    frame_path = path+video[:len(video)-dot] + '_Frames'
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    # making a file for the masks
    mask_path = path+video[:len(video)-dot] + '_Masks'
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    print('Extracting Frames . . .')
    count = 0
    progress_bar = tqdm(total=total_frames, colour='green')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(frame_path + '/' + str(count) + '.jpg', frame)
        count += 1
        progress_bar.update(1)
    progress_bar.close()

    print('Number of frames created:', total_frames)
    cap.release()

    #****************#
    #-----BLOCK------#
    #****************#
    # NOTE: uncommenting the section below lets the other code proceed directly from here, you can still use the module alone manually if its wanted
    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    '''
    import Object_detection_to_masks
    '''
    
else:
    print("The video doesn't seem to be existing in that path (°◡ °♡) Check the name of your video along with its extension please \n")


