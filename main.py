'''
NOTE: This code is made on may 2023 by BELMIR YAGHMORACEN
it reads the video and split it into frames then calling the module `object_detection_to_masks.py` if its wanted
do make it happen uncomment the block comment sectiom where it imports the module `object_detection_to_masks`
'''


import numpy as np
import cv2
import os



'''
Functionality:

This is the main function of the project, starting by setting:
    - Path to dataset
    - Video name

After both folders created ( frames + masks), the 2nd programme is called which 
NOTE: make sure to add an indice to the mask_path if you willing to create many masks sets
for the same one video.
'''

# video name and path #
path = 'Data/'
video = 'trim1.mp4' 
try :
    cap = cv2.VideoCapture(path + video)
except :
    print('The video doesnt exist (°◡°♡)')


dot=len('.mp4')
# NOTE: if you change the extension of the video make sure to change #
# the value of variable dot, counting from the `.` till the end of the extention #
# in this case extention is .mp4 => dot = len('.mp4') = 4 #

# making a file for the frames
frame_path = path+video[:len(video)-dot] + '_Frames'
if not os.path.exists(frame_path):
    os.makedirs(frame_path)

# making a file for the masks
mask_path = path+video[:len(video)-dot] + '_Masks'
if not os.path.exists(mask_path):
    os.makedirs(mask_path)


count = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite(frame_path + '/' + str(count) + '.jpg', frame)
    count += 1

print('Number of frames created:', count)
cap.release()
cv2.destroyAllWindows()


#****************#
#-----BLOCK------#
#****************#
# NOTE: uncomment the section below lets the other code proceed directly from here, you can still use the module alone manually if its wanted 
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

'''
import Object_detection_to_masks
'''
