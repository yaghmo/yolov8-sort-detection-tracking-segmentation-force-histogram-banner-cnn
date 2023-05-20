"""
NOTE: This code is made on may 2023 by BELMIR YAGHMORACEN
Using YOLOV8 to get masks of selected objects

While the code is running, closing the window Image will end the computation

"""


from ultralytics import YOLO
import cv2
import cvzone
import os
import math
import sys
import warnings
from colorama import Fore, Style
from utils.sort import *


# **************************************************************************************** #
# drawing a bounding box of the detected objects on img that contains the clicked area x,y #
# **************************************************************************************** #
def boundingbox(results, x, y, img):
    """
    Functionality:
    Draws bounding box on a selected objects and store their masks along with their bbox coordinates.

    Input : 
    results | type = Tensor, a list containing bbox, class, conf of all the detected object.
    x, y | type = Integer, represents coordinate of the cursor on the image.
    img | type = uint8, the image.

    """

    global detection_selection, region, stacker, masked

    # for every detected object called box #
    for box, mask in zip(results[0].boxes, results[0].masks.masks):

        # coordinate of top/left corner + bot/right corner #
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # calculate the confidence #
        conf = math.ceil(box.conf*100)/100
        # box.cls is the index of class, names contains the listed class with index as a dict #
        name = results[0].names.get(int(box.cls))
        # making the bounding box to draw it #
        bbox = xyxy_to_bbox(x1, y1, x2, y2)

        # ******************************************* #
        # choose only the chosen objects to bbox them #
        if detection_selection:
            if x in [i for i in range(x1, x2+1)] and y in [i for i in range(y1, y2+1)]:
                # drawing the bbox
                cvzone.cornerRect(img, bbox, l=10)
                # writing text that contains the confidence and name of object
                cvzone.putTextRect(img, f'{conf} {name}', (max(0, x1), max(
                    15, y1-5)), thickness=1, scale=1, offset=3)

                if not stacker:
                    region = region[:len(region)-1]
                    masked = masked[:len(masked)-1]
                stacker = False

                masked.append((mask.numpy()*255).astype('uint8'))

                region = np.vstack((region, [x1, y1, x2, y2, conf]))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ******************************************* #
# def to choose either read a video or frames #
# ******************************************* #
def running_file(typ, index=0):
    """
    Functionality:
    runs a video or a sequence of images.

    Input : 
    typ | type = String, either `img` or `vid`.
    index | type = Integer, represents the head reader of the frame.

    Output:
    img | type = uint8, the resized image after it got chosen.
    i | type = Integer, the index of the head reader.

    NOTE: Pressing space allows you to stop and get that frame as starting point
    """

    global img
    if typ == 'img':
        for i in range(index, num_files):
            # reding all files inside the folder #
            img = cv2.resize(cv2.imread(
                f'{frame_path}{i}.jpg'), (width, height))

            cv2.imshow('Image', img)
            key = cv2.waitKey(1)
            if key == ord(' '):
                break
            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) == 0:
                print('The window got closed')
                break

    if typ == 'vid':
        """
        NOTE: This section is not finished yet
        """
        # cap = cv2.videocaptue

    cv2.destroyAllWindows()
    return img, i


# ***************************************** #
# def that get masks for the rest of frames #
# ***************************************** #
def get_masks(typ, index, threshold):
    """
    Functionality:
    Track the objects on the remaining frames while getting their masks and store them in a folder

    Input :
    typ | type = String, either `img` or `vid`.
    index | type = Integer, represents the head reader of the frame.


    NOTE: In this case the index is equal to the index frame the video got stopped on at the beggining
    for the object selection.
    """
    global nbr_seg
    if typ == 'img':
        for i in range(index, num_files):
            # reading all the rest of files #
            img = cv2.resize(cv2.imread(
                f'{frame_path}{i}.jpg'), (width, height))
            results = model(img, classes=[0])
            region = np.empty((0, 5))

            # small size mask #
            masked = np.zeros((nbr_seg, 480, 720))

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bbox = xyxy_to_bbox(x1, y1, x2, y2)
                conf = math.ceil(box.conf*100)/100
                region = np.vstack((region, [x1, y1, x2, y2, conf]))

            # update the tracker #
            tracked = tracker.update(region)

            for id_mask, found in enumerate(tracked):
                x1, y1, x2, y2, ID = found
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bbox = xyxy_to_bbox(x1, y1, x2, y2)

                # display only selected objects #
                if int(ID) in range(1, nbr_seg+1):
                    cvzone.cornerRect(img, bbox, l=10)
                    cvzone.putTextRect(img, f'ID: #{int(ID)}', (max(0, x1), max(
                        15, y1-5)), thickness=1, scale=1, offset=3)

                    for id_mask, box in enumerate(results[0].boxes):
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1_id, y1_id, x2_id, y2_id = found[:4]
                        x1_id, y1_id, x2_id, y2_id = int(x1_id), int(
                            y1_id), int(x2_id), int(y2_id)

                        # Threshold #
                        if sum(abs(np.array([x1, y1, x2, y2]) - np.array([x1_id, y1_id, x2_id, y2_id]))) < threshold:
                            mask_resized = cv2.resize(
                                (results[0].masks.masks[id_mask].numpy()*255).astype('uint8'), (720, 480))
                            masked[int(ID)-1] = mask_resized
            cv2.imshow('Image', img)

            '''
            # NOTE: section for masks display
            for j, mask in enumerate(masked):
                cv2.imwrite(f'{mask_path + str(i)}_mask_{j+1}.jpg', mask)
                # you can resize the masks if you wish to #
                cv2.imshow(f'Mask{j+1}', mask)
            '''

            cv2.waitKey(1)

            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) == 0:
                print('The window got closed')
                break
        print('The video has ended')
        cv2.destroyAllWindows()
    if typ == 'vid':
        """
        NOTE: This section is not finished yet
        """
        # cap = cv2.videocaptue


# ****************************** #
# redifinition of mouse_callback #
# ****************************** #
def mouse_callback(event, x, y, flags, param):
    """
    Functionality:
    Redefine the mouse call back function, either picking the selected objects or draw corners if the selection is manual.

    Input : 
    event | type = cv2.EVENT, triggers when left button is down.
    x, y | type = Integer, current coordinates of the mouse cursor.
    flags | default parameter.
    param | type = Integer, decide either the selection is on detected objects or manual selection.
    """
    global img, couple, pointed, point, X, Y, region_selection

    if event == cv2.EVENT_LBUTTONDOWN:
        # if chosen detected objects #
        img = pointed.copy()
        if param[0] == '1':
            boundingbox(param[1], x, y, img)

        # if chosen region #
        if param[0] == '2' and region_selection:
            # showing the points to create a bbox once validated #
            cvzone.putTextRect(img, f'{point}', (x, max(
                0, y-5)), thickness=1, scale=1, offset=3)
            X, Y = x, y
        cv2.imshow('Image', img)


# ****************************** #
# determine the objects to track #
# ****************************** #
def tracked_seg():
    """

    Fonctionnality:
    This methode is mainly used to determine what object / region is wanted to be tracked,
    its either by choosing to select detected object, or by manually cropping the image by choosing the corners.

    Output: 
    number_region | type = Integer, the number of selected regions

    """

    global region_selection, detection_selection, original, pointed, regioned, region, tracker, stacker, couple, point, X, Y, img
    # interactive chat #
    typ = input(
        'Type in 1 to choose a detected object or 2 to select a region \n')

    if typ == '1':
        print('Loading . . .')
        results = model(img, classes=[0])
        region_selection = False
        detection_selection = True
    while typ not in ['1', '2'] and len(typ) != 1:
        print('Please put a valid number')
        typ = input(
            'Type in 1 to choose a detected object or 2 to select a region \n')

    print('##################################################################################')
    print_warning('! Please press s if you want to validate any displayed spot on the image !')
    waiting = True
    while waiting:
        try:
            number_region = int(input(
                'Please type how many reagions you wish to detect (by default 2 - max 4- any other key will be considered as default) \n'))
            waiting = False
        except:
            input('Please type a number \n)')
    if number_region < 3:
        number_region = 2
    else:
        number_region = min(4, number_region)
    cv2.waitKey(0)

    # redisplaying the paused window for the selection #
    cv2.namedWindow("Image")
    cv2.imshow('Image', img)
    cv2.setMouseCallback("Image", mouse_callback, (typ, results))

    while True:
        """
        This section is for keyboard listner

        Main function: 
        After we selected all the wanted regions/objects

        2 arrays will be sat, 1st for the bbox of selected objets, and 2nd their masks

        NOTE: After choosing an object on the image window, pressing the key `s` will allow you to save that object to be tracked later on,
        once enough objects had been selected, the code will continue by it self.
        """

        key = cv2.waitKey(0)
        # quit #
        if key == ord('q'):
            cv2.destroyAllWindows()

        # save selected point #
        if key == ord('s'):

            # **************** #
            # manual regioning #
            # **************** #
            if region_selection:
                if len(couple) == 0:
                    couple = np.vstack((couple, [X, Y]))
                    point = 'P2'
                elif len(couple) == 1:
                    couple = np.vstack((couple, [X, Y]))
                    region = np.vstack(
                        (region, np.concatenate((couple[0], couple[1]), axis=0)))
                    couple = np.empty((0, 2), int)
                    point = 'P1'

                    # testing the Points for box #
                    img = regioned.copy()
                    x1, y1, x2, y2 = tuple(region[len(region) - 1].flatten())
                    bbox = min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)
                    cvzone.cornerRect(img, bbox, l=10)

                    # writing text that contains the confidence and name of object #
                    cvzone.putTextRect(img, f'Region {len(region)}', (max(
                        0, bbox[0]), max(15, bbox[1]-5)), thickness=1, scale=1.5, offset=3)
                    regioned = img.copy()

                    cv2.imshow('Image', img)
                if len(region) == number_region:
                    region_selection = False
                    for i in range(len(region)):
                        """
                        NOTE: This section is not finished yet
                        """

            # ************************ #
            # auto-detection selection #
            # ************************ #
            if detection_selection:
                stacker = True

                if len(region) == number_region:
                    detection_selection = False
                    cv2.destroyAllWindows()

            pointed = img.copy()
        # quit if the window is closed #
        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) == 0:
            print('The window got closed')
            if not stacker:
                sys.exit()
            break
    return number_region


# *********************** #
# tiny def for conversion #
# *********************** #
def xyxy_to_bbox(x1, y1, x2, y2):
    # convert xyxy to xywh #
    return x1, y1, x2-x1, y2-y1

# *********** #
# Warning def #
# *********** #
def print_warning(message):
    warning_text = f"{Fore.YELLOW}Warning: {message}{Style.RESET_ALL}"
    print(warning_text)



# ************************************************************************************************************************* #
if __name__ == '__main__':

    #************************ init variables **************************#
    #         dont forget to reset these following variables           #
    #******************************************************************#

    try:

        # ******************* the name of video ********************* #
        work_on = 'trim1'
        # **************** the pathe to the Frames ****************** #
        frame_path = 'Data/'+work_on+'_Frames/'

        """
        NOTE:
        Creating a sub folder for the classes, ex: A0 is a folder where a set of one pair of masks for the class `players of the 1st team` are stored
        you can name them however you want just make sure to keep the same logic and leave each pair separated in its own folder
        also dont forget to change the paths at that case on the 2nd code
        """
        mask_path = 'Data/'+work_on+'_Masks/A0/'

        # *************************** #
        # making a file for the masks #
        # *************************** #
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
    except:
        print('Check your paths please ↑')
    """
    NOTE: What's above are the paths initialization if this programme is ran independently ( Skipping the main )
    Make sure to change name for mask_path every time you create a new set of masks for the same image.

    WARNING: PLEASE MAKE SURE THE FRAME'S FOLDER EXIST AND THE NAME OF THE VIDEO PRESENT IN SAME REPERTORY WITH THAT FOLDER EXISTS TOO
    """


'''
NOTE: The folowwing is a must run, either if this module is called or being a main
'''
# ******************************************** #
# getting the number of frames from the folder #
# ******************************************** #
num_files = len([f for f in os.listdir(frame_path)
                if not f.startswith('.')]) - 1

# ***************************************************** #
#  loading the pretrained model with extralarge wights  #
# NOTE: In this case, large weights are used for yolov8 #
# ***************************************************** #
model = YOLO("Model-Weights/yolov8n-seg.pt")

# change this according your resolution preferences #
width, height = 1920, 1080

print_warning("! Press the space bar key to stop the video on the frame you want to start tracking objects from !\n Closing the window will finish the process.")
typ = input('Please type 1 if you want to run images, 2 for the video \n')
while typ not in ['1', '2'] and len(typ) != 1:
    print('Please put a valid number')
    typ = input('Type in 1 to choose to run images or 2 for the video \n')

# ****************** #
# running the frames #
# ****************** #
if typ == '1':
    img, index = running_file('img')
else:
    img, index = running_file('vid')

couple = np.empty((0, 2), int)
region = np.empty((0, 5))
masked = []

region_selection = True
detection_selection = False
stacker = False
X, Y = 0, 0
point = 'P1'
original, pointed, regioned = img.copy(), img.copy(), img.copy()

# object selection function #
nbr_seg = tracked_seg()


# ************ #
# tracker init #
# ************ #
tracker = Sort(max_age=100)
tracked = tracker.update(region)

# display masks and tracked items #
img = cv2.resize(cv2.imread(f'{frame_path}{index}.jpg'), (width, height))
for found in tracked:
    x1, y1, x2, y2, ID = found
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox = xyxy_to_bbox(x1, y1, x2, y2)
    cvzone.cornerRect(img, bbox, l=10)
    cvzone.putTextRect(img, f'ID: #{int(ID)}', (max(0, x1), max(
        15, y1-5)), thickness=1, scale=1, offset=3)

cv2.imshow('Image', img)
"""
NOTE: in this case only we display 2 concatenated masks
"""
cv2.imshow('Masks', np.concatenate((cv2.resize(masked[0], (720, 480)), cv2.resize(masked[1], (720, 480))), axis=1))
cv2.waitKey(3000)
cv2.destroyAllWindows()


# get masks from the rest of frames #
# NOTE: change this Threshold as needed it representse a range for the bbox to be accepted as tracked, the bigger 
get_masks('img', index, threshold=100)
