import cv2
import utils.fhistograms as fhist
import numpy as np
from tqdm import tqdm
import os
# import utils.fhistograms as fhist 


def count(folder_path, label=None, names=False):
    # *************************************************************************** #
    # this function is used to check how many items are inside a given folder are #
    # *************************************************************************** #
    '''
    Input :
    folder_path | type = String : oath to a folder
    label | represents the prefix of a class folder. If its kept on None, it looks for the masks folders

    Output :

    '''
    item = []
    files = []
    if names:
        for folder_name in os.listdir(folder_path):
            files.append(int(folder_name.split('_')[0]))

        # ****************** #
        #!! sort the files !!#
        # ****************** #
        files = np.sort(files)
        # re-ordering the masks files
        for file_name in files[::2]:
            item.append(str(file_name)+'_mask_1.jpg')
            item.append(str(file_name)+'_mask_2.jpg')
        return item


    for folder_name in os.listdir(folder_path):
        if label == None:
            if folder_name.endswith('_Masks'):
                item.append(folder_name)
        else:
            if folder_name.startswith(label):
                item.append(folder_name)
    return item


def threshold(img):
    # ***************** #
    #   Binarization    #
    # ***************** #
    img[img>110] = 255
    img[img<=110] = 0
    return img


def banner(masks, path, degree=180, ftype = 1.0):

    # *********************************************** #
    #       making an array for the histograms        #
    # NOTE: the input degree is needed for fhistogram #
    # *********************************************** #
    banner = np.empty((0, degree),dtype=np.int16)

    # ***************** #
    # loading the masks #
    # ***************** #
    
    for i in tqdm(range(0, len(masks), 2), total=len(masks)//2):

        mask_1 = cv2.imread(path + masks[i],0)
        mask_1 = threshold(mask_1)
    
        mask_2 = cv2.imread(path + masks[i+1],0)
        mask_2 = threshold(mask_2)

        banner = np.vstack((banner, fhist.fhistogram(a = mask_1, b = mask_2, n_dirs = degree, force_type=ftype)))
    return banner



#*************************************************************************************************************************
if __name__ == '__main__':

    # *************** #
    # initializations #
    # *************** #
    folder = 'Data'
    unique_file_name = 0
    # getting the number masks folders #
    masks = count(folder)
    print("white: pair's masks forlder, magenta: class folder progress, green: trim folder progress.")
    for mask in tqdm(masks, total = len(masks), colour='green'):
        # **************************************************************************************** #
        #                  Grouping all the banners in their corresponding class                   #
        # NOTE: add as much necessary blocks as needed according to the number of existing classes #
        # **************************************************************************************** #

        print('\n Working on trim folder: ', mask)

        '''
        NOTE: add as much classes as needed to be read from the masks folder
        '''

        # Class 1 #
        datas = count(folder + '/' + mask, label='A')
        print('Class: A')
        class_name = 'Class_A'
        if not os.path.exists(folder + '/' + class_name):
            os.makedirs(folder + '/' + class_name)

        for data in tqdm(datas, total=len(datas), colour='magenta'):
            print('\n Working on class folder: ', data)
            files = count(folder + '/' + mask + '/' + data, names=True)
            cv2.imwrite(folder + '/' + class_name + '/' + str(unique_file_name)+'.tif', banner(files, path= folder + '/' + mask + '/' + data + '/'))
            unique_file_name +=1

        # Class 2 #
        datas = count(folder + '/' + mask, label='B')
        print('Class: B')
        class_name = 'Class_B'
        if not os.path.exists(folder + '/' + class_name):
            os.makedirs(folder + '/' + class_name)

        for data in tqdm(datas, total=len(datas), colour='magenta'):
            print('\n Working on class folder: ', data)
            files = count(folder + '/' + mask + '/' + data, names=True)
            cv2.imwrite(folder + '/' + class_name + '/' + str(unique_file_name)+'.tif', banner(files, path= folder + '/' + mask + '/' + data + '/'))
            unique_file_name +=1

        # Class 3 #
        datas = count(folder + '/' + mask, label='C')
        print('Class: C')
        class_name = 'Class_C'
        if not os.path.exists(folder + '/' + class_name):
            os.makedirs(folder + '/' + class_name)

        for data in tqdm(datas, total=len(datas), colour='magenta'):
            print('\n Working on class folder: ', data)
            files = count(folder + '/' + mask + '/' + data, names=True)
            cv2.imwrite(folder + '/' + class_name + '/' + str(unique_file_name)+'.tif', banner(files, path= folder + '/' + mask + '/' + data + '/'))
            unique_file_name +=1
    
    print("The data process is finished.") 
