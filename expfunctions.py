import psychopy
psychopy.useVersion('2022.1.0')
from psychopy import core, visual
from PIL import Image, ImageChops

from natsort import natsorted
import random
import numpy as np
import cv2
from zipfile import ZipFile
import os
from os.path import basename


# Zip the files from given directory that matches the filter
def zipFilesInDir(dirName, zipFileName, filter):
 # create a ZipFile object
    with ZipFile(zipFileName, 'w') as zipObj:
    # Iterate over all the files directory
        for folderName, subfolders, filenames in os.walk(dirName):
            for filename in filenames:
                if filter(filename):
                   # create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                   # Add file to zip
                    zipObj.write(filePath, basename(filePath))

def custom_experiement(listoflist, inclusive, newfunction, iswhite=True): 
    contours = np.load('handcountours.npy', allow_pickle=True)
   
    if iswhite==False:
        template = cv2.imread('emptyhand.png')
    else:
        template = cv2.imread('twohands2.png')
    inv_img = ImageChops.invert(Image.fromarray(template)).convert("RGBA")
    #generate map
    background = np.zeros( (600,800) ) 
    if not os.path.exists(os.getcwd()+"/"+newfunction):
        exp_path=os.makedirs(os.getcwd()+"/"+newfunction+"/exp")
        map_path=os.makedirs(os.getcwd()+"/"+newfunction+"/map")
    else:
        exp_path=os.getcwd()+"/"+newfunction+"/exp"
        map_path=os.getcwd()+"/"+newfunction+"/map"

    for key,list in enumerate (listoflist):
        if inclusive==False:
            background = np.zeros( (600,800) ) 
        for cont in list:
            cv2.drawContours(background, [contours[cont]],0,color=(255, 255, 255),thickness=-1)
        newp= Image.fromarray(background)
        new_p = newp.convert("L")
        print(map_path)
        new_p.save(map_path+'/image'+str(key)+'.png')
    
        #generate exp gifs   
        temp = cv2.imread(map_path+'/image'+str(key)+'.png')
        exp_tem = inv_img.copy()
        b, g, r = cv2.split(np.array(temp))
        np.multiply(b, 2, out=b, casting="unsafe")
        np.multiply(g, 0, out=g, casting="unsafe")
        np.multiply(r, 2, out=r, casting="unsafe")
        after = cv2.merge([b, g, r])
        after = ImageChops.invert(Image.fromarray(after)).convert("RGBA")
        alphaBlended = Image.blend(after, exp_tem, alpha=0.6)
        alphaBlended.save(exp_path+'/image'+str(key)+'.png')
        
    return exp_path,map_path
 
            #zipFilesInDir(os.getcwd()+"/custofdatamexp", 'customexp_' + str(now.strftime("%d_%m_%H_%M_%S"))+'.zip', lambda name : 'image' in name)
            #zipFilesInDir(os.getcwd()+"/customap", 'customap_' + str(now.strftime("%d_%m_%H_%M_%S")) +'.zip', lambda name : 'image' in name)
       

def getpicpaths(path):
    path_list=[]
    for filename in os.listdir(path):
        if "image" in filename:
            path_list.append(path+"/"+filename)
    path_list=natsorted(path_list)
    return path_list
    
def getstims(path_lst,window):
    stim_list=[]
    for item in path_lst:
        stim_list.append(visual.ImageStim(win=window, image=str(item), pos = [0,0]))
    return stim_list

