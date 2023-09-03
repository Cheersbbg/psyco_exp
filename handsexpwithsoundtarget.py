import psychopy
psychopy.useVersion('2022.1.0')
from psychopy import core, visual, event,gui, sound
from pylsl import StreamInfo, StreamOutlet
from PIL import Image, ImageChops
from pathlib import Path
import glob
from pathlib import Path
from natsort import natsorted
import random
import numpy as np
import cv2
from zipfile import ZipFile
import os
from os.path import basename
import datetime
import pathlib
import pylsl
import threading
import queue
import pickle
import csv

testing_with_wifi = False
shared_queue=queue.Queue(maxsize=1)
now = datetime.datetime.now()

exp_running=True

import time
from pylsl import StreamInfo, StreamOutlet

#Crown-215
beep1 = sound.Sound(
    value = 'A', secs = 0.2,
    volume = 0.5)


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

def custom_experiement(listoflist, inclusive, newfunction):
    contours = np.load('handcountours.npy', allow_pickle=True)
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


def main():

    # experiment information 
    numsecs=2
    numexp=20
    numrest=1
    exp_running = True

    
   
   
    if testing_with_wifi == True:
         #Set up LabStreamingLayer stream.
        print("looking for streams")
        #streams_EEG= pylsl.resolve_byprop("name", "openbci_eeg",timeout=5) #Crown-215
        streams_EEG= pylsl.resolve_byprop("name", "Crown-215",timeout=5)


        #streams_AUX=pylsl.resolve_byprop("name", "openbci_aux",timeout= 5)

        if len(streams_EEG) == 0:
            print("Could not find stream on the network.")
            return
            

        eeg_inlet = pylsl.StreamInlet(streams_EEG[0])
        #aux_inlet =  pylsl.StreamInlet(streams_AUX[0])

        sample, timestamp = eeg_inlet.pull_sample()
        print(sample, timestamp)

        #Get the sampling rate of the EEG LSL stream
        eeg_sample_rate = int(streams_EEG[0].nominal_srate())

        # Define the duration of the EEG and AUX data to save in seconds
        duration = numsecs+numrest
        print(eeg_sample_rate)

        # Define the number of samples to save
        eeg_num_samples = int(duration * eeg_sample_rate)
    #aux_num_samples = int(duration * aux_sample_rate)

    # Create an outlet for the second stream
    #stream_info = pylsl.StreamInfo('combined', 'EEG', 17, 0, 'float32', 'example_stream_out_001') #9 for crown
    #outlet = pylsl.StreamOutlet(stream_info)

    #input("Make sure it's recording then press enter...")

    #Instantiate the PsychoPy window and stimuli.
    win = visual.Window([1512, 982], allowGUI=None, fullscr=True, monitor='testMonitor',units='deg')

    #load path, load images
    thispath='/Users/qianxigong/Downloads/introspection'
    contours = np.load('handcountours.npy', allow_pickle=True)
    permu_list=list(range(0,len(contours)))
    
    exp_lib = {
    "single": os.getcwd()+"/handmapexp",
    "double":  os.getcwd()+"/handmapboth",
    "lefttoright": os.getcwd()+"/exp_allfingerlefttoright[[35, 25, 11], [19, 31, 21], [15, 17, 13], [27, 29, 23], [37, 39, 33], [36, 38, 32], [26, 28, 22], [14, 16, 12], [18, 30, 20], [34, 24, 10]]",
    "righttoleft": os.getcwd()+"/exp_allfingerrighttoleft[[34, 24, 10], [18, 30, 20], [14, 16, 12], [26, 28, 22], [36, 38, 32], [37, 39, 33], [27, 29, 23], [15, 17, 13], [19, 31, 21], [35, 25, 11]]",
    "singlefinger": os.getcwd()+"/exp_eachfinger[[35, 25, 11], [19, 31, 21], [15, 17, 13], [27, 29, 23], [37, 39, 33], [36, 38, 32], [26, 28, 22], [14, 16, 12], [18, 30, 20], [34, 24, 10]]",
    
    }
    duration_per_run = [30, 30, 30]

    exp_names = ["singlefinger"]
  
    new_exp_dic = {"updown": [[35,19,15,27,37,36,26,14,18,34],[25,24,31,17,29,39,38,28,16,30],[11,21,13,23,33,32,22,12,20,10],[6,7,1,0,5,4,9,8],[3,2]]}


    stim_matrix=[]

    for exp_name in exp_names:
        if exp_name not in exp_lib.keys():
            # create experiment directory if it doesn't exist
            if exp_name in new_exp_dic.keys():
                listoflist=new_exp_dic[exp_name]
                exp_path, map_exp=custom_experiement(listoflist, inclusive=False, newfunction=exp_name)
                exp_lib[exp_name]=exp_path
                print("new task added to lib")
                with open('lib_dict.pickle', 'wb') as f:
                # dump the dictionary to the file
                    pickle.dump(exp_lib, f)
                pathlist = getpicpaths(exp_lib[exp_name])
                print("pathlist", pathlist)
                stim_list=getstims(pathlist,window=win)
                stim_matrix.append(stim_list)   
                print("matrix", stim_matrix)
            else:
                print("class missing information ignored")
        else:
            pathlist = getpicpaths(exp_lib[exp_name])
            stim_list=getstims(pathlist, window=win)
            stim_matrix.append(stim_list)

    print("stim_matrix", len(stim_matrix[0]))


    emptystim=visual.ImageStim(win=win, image="emptyhand.png", pos = [0,0])

    # experiment goes visual(2 second), see it then close eye, then hear beap to perform task, then second beep stop and open eyes then break for one second


    def run_exp(exp_name,stim_list, numexp, win, timedif, shuffle=False, emptystim=None, hasbreak=False, numrest=0):
    #keys = event.waitKeys(keyList=['space'])
    #if 'space' in keys:
        permu_list=list(range(0,len(stim_list)))
        core.wait(3.5)
        #timer = core.Clock()
        exp_running=True
        for i in range(numexp):
            print("run: ", i)
            if shuffle == True:
                random.shuffle(permu_list)
            for idx in permu_list:
                print("idx", idx)
                print(len(permu_list), permu_list)
                stim = stim_list[idx]
                stim.draw()
                system_time = time.time()  # Get the current system time
                stim_timestamp = system_time - time_diff  # Calculate the stimulus onset time using the time difference
                shared_queue.put([idx, stim_timestamp])
                win.flip()
                core.wait(numsecs) #2 seconds
                time.sleep(0.1)
                #introduce sound for eye close after seeing the finger then wait for the beep to start task
                beep1.play()
                core.wait(1) #1 seconds 
                beep1.stop()
                core.wait(3)#3 second
                beep1.play() # second time play peep stop and then open eyes 
                core.wait(1)
                beep1.stop()
                #while timer.getTime() < numsecs:
                #passtime.sleep(numsecs)  # Display picture for 3 second.
                if hasbreak == True:
                    breaktime = random.uniform(2, 4)
                    emptystim.draw() 
                    system_time = time.time()  # Get the current system time
                    stim_timestamp = system_time - time_diff  # Calculate the stimulus onset time using the 
                    shared_queue.put([100,stim_timestamp])
                     # rest for 1 seconds between trials
                    win.flip()
                    core.wait(breaktime)
            print("get to the first save")
            system_time = time.time()  # Get the current system time
            stim_timestamp = system_time - time_diff
            shared_queue.put([200,stim_timestamp, exp_name])


            print("one task done")
    

    eeg_thread = threading.Thread(target=runstream, args=(eeg_inlet,eeg_num_samples*len(permu_list),numsecs))
    eeg_thread.start()

    

    for i, stim_list in enumerate(stim_matrix):
        sample, timestamp = eeg_inlet.pull_sample()
        start_time=time.time()
        eeg_timestamp = timestamp
        time_diff = start_time - eeg_timestamp
        run_exp(exp_names[i], stim_list, numexp=numexp, win=win, shuffle=True,hasbreak=True,emptystim=emptystim,timedif=time_diff)
        print(i, "experiment")
        core.wait(5)
    

    print("haved looped through all tasks")
    exp_running = False
    print("done")
    win.close()
    core.quit()

def runstream(eeg_inlet,num_samples,numsecs):
    global exp_running
    currentmarker=0
    tempmarker=[]
    stim_timestamps=[]
    eeg_timestamps = []
    marker_timestamps = []
    marker_indices = []


    while exp_running==True:
        # Read a sample from the inlet
        if not shared_queue.empty():
            currentmarker_timestamp=shared_queue.get()
                #save the indexes 
            if currentmarker_timestamp[0]!=200:
                tempmarker.append(currentmarker_timestamp[0])
                stim_timestamps.append(currentmarker_timestamp[1])
                print(currentmarker_timestamp[0])
                continue
            # Read a chunk of EEG and AUX data from the LSL streams+1seconds
            print(num_samples, "num_samples to pull")
            eeg_chunk, timestamp = eeg_inlet.pull_chunk(max_samples=num_samples+150)
            #experiment with 
            exp_name=currentmarker_timestamp[2]

            marker_channel = np.empty((len(eeg_chunk), 1), dtype=object)
            sti_channel =  np.empty((len(eeg_chunk), 1), dtype=float)
            marker_channel.fill([])
            sti_channel.fill(0)
            stim_timestamps_array = np.array(stim_timestamps).reshape(-1, 1)
            tempmakerarray=np.array(tempmarker).reshape(-1, 1)
            marker_channel[:len(stim_timestamps_array)] = stim_timestamps_array
            sti_channel[:len(tempmakerarray)] = tempmakerarray


            
            data = np.concatenate((np.array(timestamp).reshape(-1, 1),eeg_chunk,marker_channel,sti_channel), axis=1)
            filename = f"data_{exp_name}.csv"
            # Define format specifiers for each column
            fmt = ['%f'] + ['%f'] * 8 + ['%s']+['%f']

            # Save data to CSV file with specified formats
            if os.path.isfile(filename):
                with open(filename, 'ab') as f:
                    np.savetxt(f, data, delimiter=",", fmt=fmt)
            else:
                np.savetxt(filename, data, delimiter=",", fmt=fmt)

            tempmarker=[]
            stim_timestamps=[]
            print("saved one, should continue")
# '''''''
if __name__ == "__main__":
    # Run the run() functiona
    main()

