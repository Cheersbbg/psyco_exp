import psychopy
psychopy.useVersion('2022.1.0')
from psychopy import visual, core, event
from psychopy import core, visual, sound
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
import math


shared_queue=queue.Queue(maxsize=1)
now = datetime.datetime.now()

exp_running=True

import time
# from pyOpenBCI import OpenBCICyton
# from pylsl import StreamInfo, StreamOutlet

#Crown-215
beep1 = sound.Sound(
    value = 'A', secs = 0.2,
    volume = 0.5)
beep2 = sound.Sound(
    value = 'B', secs = 0.2,
    volume = 0.8)

class ExperimentConfig:
    def __init__(self, vis_second=6.0, numrest_aftervisual=1.0, numrest=2.0, tempo_list=['fast', 'slow', 'even'], exp_names = ["midtoleft","midtoright","midtoside","lefttomid","sidetomid","righttomid"],numruns_each_exp=3):
        self.vis_second = vis_second
        self.numrest_aftervisual = numrest_aftervisual
        self.numrest = numrest
        self.tempo_list = tempo_list
        self.numruns_each_exp = numruns_each_exp
        self.exp_names= exp_names

config = ExperimentConfig()

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
    exp_running = True
   
    #Set up LabStreamingLayer stream.
    print("looking for streams")
    #streams_EEG= pylsl.resolve_byprop("name", "openbci_eeg",timeout=5) 
    streams_EEG= pylsl.resolve_byprop("name", "Crown-215",timeout=5)

    #streams_AUX=pylsl.resolve_byprop("name", "openbci_aux",timeout= 5)

    if len(streams_EEG) == 0:
        print("Could not find stream on the network.")
        return

    eeg_inlet = pylsl.StreamInlet(streams_EEG[0])
    #aux_inlet =  pylsl.StreamInlet(streams_AUX[0])

    sample, timestamp = eeg_inlet.pull_sample()

    #Get the sampling rate of the EEG LSL stream
    eeg_sample_rate = int(streams_EEG[0].nominal_srate())
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
   
    new_exp_dic = {"updown": [[35,19,15,27,37,36,26,14,18,34],[25,24,31,17,29,39,38,28,16,30],[11,21,13,23,33,32,22,12,20,10],[7,1,6,0],[5,9,4,8],[3,2]],
                   "midtoright": [[6, 0, 4, 8, 2],[36, 38, 32], [26, 28, 22],[14, 16, 12],[18, 30, 20],[34, 24, 10, 4]],
                   "midtoleft": [[7,1,5,9,3],[37,39,33],[27,29,23],[15,17,13],[19,31,21],[35,25,11,5]],
                   "midtoside": [[6, 0, 4, 8, 2, 7, 1, 5, 9, 3],[36, 38, 32, 37, 39, 33],[26, 28, 22, 27, 29, 23],[14, 16, 12, 15, 17, 13],[18, 30, 20, 19, 31, 21],[34, 24, 10, 4, 35, 25, 11, 5]]
                   }
    new_exp_dic["downup"]=new_exp_dic["updown"][::-1]
    new_exp_dic["lefttomid"]=new_exp_dic["midtoleft"][::-1]
    new_exp_dic["righttomid"]=new_exp_dic["midtoright"][::-1]
    new_exp_dic["sidetomid"]=new_exp_dic["midtoside"][::-1]
    


    stim_dict={}

    for exp_name in config.exp_names:
        stim_dict[exp_name] = []
        if exp_name not in exp_lib.keys():
            # create experiment directory if it doesn't exist
            if exp_name in new_exp_dic.keys():
                listoflist=new_exp_dic[exp_name]
                print(exp_name)
                exp_path, map_exp=custom_experiement(listoflist, inclusive=False, newfunction=exp_name)
                exp_lib[exp_name]=exp_path
                print("new task added to lib")
                with open('lib_dict.pickle', 'wb') as f:
                # dump the dictionary to the file
                    pickle.dump(exp_lib, f)
                pathlist = getpicpaths(exp_lib[exp_name])
                print("pathlist", pathlist)
                stim_list=getstims(pathlist,window=win)
                stim_dict[exp_name]=stim_list
            else:
                print("class missing information ignored")
        else:
            pathlist = getpicpaths(exp_lib[exp_name])
            stim_list=getstims(pathlist, window=win)
            stim_dict[exp_name]=stim_list


    emptystim=visual.ImageStim(win=win, image="emptyhand.png", pos = [0,0])


    def calculatetempo(unit, tempotype,lenstims, ratio=0.5):
        intervallst=[unit]*lenstims
        first_half=lenstims//2
        fast_length= unit*ratio 
        slow_length= (lenstims*unit- first_half*fast_length)/(lenstims-first_half)  
        
        if tempotype=="fast":    
            intervallst = [fast_length if i % 2 == 0 else slow_length for i in range(len(intervallst))]

        elif tempotype=="slow":
            intervallst = [slow_length if i%2 == 0 else fast_length for i in range(len(intervallst))]
            
        return intervallst


    def run_exp(exp_name, vis_second,numrest_aftervisual, stim_list, tempotype,numexp, win, shuffle=False, emptystim=None, hasbreak=False, numrest=3):
    #keys = event.waitKeys(keyList=['space'])
    #if 'space' in keys:

        # permu_list=list(range(0,len(stim_list)))
        # permu_list=permu_list*numruns
        #core.wait(3.5)
        #timer = core.Clock()
        exp_running=True
        # for i in range(len(permu_list)):
        #     print("run: ", i)
        #     if shuffle == True:
        #         random.shuffle(permu_list)
                
        each_stimulus_presented=vis_second/len(stim_list)
        timelist=calculatetempo(unit=each_stimulus_presented, tempotype=tempotype, lenstims=len(stim_list))
        print(stim_list)
           
        for idx in range(0, len(stim_list)):
            print("idx", idx)
            stim = stim_list[idx]
            print(stim)
            stim.draw()
            win.flip()
            #keey looping
            core.wait(timelist[idx]/2)  #only show half   vis_second/2 + numrest_aftervisual +vis_second + numrest=3
        numrest_aftervisual = random.uniform(1, numrest_aftervisual)
        emptystim.draw()
        win.flip()
        core.wait(numrest_aftervisual)
        sample, timestamp = eeg_inlet.pull_sample()
        start_time=time.time()
        eeg_timestamp = timestamp
        time_diff = start_time - eeg_timestamp


        for idx, inter in enumerate(timelist):
            shared_queue.put([idx, time.time() - time_diff+0.01])
            time.sleep(0.01)
            beep1.play()
            core.wait(inter)
            beep1.stop()
        beep2.play() # openeyes
        core.wait(1)
        beep2.stop()
        if hasbreak == True:
            breaktime = random.uniform(1, numrest)
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
        shared_queue.put([200,stim_timestamp, exp_name+"_"+tempotype])
        print("one task done")

    

    eeg_thread = threading.Thread(target=runstream, args=(eeg_inlet,math.ceil(eeg_sample_rate*(config.vis_second/2 + config.numrest_aftervisual +config.vis_second + config.numrest))))
    eeg_thread.start()


    #create permutation, 3 experiment 30 runs 90 runs total
    experiments = sorted([name for name in config.exp_names for _ in range(config.numruns_each_exp)],
                     key=lambda x: (x.split("_")[0], config.exp_names.index(x.split("_")[0])))
    
    print("total", len(experiments))


    repeat_per_tempo=config.numruns_each_exp//len(config.tempo_list)

    tempo_list = [val for val in config.tempo_list for _ in range(repeat_per_tempo)]*len(config.exp_names)

    exp_tuple = list(zip(experiments, tempo_list))

    random.shuffle(exp_tuple)
  

    for i, (exp_name,tempotype) in enumerate(exp_tuple):
        stim_list=stim_dict[str(exp_name)]
        print(stim_list)
        sample, timestamp = eeg_inlet.pull_sample()
        start_time=time.time()
        eeg_timestamp = timestamp
        time_diff = start_time - eeg_timestamp
        run_exp(exp_name,vis_second=config.vis_second, numrest_aftervisual=config.numrest_aftervisual, stim_list=stim_dict[str(exp_name)], tempotype=tempotype, numexp=1, win=win, shuffle=False,hasbreak=False,emptystim=emptystim,numrest=config.numrest)
        print(i, "trial")
        time.sleep(0.1)

    print("haved looped through all tasks")
    exp_running = False
    print("done")
    win.close()
    core.quit()

def runstream(eeg_inlet,num_samples):
    global exp_running
    tempmarker=[]
    stim_timestamps=[]

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

