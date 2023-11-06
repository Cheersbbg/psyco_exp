import psychopy
#psychopy.useVersion('2022.1.0')
psychopy.useVersion('2023.2.1')
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
from expfunctions import *


testing_with_wifi = True
shared_queue=queue.Queue(maxsize=1)
now = datetime.datetime.now()

exp_running=True

import time
from pylsl import StreamInfo, StreamOutlet

paradigms=['go','nogo', 'freehand']
#Crown-215
# beep1 = sound.Sound(
#     value = 'A', secs = 0.2,
#     volume = 0.5)


def main():

    # experiment information 
    numsecs=2
    numexp=1
    numrest=1
    exp_running = True

   
    if testing_with_wifi == True:
         #Set up LabStreamingLayer stream.
        print("looking for streams")
        streams_EEG= pylsl.resolve_byprop("name", "openbci_eeg",timeout=5) #Crown-215
        #streams_EEG= pylsl.resolve_byprop("name", "Crown-215",timeout=5)


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
    # win = visual.Window([1440, 900], allowGUI=None, fullscr=True, monitor='testMonitor',units='deg')

    # whiterect = visual.Rect(
    # win=win,
    # width=300,  # Width of the rectangle (adjust as needed)
    # height=300,  # Height of the rectangle (adjust as needed)
    # fillColor='white',  # Initial fill color
    # lineColor='white',  # Border color
    # pos=(1440//2, -900//2))
    from psychopy import visual, core, event
    win = visual.Window([1512, 982], allowGUI=None, fullscr=True, monitor='testMonitor', units='pix')

    whiterect = visual.Rect(
        win=win,
        width=300,  # Width of the rectangle (adjust as needed) 
        height=300,  # Height of the rectangle (adjust as needed)
        fillColor='white',  # Initial fill color
        lineColor='white',  # Border color
        pos=(1512/2, -982/2))


    # Create a white and black rectangle
    blackrect = visual.Rect(
        win=win,
        width=300,  # Width of the rectangle (adjust as needed)
        height=300,  # Height of the rectangle (adjust as needed)
        fillColor='black',  # Initial fill color
        lineColor='black',  # Border color
        pos=(1512/2, -982/2))


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
    "singlefingerdark": os.getcwd()+"/singlefingerdark/exp"
    }
    duration_per_run = [30, 30, 30]

    exp_names = ["singlefinger", "singlefingerdark"]
  
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

    print("stim_matrix", len(stim_matrix))



    emptystim=visual.ImageStim(win=win, image="emptyhand.png", pos=[0, 0])

    # Create another empty screen with just white pixels and the same size as emptystim
    whitestim = visual.GratingStim(win=win, size=emptystim.size, tex=None, color='gray')

    emptystim2 = visual.ImageStim(win=win, image="twohands2.png", pos = [0,0])

    # # Calculate the position for the center of the lower-right corner
    # win_width, win_height = win.size
    # rect_width = 300  # Width of the rectangle
    # rect_height = 300  # Height of the rectangle
    # rect_x = win_width / 2 - rect_width / 2  # X-coordinate
    # rect_y = -win_height / 2 + rect_height / 2  # Y-coordinate

    # # Create a black rectangle
    # blackrect = visual.Rect(
    #     win=win,
    #     width=rect_width,
    #     height=rect_height,
    #     fillColor='black',  # Initial fill color
    #     lineColor='black',  # Border color
    #     pos=(rect_x, rect_y),  # Position at the center of the lower right corner
    # )

   

    # experiment goes visual(2 second), see it then close eye, then hear beap to perform task, then second beep stop and open eyes then break for one second


    def run_exp(stim_matrix, numexp, win, emptystim=None, hasbreak=False, break_interval=0.5,trial_interval=8, whitestim=None, whiterect = None, blackrect = None):
        saolist = stim_matrix[0]
        movementlist = stim_matrix[1]

        type_list = [0,1,2] #paradigms=['go','nogo', 'freehand']
        core.wait(3.5)
        #timer = core.Clock()
        exp_running=True
        for i in range(numexp):
            
            print("run: ", i)
           
            random.shuffle(type_list)

            stim_list_total = []
            movement_list_total = []
            idx1_total = []
            idx2_total = []

            for i in type_list:
                currentexp = paradigms[i]
                permu_list=list(range(0,len(saolist)))
                permu_list2=list(range(0,len(movementlist)))

                random.shuffle(permu_list)
                random.shuffle(permu_list2)

                if currentexp == "go":
                    stim_list_total.extend([saolist[i] for i in permu_list]) 
                    movement_list_total.extend([movementlist[i] for i in permu_list])
                    idx1_total.extend(permu_list)
                    idx2_total.extend(permu_list)# exactly the same index 

                elif currentexp == "freehand":
                    stim_list_total.extend([emptystim for i in permu_list]) 
                    movement_list_total.extend([emptystim2 for i in permu_list])
                    idx1_total.extend([456 for i in permu_list])
                    idx2_total.extend([654 for i in permu_list])

                elif currentexp == "nogo":
                    stim_list_total.extend([saolist[i] for i in permu_list])
                    while permu_list2 == permu_list:
                        random.shuffle(permu_list2)
                    movement_list_total.extend([movementlist[i] for i in permu_list2])
                    idx1_total.extend(permu_list)
                    idx2_total.extend(permu_list2)

            # Shuffle the lists
            zipped_lists = list(zip(stim_list_total, movement_list_total, idx1_total, idx2_total))
            random.shuffle(zipped_lists)
            stim_list_total, movement_list_total, idx1_total, idx2_total = zip(*zipped_lists)
                        
            for stim1, stim2, idx1, idx2 in zip(stim_list_total, movement_list_total, idx1_total, idx2_total):

                stim1.draw()
                whiterect.draw()
                

                sample, timestamp = eeg_inlet.pull_sample()
                start_time=time.time()
                eeg_timestamp = timestamp
                time_diff = start_time - eeg_timestamp
                trial_multiplier = 2 #random.randint(1, trial_interval)  # This will give you a number between 1 and 8
                # numsecs = trial_multiplier * 0.5 
                system_time = time.time()  # Get the current system time
                stim_timestamp = system_time - time_diff  # Calculate the stimulus onset time using the time difference
                shared_queue.put([idx1, stim_timestamp, 1])

                win.flip()
                core.wait(1.5) #visual presentation seconds

                stim2.draw() # background while doing it
                whiterect.draw()

                system_time = time.time()  # Get the current system time
                stim_timestamp = system_time - time_diff  # Calculate the stimulus onset time using the time difference
                shared_queue.put([idx2, stim_timestamp, 2])

                win.flip()
                core.wait(1.5) #random interval seconds
                
                if hasbreak == True:
                    break_multiplier = random.randint(3, 5) 
                    system_time = time.time()  # Get the current system time
                    stim_timestamp = system_time - time_diff  # Calculate the stimulus onset time using the 
                    shared_queue.put([100,stim_timestamp,break_multiplier])
                    whitestim.draw() 
                    blackrect.draw()
                    #draw a black one
                    win.flip()
                    # This will give you a number between 1 and 8
                    breaktime = break_multiplier * 0.25
                    # rest for 1 seconds between trials 
                    core.wait(breaktime)
                    #time.sleep(0.1)

            
        system_time = time.time()  # Get the current system time
        stim_timestamp = system_time - time_diff
        shared_queue.put([200,stim_timestamp])
        print("one round done")
        core.wait(2)

    if testing_with_wifi:
        eeg_thread = threading.Thread(target=runstream, args=(eeg_inlet,eeg_num_samples*len(permu_list)))
        eeg_thread.start()

    run_exp(stim_matrix, numexp, win, emptystim=emptystim, hasbreak=True, break_interval=3 ,trial_interval=8, whitestim=whitestim, whiterect = whiterect, blackrect = blackrect)
    core.wait(5)
    

    print("haved looped through all tasks")
    exp_running = False
    print("done")
    win.close()
    core.quit()


def runstream(eeg_inlet, num_samples, batch_size=1,exp_name="nogotest"):
    global exp_running
    if eeg_inlet is None:  # Do nothing if there's no EEG inlet (i.e., in testing mode)
        return

    # Initialize variables
    currentmarker = 0
    data_batch = []
    counter = 0
    stim_timestamps=[]
    eeg_timestamps = []
    marker_timestamps = []
    marker_indices = []
    durations=[]
    duration_channel=[]
    tempmarker=[]
    

    # Check if file exists outside of the loop
    filename_template = "data_{}.csv"
    file_mode = 'ab' if os.path.isfile(filename_template.format("placeholder")) else 'ab'

    while exp_running:
        # Read from the queue
        if not shared_queue.empty():
            currentmarker_timestamp = shared_queue.get()
            
            if currentmarker_timestamp[0] != 200:
                tempmarker.append(currentmarker_timestamp[0])
                stim_timestamps.append(currentmarker_timestamp[1])
                durations.append(currentmarker_timestamp[2])
                continue

            # Read a chunk of EEG and AUX data
            eeg_chunk, timestamp = eeg_inlet.pull_chunk(max_samples=num_samples + 150)

            #aux_chunk, aux_timestamp = aux_inlet.pull_chunk(max_samples=num_samples + 150)

            filename = filename_template.format(exp_name)

            data = np.hstack([
                np.array(timestamp).reshape(-1, 1),
                eeg_chunk
                #aux_chunk
            ])

            # Add to data batch
            data_batch.append(data)
            counter += 1

            if counter >= batch_size:
                # Save data batch to CSV file
                with open(filename, file_mode) as f:
                    for batch in data_batch:
                        np.savetxt(f, batch, delimiter=",", fmt=['%f'] + ['%f'] * 8) #add 3 more channels
                with open("stimulus "+exp_name, 'ab') as f:
                    marker_array = np.array(tempmarker).reshape(-1, 1)
                    stim_timestamps_array = np.array(stim_timestamps).reshape(-1, 1)
                    durations_array = np.array(durations).reshape(-1, 1)
                    stimulus_data = np.hstack([stim_timestamps_array,marker_array, durations_array])
                    np.savetxt(f, stimulus_data, delimiter=",", fmt=['%f', '%d', '%d'])
                        
                data_batch = []
                stim_timestamps=[]
                tempmarker = []
                durations = []

                counter = 0


    # Save any remaining data in the batch after loop ends
    if data_batch:
        with open(filename, file_mode) as f:
            for batch in data_batch:
                np.savetxt(f, batch, delimiter=",", fmt=['%f'] + ['%f'] * 8 + ['%s']+['%f']+[ '%f'])

# '''''''
if __name__ == "__main__":
    # Run the run() functiona
    main()

