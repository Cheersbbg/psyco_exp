from PIL import Image, ImageChops
from expfunctions import *
from natsort import natsorted
from os.path import basename
from pathlib import Path
from psychopy import core, visual, event,gui, sound
from pylsl import StreamInfo, StreamOutlet
from zipfile import ZipFile
import csv
import cv2
import datetime
import glob
import numpy as np
import os
import pathlib
import pickle
import psychopy
import pylsl
import queue
import random
import threading
import time
import serial

#psychopy.useVersion('2022.1.0')
psychopy.useVersion('2023.2.1')

testing_with_wifi = True
shared_queue=queue.Queue(maxsize=1)
now = datetime.datetime.now()
exp_running=True
paradigms=['go','nogo', 'freehand']

if True:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, local_clock
    from serial import Serial
    from threading import Thread, Event
    import sys
    CYTON_BOARD_ID = 0
    BAUD_RATE = 115200
    ANALOGUE_MODE = '/2'
    def find_openbci_port():
        """Finds the port to which the Cyton Dongle is connected to."""
        # Find serial port names per OS
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        openbci_port = ''
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                line = ''
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    c = ''
                    while '$$$' not in line:
                        c = s.read().decode('utf-8', errors='replace')
                        line += c
                    if 'OpenBCI' in line:
                        openbci_port = port
                s.close()
            except (OSError, serial.SerialException):
                pass
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port.')
            return openbci_port
    def start_cyton_lsl():
        """ 'start streaming cyton to lsl'
        Stream EEG and analogue(AUX) data from Cyton onto the Lab Streaming 
        Layer(LSL).
        (LSL is commonly used in EEG labs for different devices to push and pull
        data from a shared network layer to ensure good synchronization and timing
        across all devices)
        Returns
        -------
        board : board instance for the amplifier board, in this case OpenBCI Cyton
        push_thread : the thread instance that pushes data onto the LSL constantly
        Note
        ----
        To properly end the push_thread, call board.stop_stream(). If this isn't done,
        the program could freeze or show error messages. Do not lose the board instance
        Examples
        --------
        >>> board, _ = start_lsl() # to start pushing onto lsl
        ...
        >>> board.stop_streaming() # to stop pushing onto lsl
        """
        # print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCItestEEG\n")
        if CYTON_BOARD_ID != 0:
            info_eeg = StreamInfo('OpenBCIEEG', 'EEG', 16, 250, 'float32', 'OpenBCItestEEG')
            info_eeg = StreamInfo('OpenBCIEEG', 'EEG', 8, 250, 'float32', 'OpenBCItestEEG')
        print(BoardShim.get_board_descr(CYTON_BOARD_ID))
        
        # print("Creating LSL stream for AUX. \nName: OpenBCIAUX\nID: OpenBCItestEEG\n")
        info_aux = StreamInfo('OpenBCIAUX', 'AUX', 3, 250, 'float32', 'OpenBCItestAUX')
        outlet_eeg = StreamOutlet(info_eeg)
        outlet_aux = StreamOutlet(info_aux)
        params = BrainFlowInputParams()
        if CYTON_BOARD_ID != 6:
            params.serial_port = find_openbci_port()
        elif CYTON_BOARD_ID == 6:
            params.ip_port = 9000
        board = BoardShim(CYTON_BOARD_ID, params)
        board.prepare_session()
        res_query = board.config_board('/0')
        print(res_query)
        res_query = board.config_board('//')
        res_query = board.config_board(ANALOGUE_MODE)
        board.start_stream(45000)
        # time.sleep(1)
        stop_event = Event()
        def push_sample():
            start_time = local_clock()
            sent_eeg = 0
            sent_aux = 0
            while not stop_event.is_set():
                elapsed_time = local_clock() - start_time
                data_from_board = board.get_board_data()
                required_eeg_samples = int(250 * elapsed_time) - sent_eeg
                eeg_data = data_from_board[board.get_eeg_channels(CYTON_BOARD_ID)]
                # print(data_from_board[BoardShim.get_timestamp_channel(CYTON_BOARD_ID)])
                # print(BoardShim.get_sampling_rate(CYTON_BOARD_ID))
                datachunk = []
                for i in range(len(eeg_data[0])):
                    datachunk.append(eeg_data[:,i].tolist())
                stamp = local_clock()
                outlet_eeg.push_chunk(datachunk, stamp)
                sent_eeg += required_eeg_samples
                
                required_aux_samples = int(250 * elapsed_time) - sent_aux
                aux_data = data_from_board[board.get_analog_channels(CYTON_BOARD_ID)]
                for i in range(len(aux_data[0])):
                    datachunk.append(aux_data[:,i].tolist())
                outlet_aux.push_chunk(datachunk, stamp)
                sent_aux += required_aux_samples
                # time.sleep(0.02) # 20 ms
        push_thread = Thread(target=push_sample)
        push_thread.start()
        return board, stop_event
    board, stop_cyton = start_cyton_lsl()
def main():
    # experiment information 
    numsecs=2
    numexp=5
    numrest=1
    exp_running = True
   
    if testing_with_wifi == True:
         #Set up LabStreamingLayer stream.
        print("looking for streams")
        streams_EEG= pylsl.resolve_stream("type", "EEG") #Crown-215 
        streams_aux= pylsl.resolve_stream("type", "AUX")
        print(streams_EEG)
        print(streams_aux)
        #streams_AUX=pylsl.resolve_byprop("name", "openbci_aux",timeout= 5)
        if len(streams_EEG) == 0:
            print("Could not find stream on the network.")
            return
            
        eeg_inlet = pylsl.StreamInlet(streams_EEG[0])
        aux_inlet =  pylsl.StreamInlet(streams_aux[0])
        sample, timestamp = eeg_inlet.pull_sample()
        print(sample, timestamp)
        sample, timestamp = aux_inlet.pull_sample()
        print('a')
        #Get the sampling rate of the EEG LSL stream
        eeg_sample_rate = int(streams_EEG[0].nominal_srate())
        # Define the duration of the EEG and AUX data to save in seconds
        duration = numsecs+numrest
        print(eeg_sample_rate)
        # Define the number of samples to save
        eeg_num_samples = int(duration * eeg_sample_rate)

    # Create a photon sensor stimulus
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
        width=300,  # Width of the rectangle (adjust as needed)
        fillColor='black',  # Initial fill color
        lineColor='black',  # Border color
        )
    #load path, load images
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
            pathlist = getpicpaths(exp_lib[exp_name])
            stim_list=getstims(pathlist, window=win)
            stim_matrix.append(stim_list)
    print("stim_matrix", len(stim_matrix))
    emptystim=visual.ImageStim(win=win, image="emptyhand.png", pos=[0, 0])
    # Create another empty screen with just white pixels and the same size as emptystim
    whitestim = visual.GratingStim(win=win, size=emptystim.size, tex=None, color='gray')
    emptystim2 = visual.ImageStim(win=win, image="twohands2.png", pos = [0,0])
 

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
                # core.wait(0.1)
                stim2.draw() # background while doing it
                shared_queue.put([idx2, stim_timestamp, 2])
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
        eeg_thread = threading.Thread(target=runstream, args=(eeg_inlet,aux_inlet, eeg_num_samples*len(permu_list)))
        eeg_thread.start()
    run_exp(stim_matrix, numexp, win, emptystim=emptystim, hasbreak=True, break_interval=3 ,trial_interval=8, whitestim=whitestim, whiterect = whiterect, blackrect = blackrect)
    core.wait(5)
    print("haved looped through all tasks")
    exp_running = False
    print("done")
    win.close()
    core.quit()
def runstream(eeg_inlet, aux_inlet, num_samples, batch_size=1,exp_name="nogoactual2_eeg"):
    global exp_running
    if eeg_inlet is None:  # Do nothing if there's no EEG inlet (i.e., in testing mode)
        return
    # Initialize variables
    currentmarker = 0
    data_batch = []
    aux_batch = []
    counter = 0
    stim_timestamps=[]
    durations=[]
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
            aux_chunk, aux_timestamp = aux_inlet.pull_chunk(max_samples=num_samples + 150)
            filename = filename_template.format(exp_name)
            print(eeg_chunk, aux_chunk)
            data = np.hstack([
                np.array(timestamp).reshape(-1, 1),
                eeg_chunk
            ])
            aux_data = np.hstack([
                np.array(aux_timestamp).reshape(-1, 1),
                aux_chunk])
            # Add to data batch
            data_batch.append(data)
            aux_batch.append(aux_data)
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
                with open("aux_"+exp_name, 'ab') as f:
                    for batch in aux_batch:
                        np.savetxt(f, batch, delimiter=",", fmt=['%f'] + ['%f'] * 3)
                data_batch = []
                aux_batch = []
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