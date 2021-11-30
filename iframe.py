'''
iframe.py - ffmpeg i-frame extraction
'''

import sys, getopt, os

import subprocess, cv2
import numpy as np

# ffmpeg -i inFile -f image2 -vf "select='eq(pict_type,PICT_TYPE_I)'" -vsync vfr oString%03d.png

def main(argv):

    inFile = ''
    oString = 'out'
    usage = 'usage: python iframe.py -i <inputfile> [-o <oString>]'
    oPath = ''

   
    
    try:
        opts, args = getopt.getopt(argv,"hi:p:o",["ifile=","oPath=","oString="])
    except getopt.getopt.GetoptError:
        print (usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print (usage)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inFile = arg
        elif opt in ("-p", "--oPath"):
            oPath = arg
        elif opt in ("-o", "--oString"):
            oString = arg
        
        print('Input file is ' + inFile)
        print('oString is ' + oString)
        print('oPath is ' + oPath)

    if inFile == '':
        print (usage)
        sys.exit()

    # Start capturing the feed
    cap = cv2.VideoCapture(inFile)

    # Frame rate per second
    frame_rate = np.floor(cap.get(5))

    # Total number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    FRAMES_TO_SAVE_PER_VIDEO = 300
    # Calculate modulo to save frames throughout complete video, rather than frames [1:1+FRAMES_PER_VIDEO]
    mod = 1
    if video_length > FRAMES_TO_SAVE_PER_VIDEO:
        mod = video_length // FRAMES_TO_SAVE_PER_VIDEO

    number_of_frames_to_save = FRAMES_TO_SAVE_PER_VIDEO
    
    print(f"Video: {inFile}, #frames: {video_length}, FPS: {frame_rate}, #frames to save: {number_of_frames_to_save}.")

    frames_saved = 0
    count = 0

    # while cap.isOpened():
    #     # Extract the frame
    #     ret, frame = cap.read()

    #     # Frame is available
    #     if ret:
    #         # Get current frame id
    #         frame_id = cap.get(1)

            
    #         save_frame = frame_id % mod == 0

    #         # Write frame to disk
    #         if save_frame:
    #             # Check whether we have to resize or crop the frame
    #             cv2.imwrite(oPath + f"/{oString}-" + "%#05d.jpg" % frame_id, frame)
    #             frames_saved = frames_saved + 1
    #     count += 1

    #     if (frames_saved >= number_of_frames_to_save or count >= video_length):
    #         # Release the feed
    #         if cap.isOpened():
    #             cap.release()

    #         break

    # return frames_saved
    # # home = os.path.expanduser("~")
    # print("Path at terminal when executing this file")
    # home = os.path.expanduser("~")
    # print(home)
    ffmpeg = '/usr/bin/ffmpeg'
    # # path = os.path.join(os.getcwd() + ffmpeg)
    # # os.mkdir(path)
    outFile = oString + '%03d.jpg'
    outFilePath = os.path.join(oPath, outFile)
    
    cmd = [ffmpeg,'-i', inFile,'-f', 'image2','-vf', 
               "select='eq(pict_type,PICT_TYPE_I)'",'-vsync','vfr',outFilePath]

    
    print("hello")
    print (cmd)
    subprocess.call(cmd)

if __name__ == "__main__":
	main(sys.argv[1:])