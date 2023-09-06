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
    videoType = ''
    videoName = ''

   
    
    try:
        opts, args = getopt.getopt(argv,"hi:p:o:c:d:",["ifile=","oPath=","videoType=","videoName=","oString="])
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
        elif opt in ("-o", "--videoType"):
            videoType = arg
        elif opt in ("-c", "--videoName"):
            videoName = arg
        elif opt in ("-d", "--videoName"):
            oString = arg
        
        print('Input file is ' + inFile)
        print('oString is ' + oString)
        print('videoType is ' + videoType)
        print('videoName is ' + videoName)

    if inFile == '':
        print (usage)
        sys.exit()
    
    # ffmpeg = '/usr/bin/ffmpeg'
    ffprobe = '/usr/local/bin/ffprobe'
    # # path = os.path.join(os.getcwd() + ffmpeg)
    # # os.mkdir(path)
    outFile = videoName + '.txt'
    
        
    outFilePath = os.path.join(oPath, outFile)

    cmd = [ffprobe, inFile, '-show_frames']

    
    print("hello")
    print (cmd)
    print(inFile)
    subprocess.call(cmd)

if __name__ == "__main__":
	main(sys.argv[1:])