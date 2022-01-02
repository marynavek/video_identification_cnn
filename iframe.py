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
    
    # ffmpeg = '/usr/bin/ffmpeg'
    ffmpeg = '/usr/local/bin/ffmpeg'
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