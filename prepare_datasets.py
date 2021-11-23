import os
import numpy as np
import cv2

TRAIN_DIR="/Users/marynavek/Projects/Video_Project/train"
TEST_DIR="/Users/marynavek/Projects/Video_Project/test"

def create_dataset(directory, model="all"):

    LABELS = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]

    features = []
    labels = []
    temp_features = [] 
    
    
    for class_index, class_name in enumerate(LABELS):
        print('Extracting Data of Class: ' + class_name)
        files_list = os.listdir(os.path.join(directory, class_name))
        frames_list = []
        if model == "all":
            for file in files_list:
                image_file_path = os.path.join(directory, class_name, file)

                frame = cv2.imread(image_file_path)
                resized_frame = cv2.resize(frame, (60,100))
                frames_list.append(resized_frame)
            features.extend(frames_list)
            labels.extend([class_index]*len(files_list))
            print(class_index)
        else:
            if model == class_name:
                for file in files_list:
                    image_file_path = os.path.join(directory, class_name, file)
                    frame = cv2.imread(image_file_path)
                    resized_frame = cv2.resize(frame, (60,100))
                    frames_list.append(resized_frame)
                features.extend(frames_list)
                labels.extend([class_index]*len(files_list))
                print(class_index)

        # print(len(frames_list))
        # print(len([class_index]*len(files_list)))

            

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels