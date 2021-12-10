import enum
import numpy as np
import pathlib, csv
import os, random
from glob import glob


class DataSetGenerator:
    def __init__(self, input_dir_frames=None, input_dir_prnu=None, test_dir_suffix=""): 
        self.data_dir_frames = pathlib.Path(input_dir_frames)
        self.data_dir_prnu = pathlib.Path(input_dir_prnu)


        self.train_dir_frames = pathlib.Path(os.path.join(self.data_dir_frames, "train"))
        self.test_dir_frames = pathlib.Path(os.path.join(self.data_dir_frames, f"test{test_dir_suffix}"))
        self.train_dir_prnu = pathlib.Path(os.path.join(self.data_dir_prnu, "train"))
        self.test_dir_prnu = pathlib.Path(os.path.join(self.data_dir_prnu, f"test{test_dir_suffix}"))
        
        self.device_types = np.array([item.name for item in self.train_dir_frames.glob('*')])
        self.train_image_count = len(list(self.train_dir_frames.glob('*/*.jpg')))
        self.test_image_count = len(list(self.test_dir_frames.glob('*/*.jpg')))

        class_names = sorted(self.train_dir_frames.glob("*"))
        self.class_names = np.array([x.name for x in class_names])

    def get_class_names(self):
        return self.class_names

    def device_count(self):
        return len(self.device_types)

    def get_image_count(self, type="train"):
        if type == "train":
            return self.train_image_count
        else:
            return self.test_image_count

    def listdir_nonhidden(path):
        return [f for f in os.listdir(path) if not f.startswith('.')]

    def determine_label(self, file_path):
        classes = self.get_class_names()
        label_vector_lenght = self.device_count()
        label = np.zeros((label_vector_lenght,), dtype=int)
        classes.sort()
        for i, class_name in enumerate(classes):
            if class_name in file_path:
                label[i] = 1
        return label

    def create_train_dataset(self):

        train_input_frames_file_names = np.array(glob(str(self.train_dir_frames) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()

        random.shuffle(train_input_frames_file_names)

        for i, file_path in enumerate(train_input_frames_file_names):
            # if i <10:
            prnu_path = self.get_prnu_file_train(file_path)
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "frame_path": file_path, "class_label": class_label, "prnu_path": prnu_path}
                
                # print(ds_row)
            labeled_dictionary.append(ds_row)

        with open("train_dataset_images.csv", "w") as file:
            keys = ["item_ID", "frame_path", "class_label", "prnu_path"]
            csvwriter = csv.DictWriter(file, keys)
            csvwriter.writeheader()
            csvwriter.writerows(labeled_dictionary)
        return labeled_dictionary

    def get_prnu_file_train(self, file_path):
        train_input_prnu_file_names = np.array(glob(str(self.train_dir_prnu) + "/**/*.jpg", recursive = True))
        classes = self.get_class_names()
        for prnu_frame_path in train_input_prnu_file_names:
            for device in classes:
                if device in prnu_frame_path and device in file_path:
                    return prnu_frame_path

    def create_validation_dataset(self):

        validation_input_frames_file_names = np.array(glob(str(self.test_dir_frames) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()

        random.shuffle(validation_input_frames_file_names)

        for i, file_path in enumerate(validation_input_frames_file_names):
            prnu_path = self.get_prnu_file_validation(file_path)
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "frame_path": file_path, "class_label": class_label, "prnu_path": prnu_path}
            labeled_dictionary.append(ds_row)

        with open("validation_dataset_images.csv", "w") as file:
            keys = ["item_ID", "frame_path", "class_label", "prnu_path"]
            csvwriter = csv.DictWriter(file, keys)
            csvwriter.writeheader()
            csvwriter.writerows(labeled_dictionary)
        return labeled_dictionary

    def get_prnu_file_validation(self, file_path):
        validation_input_prnu_file_names = np.array(glob(str(self.test_dir_prnu) + "/**/*.jpg", recursive = True))
        classes = self.get_class_names()
        
        for prnu_frame_path in validation_input_prnu_file_names:
            for device in classes:
                if device in prnu_frame_path and device in file_path:
                    return prnu_frame_path
