import numpy as np
import pathlib
import os, random
from glob import glob


class DataSetGeneratorAudioSecond:
    def __init__(self, input_dir_patchs=None, test_dir_suffix="", classes=None): 


        self.audio_main_folder = input_dir_patchs
        # self.data_dir_patchs = pathlib.Path(input_dir_patchs)
        self.train_dir_patchs = pathlib.Path(os.path.join(self.audio_main_folder, "train"))
        self.test_dir_patchs = pathlib.Path(os.path.join(self.audio_main_folder, f"test{test_dir_suffix}"))

        self.class_names = self.get_classes(classes)
        print(self.class_names)

    def get_classes(self, classes):
        class_names = sorted(pathlib.Path(self.train_dir_patchs).glob("*"))
        return np.array([x.name for x in class_names if not x.name.startswith('.')])            

    def get_class_names(self):
        return self.class_names

    def device_count(self):
        return len(self.class_names)

    def get_image_count(self, type="train"):
        if type == "train":
            return self.train_image_count
        else:
            return self.test_image_count

    def listdir_nonhidden(self, path):
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

        train_input_patchs_file_names = np.array(glob(str(self.train_dir_patchs) + "/**/*.txt", recursive = True))
        labeled_dictionary = list()

        sorted_files = []
        for file in train_input_patchs_file_names:
            if "_flat_" in file or "_indoor_" in file or "_outdoor_" in file:
                sorted_files.append(file)

        random.shuffle(sorted_files)
        
        # random.shuffle(train_input_patchs_file_names)
        for i, file_path in enumerate(sorted_files):
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "patch_path": file_path, "class_label": class_label}                        
            labeled_dictionary.append(ds_row)

        return labeled_dictionary

    def create_validation_dataset(self):

        validation_input_patchs_file_names = np.array(glob(str(self.test_dir_patchs) + "/**/*.txt", recursive = True))
        labeled_dictionary = list()

        sorted_files = []
        for file in validation_input_patchs_file_names:
            if "_flat_" in file or "_indoor_" in file or "_outdoor_" in file:
                sorted_files.append(file)

        random.shuffle(sorted_files)

        for i, file_path in enumerate(sorted_files):
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "patch_path": file_path, "class_label": class_label}
            labeled_dictionary.append(ds_row)

        return labeled_dictionary

