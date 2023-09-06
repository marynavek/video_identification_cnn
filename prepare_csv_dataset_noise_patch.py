import enum
import numpy as np
import pathlib, csv
import os, random
from glob import glob


class DataSetGeneratorNoisePatch:
    def __init__(self, input_dir_patches=None, input_dir_patches_noise=None,  test_dir_suffix=""): 
        self.data_dir_patches = pathlib.Path(input_dir_patches)
        self.data_dir_patches_noises = pathlib.Path(input_dir_patches_noise)

        self.train_dir_patches = pathlib.Path(os.path.join(self.data_dir_patches, "train"))
        self.test_dir_patches = pathlib.Path(os.path.join(self.data_dir_patches, f"test{test_dir_suffix}"))
        self.train_dir_patches_noises = pathlib.Path(os.path.join(self.data_dir_patches_noises, "train"))
        self.test_dir_patches_noises= pathlib.Path(os.path.join(self.data_dir_patches_noises, f"test{test_dir_suffix}"))
        
        self.device_types = np.array([item.name for item in self.train_dir_patches.glob('*') if not item.name.startswith('.')])
        self.train_image_count = len(list(self.train_dir_patches.glob('*/*.jpg')))
        self.test_image_count = len(list(self.test_dir_patches.glob('*/*.jpg')))

        class_names = sorted(self.train_dir_patches.glob("*"))
        self.class_names = np.array([x.name for x in class_names if not x.name.startswith('.')])
        print(self.class_names)

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

        train_input_patches_file_names = np.array(glob(str(self.train_dir_patches) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()
        random.shuffle(train_input_patches_file_names)
        counter = 0
        for i, file_path in enumerate(train_input_patches_file_names):
            # if i <10:
            noise_path = self.get_patches_noise_file_name_train(file_path)
            if len(noise_path) < 1:
                continue
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": counter, "patch_path": file_path, "class_label": class_label, "noise_path": noise_path}
                
                # print(ds_row)
            labeled_dictionary.append(ds_row)
            counter += 1

        with open("train_dataset_images_transfer.csv", "w") as file:
            keys = ["item_ID", "patch_path", "class_label", "noise_path"]
            csvwriter = csv.DictWriter(file, keys)
            csvwriter.writeheader()
            csvwriter.writerows(labeled_dictionary)
        return labeled_dictionary

    def get_patches_noise_file_name_train(self, file_path):
        file_path, file_name = os.path.split(file_path)
        remove_part, patch_name = file_name.split("_V_")
        classes = self.get_class_names()
        for device in classes:
            if device in file_path:
                device_path_name = device
                break
        train_input_patches_file_names_noise = np.array(glob(str(os.path.join(self.train_dir_patches_noises,device_path_name)) + "/*.jpg", recursive = True))
        noise_file = ""
        for noise_file_path in train_input_patches_file_names_noise:
            if patch_name in noise_file_path:
                noise_file = noise_file_path
                break

        noise_print_path = noise_file
        return noise_print_path

    def create_validation_dataset(self):

        validation_input_patches_file_names = np.array(glob(str(self.test_dir_patches) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()

        random.shuffle(validation_input_patches_file_names)
        counter = 0
        for i, file_path in enumerate(validation_input_patches_file_names):

            noise_path = self.get_patches_noise_file_name_validation(file_path)
            if len(noise_path) < 1:
                continue
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": counter, "patch_path": file_path, "class_label": class_label, "noise_path": noise_path}
            labeled_dictionary.append(ds_row)
            counter += 1

        with open("validation_dataset_images_transfer.csv", "w") as file:
            keys = ["item_ID", "patch_path", "class_label", "noise_path"]
            csvwriter = csv.DictWriter(file, keys)
            csvwriter.writeheader()
            csvwriter.writerows(labeled_dictionary)
        return labeled_dictionary

    def get_patches_noise_file_name_validation(self, file_path):
        file_path, file_name = os.path.split(file_path)
        remove_part, patch_name = file_name.split("_V_")
        classes = self.get_class_names()
        for device in classes:
            if device in file_path:
                device_path_name = device
                break
        validation_input_patches_file_names = np.array(glob(str(os.path.join(self.test_dir_patches_noises,device_path_name)) + "/**/*.jpg", recursive = True))
        noise_file = ""
        for noise_file_path in validation_input_patches_file_names:
            if patch_name in noise_file_path:
                noise_file = noise_file_path
                break
        noise_print_path = noise_file
        return noise_print_path

    def create_test_dataset(self):

        validation_input_patches_file_names = np.array(glob(str(self.test_dir_patches) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()

        random.shuffle(validation_input_patches_file_names)
        counter = 0
        for i, file_path in enumerate(validation_input_patches_file_names):
            noise_path = self.get_patches_noise_file_name_test(file_path)
            if len(noise_path) < 1:
                continue
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": counter, "patch_path": file_path, "class_label": class_label, "noise_path": noise_path}
            labeled_dictionary.append(ds_row)
            counter += 1

        with open("test_dataset_images_transfer.csv", "w") as file:
            keys = ["item_ID", "patch_path", "class_label", "noise_path"]
            csvwriter = csv.DictWriter(file, keys)
            csvwriter.writeheader()
            csvwriter.writerows(labeled_dictionary)
        return labeled_dictionary

    def get_patches_noise_file_name_test(self, file_path):
        file_path, file_name = os.path.split(file_path)
        remove_part, patch_name = file_name.split("_V_")
        classes = self.get_class_names()
        for device in classes:
            if device in file_path:
                device_path_name = device
                break
        validation_input_patches_file_names = np.array(glob(str(os.path.join(self.test_dir_patches_noises,device_path_name)) + "/**/*.jpg", recursive = True))
        noise_file = ""
        for noise_file_path in validation_input_patches_file_names:
            # print(patch_name)
            # print(noise_file_path)
            if patch_name in noise_file_path:
                # print('hi there')
                noise_file = noise_file_path
                break

        noise_print_path = noise_file
        return noise_print_path
