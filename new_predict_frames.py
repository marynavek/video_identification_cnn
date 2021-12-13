import numpy as np
import pandas as pd
import tensorflow as tf
import os
from constrained_net import Constrained3DKernelMinimal
from contrained_net_PRNU_transfer import Constrained3DKernelMinimalPRNU
from data_generator import DataGenerator

class PredictFrames():
    
    def __init__(self, model_dir=None, results_dir=None, model_fname=None, constrained=True):
        self.model_dir = model_dir
        self.model_fname = model_fname
        self.result_dir = results_dir

        #Loading the model
        model_path = os.path.join(model_dir, model_fname)
        if constrained:
            # This is necessary to load custom objects, or in this constraints
            self.model = tf.keras.models.load_model(model_path, custom_objects={
                                                     'Constrained3DKernelMinimalPRNU': Constrained3DKernelMinimalPRNU})
        else:
            self.model = tf.keras.models.load_model(model_path)


    def start_predictions(self, test_dictionary):
        print("Starting predictions for "+ self.model_fname + " model.") 

        output_file = self.__get_output_file()

        return self.__predict_and_save(test_dictionary=test_dictionary, output_file=output_file)

    def __get_output_file(self):
        output_file = f"{self.model_fname.split('.')[0]}_F_predictions.csv"
        return os.path.join(self.result_dir, output_file)

    def __predict_and_save(self, test_dictionary, output_file):
        predict_ds_generator = DataGenerator(test_dictionary, shuffle=False, to_fit=False)

        #get actual labels and frames_names
        actual_encoded_labels, frames_list = self.__get_files_names_and_labels(test_dictionary)

        predictions = self.model.predict(predict_ds_generator)
        tf.losses.categorical_crossentropy
        # cce = tf.losses.categorical_crossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # cce_losses = cce(actual_encoded_labels, predictions).numpy()

        predicted_labels = [np.argmax(x) for x in predictions]
        actual_labels = [np.argmax(x) for x in actual_encoded_labels]
        # prediction_losses = [x for x in cce_losses]

        data_results = pd.DataFrame(list(zip(frames_list, actual_labels, predicted_labels)), columns=["File", "True Label", "Predicted Label"])
        
        data_results.to_csv(output_file, index=False)        
        return output_file

    
    def __get_files_names_and_labels(self, test_dictionary):

        label_key = "class_label"
        frame_path = "frame_path"
        print(test_dictionary)

        actual_encoded_list = []
        frames_list = []
        for d in test_dictionary:
            actual_encoded_list.append([val for key, val in d.items() if label_key == key])
            frames_list.append([val for key, val in d.items() if frame_path == key])

        # decoded_list_of_labels = self.__convert_hot_verctor_label_to_device_labels(actual_encoded_list)
        return actual_encoded_list, frames_list

    def __convert_hot_verctor_label_to_device_labels(self, list_of_labels):
        device_labels_list = list()
        for encoded_label in list_of_labels:
            for label, number in enumerate(encoded_label):
                if number == 1:
                    device_labels_list.append(label)
                    break
        
        return device_labels_list