import numpy as np
import pandas as pd
import tensorflow as tf
import os
from vcmi_data_gen import DataGeneratorVCMI
# from bayar_singular_net import ConstrainedBayar
# from vcmi_data_gen import DataGeneratorBayar
from vcmi_constrained_net import Constrained3DKernelMinimal



class PredictBayarPatches():
    
    def __init__(self, num_classes, model_dir=None, results_dir=None, model_fname=None,):
        self.model_dir = model_dir
        self.model_fname = model_fname
        self.result_dir = results_dir
        self.num_classes = num_classes

        #Loading the model
        model_path = os.path.join(model_dir, model_fname)
        self.model = tf.keras.models.load_model(model_path, custom_objects={
                                                     'Constrained3DKernelMinimal': Constrained3DKernelMinimal})


    def start_predictions(self, test_dictionary, batch_size=32):
        print("Starting predictions for "+ self.model_fname + " model.") 

        output_file = self.__get_output_file()

        return self.__predict_and_save(test_dictionary=test_dictionary, output_file=output_file,batch_size=batch_size)

    def __get_output_file(self):
        output_file = f"{self.model_fname.split('.')[0]}_F_predictions.csv"
        return os.path.join(self.result_dir, output_file)

    def __predict_and_save(self, test_dictionary, output_file, batch_size):
        predict_ds_generator = DataGeneratorVCMI(test_dictionary, num_classes=self.num_classes, batch_size=32, shuffle=False, to_fit=False)

        #get actual labels and frames_names
        actual_encoded_labels, frames_list = self.__get_files_names_and_labels(test_dictionary)
        predictions = self.model.predict(predict_ds_generator)
        
        predicted_labels = np.argmax(predictions, axis = 1)
        actual_labels = [np.argmax(x) for x in actual_encoded_labels]

        data_results = pd.DataFrame(list(zip(frames_list, actual_labels, predicted_labels)), columns=["File", "True Label", "Predicted Label"])
       
        data_results.to_csv(output_file, index=False)        
        return output_file

    
    def __get_files_names_and_labels(self, test_dictionary):

        label_key = "class_label"
        frame_path = "patch_path"
        actual_encoded_list = [v for list in test_dictionary for k, v in list.items() if k == label_key]
        frames_list = [v for list in test_dictionary for k, v in list.items() if k == frame_path]
        # actual_encoded_list = [val[label_key] for val in test_dictionary.values() if label_key in val.keys()]
        # frames_list =  [val[frame_path] for val in test_dictionary.values() if frame_path in val.keys()]

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

    