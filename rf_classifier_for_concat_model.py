from contrained_net_PRNU_transfer import Constrained3DKernelMinimalPRNU
from new_model import Constrained3DKernelMinimalTransfer
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from prepare_csv_dataset_noise_patch import DataSetGeneratorNoisePatch
from sklearn.metrics import accuracy_score

model_path ="/Users/marynavek/Projects/video_identification_cnn/new_noise_models_concat_transfer_noise/ccnn-FC2x100-480_800-f3-k_s5/fm-e00060.h5"
noise_dataset = "/Users/marynavek/Projects/files/noise_patches_iframes_minimum_1st_sector_all"
reg_patches_dataset = "/Users/marynavek/Projects/files/patches_iframes_minimum_1st_sector_all"    
saved_model = tf.keras.models.load_model(model_path, custom_objects={
                'Constrained3DKernelMinimalPRNU': Constrained3DKernelMinimalPRNU})

FC_layer_model = Model(inputs=saved_model.input,
                                 outputs=saved_model.layers[-3].output)


FC_layer_shape = 100

# generate dataset
data_factory = DataSetGeneratorNoisePatch(input_dir_frames=noise_dataset,
                            input_dir_noiseprint=reg_patches_dataset)
train_dataset_dict = data_factory.create_train_dataset()
# valid_dataset_dict = data_factory.create_validation_dataset()
test_dataset_dict = data_factory.create_test_dataset()

i=0

list_of_train_patches = []
list_of_train_noise = []
list_of_train_labels = []
for item in train_dataset_dict:
    list_of_train_patches.append(item["frame_path"])
    list_of_train_labels.append(item["class_label"])
    list_of_train_noise.append(item["noise_path"])

array_of_train_labels = np.array(list_of_train_labels)

list_of_test_patches= []
list_of_test_noise= []
list_of_test_labels = []
for item in test_dataset_dict:
    list_of_test_patches.append(item["frame_path"])
    list_of_test_labels.append(item["class_label"])
    list_of_test_noise.append(item["noise_path"])

array_of_test_labels = np.array(list_of_test_labels)

print("getting feature for the model")
features = np.zeros(shape=(len(list_of_train_patches), FC_layer_shape))
for k, image_path in enumerate(list_of_train_patches):
    img= cv2.imread(image_path)
    img = np.expand_dims(img, axis=0)
    noise = cv2.imread(list_of_train_noise[k])
    noise = np.expand_dims(noise, axis=0)
    FC_output = FC_layer_model.predict([img, noise])
    features[i]=FC_output
    i+=1                           

#Save the features of the train images to use it in future.
np.save('features', features)

#Name the feature rows as f_0, f_1, f_2...
feature_col=[]
for i in range(FC_layer_shape):
    feature_col.append("f_"+str(i))
    i+=1
    
#Create DataFrame with features and coloumn name
train_features=pd.DataFrame(data=features,columns=feature_col)
feature_col = np.array(feature_col)

train_class = list(np.unique(array_of_train_labels))
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', array_of_train_labels.shape)
print(train_class)

print("start RF")
rf = RandomForestClassifier(n_estimators = 20, random_state = 42,max_features=4)

rf.fit(train_features, array_of_train_labels)

features_test = np.zeros(shape=(len(list_of_test_patches), FC_layer_shape))
i = 0
for k, image_path in enumerate(list_of_test_patches):
    img= cv2.imread(image_path)
    img = np.expand_dims(img, axis=0)
    noise = cv2.imread(list_of_test_noise[k])
    noise = np.expand_dims(noise, axis=0)
    FC_output = FC_layer_model.predict([img, noise])
    features_test[i]=FC_output
    i+=1   

#Create DataFrame with features and coloumn name for test
test_features=pd.DataFrame(data=features_test,columns=feature_col)
feature_col = np.array(feature_col)

print('Test Features Shape:', test_features.shape)
print('Test Labels Shape:', array_of_test_labels.shape)

print("predict RF")
#Feed the features of the test images to Random Forest Classifier to predict its class
predictions = rf.predict(test_features)

output_base = "/Users/marynavek/Projects/video_identification_cnn/new_noise_models_concat_transfer_noise/"
output_file_path_total = os.path.join(output_base, "predictions_transfer.csv")

accuracy=accuracy_score(predictions, array_of_test_labels)
predicted_labels = np.argmax(predictions, axis = 1)
actual_labels = np.argmax(array_of_test_labels, axis = 1)
data_results = pd.DataFrame(list(zip(list_of_test_patches, actual_labels, predicted_labels)), columns=["File", "True Label", "Predicted Label"])
data_results.to_csv(output_file_path_total, index=False) 

print('Accuracy:', accuracy*100, '%.')





