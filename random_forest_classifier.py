from new_model import Constrained3DKernelMinimalTransfer
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from prepare_csv_dataset_singular import DataSetGeneratorSingular
from sklearn.metrics import accuracy_score

model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/reg_patches/sector_1/all/32/FC_Bayayr_model-sector_1/fm-e00060.h5"
dataset_path = "/Users/marynavek/Projects/files/equalized_patches_iframes_minimum_1st_sector_all"
saved_model = tf.keras.models.load_model(model_path, custom_objects={
                'Constrained3DKernelMinimalTransfer': Constrained3DKernelMinimalTransfer})

FC_layer_model = Model(inputs=saved_model.input,
                                 outputs=saved_model.layers[-5].output)


FC_layer_shape = 100

# generate dataset
data_factory = DataSetGeneratorSingular(input_dir_frames=dataset_path)
train_dataset_dict = data_factory.create_train_dataset()
# valid_dataset_dict = data_factory.create_validation_dataset()
test_dataset_dict = data_factory.create_test_dataset()

i=0

list_of_train_images = []
list_of_train_labels = []
for item in train_dataset_dict:
    list_of_train_images.append(item["frame_path"])
    list_of_train_labels.append(item["class_label"])

array_of_train_labels = np.array(list_of_train_labels)

list_of_test_images = []
list_of_test_labels = []
for item in test_dataset_dict:
    list_of_test_images.append(item["frame_path"])
    list_of_test_labels.append(item["class_label"])

array_of_test_labels = np.array(list_of_test_labels)

print("getting feature for the model")
features = np.zeros(shape=(len(list_of_train_images), FC_layer_shape))
for image_path in list_of_train_images:
    img= cv2.imread(image_path)
    img = np.expand_dims(img, axis=0)
    FC_output = FC_layer_model.predict(img)
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

features_test = np.zeros(shape=(len(list_of_test_images), FC_layer_shape))
i = 0
for image_path in list_of_test_images:
    img= cv2.imread(image_path)
    img = np.expand_dims(img, axis=0)
    FC_output = FC_layer_model.predict(img)
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

output_base = "/Users/marynavek/Projects/video_identification_cnn/bayar_models/noise/sector_1"
output_file_path_total = os.path.join(output_base, "predictions.csv")

accuracy=accuracy_score(predictions, array_of_test_labels)
predicted_labels = np.argmax(predictions, axis = 1)
actual_labels = np.argmax(array_of_test_labels, axis = 1)
data_results = pd.DataFrame(list(zip(list_of_test_images, actual_labels, predicted_labels)), columns=["File", "True Label", "Predicted Label"])
data_results.to_csv(output_file_path_total, index=False) 

print('Accuracy:', accuracy*100, '%.')





