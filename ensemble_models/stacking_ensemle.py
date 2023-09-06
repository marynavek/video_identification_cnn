import numpy as np
import os
from numpy import dstack
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import csv

from bayar_net import ConstrainedBayar
from generator_ensemble import DataGeneratorEnsemble
from prepare_csv_dataset import DataSetGeneratorBayar
def get_files_and_labels(test_dictionary):
    label_key = "class_label"
    frame_path = "sector_1_patch"
    actual_encoded_list = [v for list in test_dictionary for k, v in list.items() if k == label_key]
    frames_list = [v for list in test_dictionary for k, v in list.items() if k == frame_path]
    return actual_encoded_list, frames_list

def string_to_array(label_str):
    label_str = label_str.replace("[", "")
    label_str = label_str.replace("]", "")
    label_str = label_str.replace("\n", "")
    label_str = label_str.split(" ")
    while("" in label_str):
        label_str.remove("")
    label = np.array(label_str)
    label = label.astype(np.float64)
    return label

val_predictions_file = ""


val_dict_1 = []
val_dict_2 = []
val_dict_3 = []
val_dict_4 = []
true_labels = []
frames_list = []

with open(val_predictions_file) as csv_file1:
    reader1 = csv.DictReader(csv_file1)
    for row in reader1:
        true_label = row["true_l"]
        true_label = string_to_array(true_label)
        file_name = row["file_name"]
        predicted_labels_s1 = row["predicted_l1"]
        predicted_labels_s1 = string_to_array(predicted_labels_s1)

        predicted_labels_s2 = row["predicted_l2"]
        predicted_labels_s2 = string_to_array(predicted_labels_s2)

        predicted_labels_s3 = row["predicted_l3"]
        predicted_labels_s3 = string_to_array(predicted_labels_s3)

        predicted_labels_s4 = row["predicted_l4"]
        predicted_labels_s4 = string_to_array(predicted_labels_s4)

        val_dict_1.append(predicted_labels_s1)
        val_dict_2.append(predicted_labels_s2)
        val_dict_3.append(predicted_labels_s3)
        val_dict_4.append(predicted_labels_s4)
        true_labels.append(true_label)
        frames_list.append(file_name)


stack_train = None
stack_train =  np.array(val_dict_1)
stack_train = dstack((stack_train, np.array(val_dict_2)))
stack_train = dstack((stack_train, np.array(val_dict_3)))
stack_train = dstack((stack_train, np.array(val_dict_4)))

stack_train = stack_train.reshape((stack_train.shape[0], stack_train.shape[1]*stack_train.shape[2]))

print("Start Training")

rf_classifier = RandomForestClassifier(n_estimators = 20, criterion="entropy",random_state = 42, max_features=4, max_depth=20)
rf_classifier.fit(stack_train, true_labels)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(stack_train, true_labels)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(stack_train, true_labels)

lr_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_classifier.fit(stack_train, true_labels)


###
print("Start Testing\n")

test_predictions_file = ""

test_dict_1 = []
test_dict_2 = []
test_dict_3 = []
test_dict_4 = []
true_labels_test = []
frames_list_test = []

with open(test_predictions_file) as csv_file2:
    reader2 = csv.DictReader(csv_file2)
    for row1 in reader2:
        true_label = row1["true_l"]
        true_label = string_to_array(true_label)
        file_name = row1["file_name"]
        predicted_labels_s1 = row1["predicted_l1"]
        predicted_labels_s1 = string_to_array(predicted_labels_s1)

        predicted_labels_s2 = row1["predicted_l2"]
        predicted_labels_s2 = string_to_array(predicted_labels_s2)

        predicted_labels_s3 = row1["predicted_l3"]
        predicted_labels_s3 = string_to_array(predicted_labels_s3)

        predicted_labels_s4 = row1["predicted_l4"]
        predicted_labels_s4 = string_to_array(predicted_labels_s4)

        test_dict_1.append(predicted_labels_s1)
        test_dict_2.append(predicted_labels_s2)
        test_dict_3.append(predicted_labels_s3)
        test_dict_4.append(predicted_labels_s4)
        true_labels_test.append(true_label)
        frames_list_test.append(file_name)


stack_test = None
stack_test =  np.array(test_dict_1)
stack_test = dstack((stack_test, np.array(test_dict_2)))
stack_test = dstack((stack_test, np.array(test_dict_3)))
stack_test = dstack((stack_test, np.array(test_dict_4)))


stack_test = stack_test.reshape((stack_test.shape[0], stack_test.shape[1]*stack_test.shape[2]))

rf_predictions = rf_classifier.predict(stack_test)

knn_predictions = knn_classifier.predict(stack_test)

dt_predictions = dt_classifier.predict(stack_test)

lr_predictions = lr_classifier.predict(stack_test)


actual_labels1 = np.argmax(true_labels, axis = 1)

print("evaluate on test_ds\n")

acc_rf = accuracy_score(rf_predictions, true_labels)
print(f"RF accuracy score  {acc_rf}")

acc_knn = accuracy_score(knn_predictions, true_labels)
print(f"KNN accuracy score  {acc_knn}")

acc_dt = accuracy_score(dt_predictions, true_labels)
print(f"DT accuracy score  {acc_dt}")

acc_lr = accuracy_score(lr_predictions, true_labels)
print(f"LR accuracy score  {acc_lr}")

output_base = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_one"
if not os.path.exists(output_base):
    os.makedirs(output_base)
output_file_path_rf = os.path.join(output_base, "predictions_rf.csv")
output_file_path_knn = os.path.join(output_base, "predictions_knn.csv")
output_file_path_dt = os.path.join(output_base, "predictions_dt.csv")
output_file_path_lr = os.path.join(output_base, "predictions_lr.csv")

actual_labels = np.argmax(true_labels, axis = 1)

predicted_labels_rf = np.argmax(rf_predictions, axis = 1)
predicted_labels_knn = np.argmax(knn_predictions, axis = 1)
predicted_labels_dt = np.argmax(dt_predictions, axis = 1)
predicted_labels_lr = np.argmax(lr_predictions, axis = 1)

data_results_rf = pd.DataFrame(list(zip(frames_list_test, actual_labels, predicted_labels_rf)), columns=["File", "True Label", "Predicted Label"])
data_results_knn = pd.DataFrame(list(zip(frames_list_test, actual_labels, predicted_labels_knn)), columns=["File", "True Label", "Predicted Label"])
data_results_dt = pd.DataFrame(list(zip(frames_list_test, actual_labels, predicted_labels_dt)), columns=["File", "True Label", "Predicted Label"])
data_results_lr = pd.DataFrame(list(zip(frames_list_test, actual_labels, predicted_labels_lr)), columns=["File", "True Label", "Predicted Label"])


data_results_rf.to_csv(output_file_path_rf, index=False) 
data_results_knn.to_csv(output_file_path_knn, index=False) 
data_results_dt.to_csv(output_file_path_dt, index=False) 
data_results_lr.to_csv(output_file_path_lr, index=False) 