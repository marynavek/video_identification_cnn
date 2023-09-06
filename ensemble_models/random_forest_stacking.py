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

from bayar_net import ConstrainedBayar
from generator_ensemble import DataGeneratorEnsemble
from prepare_csv_dataset import DataSetGeneratorBayar
def get_files_and_labels(test_dictionary):
    label_key = "class_label"
    frame_path = "sector_1_patch"
    actual_encoded_list = [v for list in test_dictionary for k, v in list.items() if k == label_key]
    frames_list = [v for list in test_dictionary for k, v in list.items() if k == frame_path]
    return actual_encoded_list, frames_list

sector_1_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_1/fm-e00156.h5"
sector_2_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_2/fm-e00127.h5"
sector_3_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_3/fm-e00124.h5"
sector_4_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_4/fm-e00178.h5"

sector_1_ds_path = "/Users/marynavek/Projects/files/experiment_STD_15_AP_O/sector_1_patch"
sector_2_ds_path = "/Users/marynavek/Projects/files/experiment_STD_15_AP_O/sector_2_patch"
sector_3_ds_path = "/Users/marynavek/Projects/files/experiment_STD_15_AP_O/sector_3_patch"
sector_4_ds_path = "/Users/marynavek/Projects/files/experiment_STD_15_AP_O/sector_4_patch"


sector_1_model = tf.keras.models.load_model(sector_1_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
sector_2_model = tf.keras.models.load_model(sector_2_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
sector_3_model = tf.keras.models.load_model(sector_3_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
sector_4_model = tf.keras.models.load_model(sector_4_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})

data_factory = DataSetGeneratorBayar(input_dir_patchs=sector_1_ds_path)
test_ds = data_factory.create_test_ds_4_sectors_after_selection(sector_1_ds_path, sector_2_ds_path, sector_3_ds_path, sector_4_ds_path)

actual_encoded_labels_test, frames_list_test = get_files_and_labels(test_ds)      

train_ds = data_factory.create_train_ds_4_sectors(sector_1_ds_path, sector_2_ds_path, sector_3_ds_path, sector_4_ds_path)

actual_encoded_labels_train, frames_list_train = get_files_and_labels(train_ds)
num_classes = len(data_factory.get_class_names())


stack_train = None

generator2_1 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_1 = sector_1_model.predict(generator2_1)

stack_train = predictsSector_1

generator2_2 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_2 = sector_2_model.predict(generator2_2)

stack_train = dstack((stack_train, predictsSector_2))

generator2_3 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_3 = sector_3_model.predict(generator2_3)

stack_train = dstack((stack_train, predictsSector_3))

generator2_4 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes,  to_fit=False, shuffle=False)
predictsSector_4 = sector_4_model.predict(generator2_4)

stack_train = dstack((stack_train, predictsSector_4))

array_difference1 = len(actual_encoded_labels_train) - len(predictsSector_1)
for i in range(array_difference1):
    actual_encoded_labels_train.pop()
true_labels1 = np.array(actual_encoded_labels_train)

stack_train = stack_train.reshape((stack_train.shape[0], stack_train.shape[1]*stack_train.shape[2]))


# weak_learners = [('DT', DecisionTreeClassifier()),
#                     ('KNN', KNeighborsClassifier()),
#                     ('RF', RandomForestClassifier()),
#                     ('NB', GaussianNB()),
#                     ('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs'))]



rf_classifier = RandomForestClassifier(n_estimators = 20, criterion="entropy",random_state = 42, max_features=4, max_depth=20)
rf_classifier.fit(stack_train, true_labels1)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(stack_train, true_labels1)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(stack_train, true_labels1)

lr_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_classifier.fit(stack_train, true_labels1)

# rf_classifier = RandomForestClassifier(n_estimators = 20, criterion="entropy",random_state = 42, max_features=4, max_depth=20)
# rf_classifier.fit(stack_train, true_labels1)


stack_test = None

generator1 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=1,num_classes=num_classes,  to_fit=False, shuffle=False)
predictsSector_1 = sector_1_model.predict(generator1)

stack_test = predictsSector_1

generator2 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=2, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_2 = sector_2_model.predict(generator2)

stack_test = dstack((stack_test, predictsSector_2))

generator3 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=3, num_classes=num_classes,  to_fit=False, shuffle=False)
predictsSector_3 = sector_3_model.predict(generator3)

stack_test = dstack((stack_test, predictsSector_3))

generator4 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=4, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_4 = sector_4_model.predict(generator4)

stack_test = dstack((stack_test, predictsSector_4))

array_difference = len(actual_encoded_labels_test) - len(predictsSector_1)
for i in range(array_difference):
    actual_encoded_labels_test.pop()
true_labels = np.array(actual_encoded_labels_test)


stack_test = stack_test.reshape((stack_test.shape[0], stack_test.shape[1]*stack_test.shape[2]))

rf_predictions = rf_classifier.predict(stack_test)

knn_predictions = knn_classifier.predict(stack_test)

dt_predictions = dt_classifier.predict(stack_test)

lr_predictions = lr_classifier.predict(stack_test)


actual_labels1 = np.argmax(true_labels, axis = 1)

predicted1 = np.argmax(predictsSector_1, axis = 1)
acc1 = accuracy_score(predicted1, actual_labels1)
print(f"accuracy score sector_1 {acc1}")

predicted2 = np.argmax(predictsSector_2, axis = 1)
acc2 = accuracy_score(predicted2, actual_labels1)
print(f"accuracy score sector_2 {acc2}")

predicted3 = np.argmax(predictsSector_3, axis = 1)
acc3 = accuracy_score(predicted3, actual_labels1)
print(f"accuracy score sector_3 {acc3}")

predicted4 = np.argmax(predictsSector_4, axis = 1)
acc4 = accuracy_score(predicted4, actual_labels1)
print(f"accuracy score sector_4 {acc4}")

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