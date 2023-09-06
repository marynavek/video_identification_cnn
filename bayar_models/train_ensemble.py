import os
import tensorflow as tf
from bayar_singular_net import ConstrainedBayar
from data_generator_ensemble import DataGeneratorEnsemble
# from ensemble_net import EnsembleModelNet
from prepate_csv_dataset_for_bayar import DataSetGeneratorBayar
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc
from matplotlib import pyplot as plt
from numpy import dstack
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle

def get_files_and_labels(test_dictionary):
    label_key = "class_label"
    frame_path = "sector_1_patch"
    actual_encoded_list = [v for list in test_dictionary for k, v in list.items() if k == label_key]
    frames_list = [v for list in test_dictionary for k, v in list.items() if k == frame_path]
        # actual_encoded_list = [val[label_key] for val in test_dictionary.values() if label_key in val.keys()]
        # frames_list =  [val[frame_path] for val in test_dictionary.values() if frame_path in val.keys()]

        # decoded_list_of_labels = self.__convert_hot_verctor_label_to_device_labels(actual_encoded_list)
    return actual_encoded_list, frames_list

sector_1_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_1/fm-e00156.h5"
sector_2_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_2/fm-e00127.h5"
sector_3_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_3/fm-e00124.h5"
sector_4_model_path ="/Users/marynavek/Projects/video_identification_cnn/bayar_models/experiment_one/sector_4/fm-e00178.h5"

sector_1_ds_path = "/Users/marynavek/Projects/files/experiment_1_ten_devices_frames_1st_sector"
sector_2_ds_path = "/Users/marynavek/Projects/files/experiment_1_ten_devices_frames_2nd_sector"
sector_3_ds_path = "/Users/marynavek/Projects/files/experiment_1_ten_devices_frames_3rd_sector"
sector_4_ds_path = "/Users/marynavek/Projects/files/experiment_1_ten_devices_frames_4th_sector"

sector_1_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_1_.csv"
sector_2_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_2_.csv"
sector_3_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_3_.csv"
sector_4_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_4_.csv"

labels = []
sector_1_model = tf.keras.models.load_model(sector_1_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
sector_2_model = tf.keras.models.load_model(sector_2_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
sector_3_model = tf.keras.models.load_model(sector_3_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
sector_4_model = tf.keras.models.load_model(sector_4_model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})

data_factory = DataSetGeneratorBayar(input_dir_patchs=sector_1_ds_path)
test_ds = data_factory.create_test_ds_4_sectors(sector_1_ds_path, sector_2_ds_path, sector_3_ds_path, sector_4_ds_path)

actual_encoded_labels_test, frames_list_test = get_files_and_labels(test_ds)

# train_ds = data_factory.create_train_ds_4_sectors_after_selection(sector_1_ds_path, sector_2_ds_path, sector_3_ds_path, sector_4_ds_path, sector_1_selected_path, sector_2_selected_path, sector_3_selected_path, sector_4_selected_path)
train_ds = data_factory.create_train_ds_4_sectors(sector_1_ds_path, sector_2_ds_path, sector_3_ds_path, sector_4_ds_path)

actual_encoded_labels_train, frames_list_train = get_files_and_labels(train_ds)
num_classes = len(data_factory.get_class_names())

stackX = None
print("predict sector for training")
generator2_1 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_1 = sector_1_model.predict(generator2_1)
stackX = predictsSector_1

generator2_2 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_2 = sector_2_model.predict(generator2_2)
stackX = dstack((stackX, predictsSector_2))

generator2_3 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_3 = sector_3_model.predict(generator2_3)
stackX = dstack((stackX, predictsSector_3))

generator2_4 = DataGeneratorEnsemble(patches_path_dict=train_ds, sector=1, num_classes=num_classes,  to_fit=False, shuffle=False)
predictsSector_4 = sector_4_model.predict(generator2_4)
stackX = dstack((stackX, predictsSector_4))

array_difference1 = len(actual_encoded_labels_train) - len(predictsSector_1)
for i in range(array_difference1):
    actual_encoded_labels_train.pop()
true_labels1 = np.array(actual_encoded_labels_train)


# print(np.shape(true_labels))
stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
print(np.shape(stackX))

rf = RandomForestClassifier(n_estimators = 20, criterion="entropy",random_state = 42, max_features=4, max_depth=20)
rf.fit(stackX, true_labels1)

stackY = None
print("predict sector for predictions")
generator1 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=1,num_classes=num_classes,  to_fit=False, shuffle=False)
predictsSector_1 = sector_1_model.predict(generator1)
stackY = predictsSector_1

generator2 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=2, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_2 = sector_2_model.predict(generator2)
stackY = dstack((stackY, predictsSector_2))

generator3 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=3, num_classes=num_classes,  to_fit=False, shuffle=False)
predictsSector_3 = sector_3_model.predict(generator3)
stackY = dstack((stackY, predictsSector_3))

generator4 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=4, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_4 = sector_4_model.predict(generator4)
stackY = dstack((stackY, predictsSector_4))

array_difference = len(actual_encoded_labels_test) - len(predictsSector_1)
for i in range(array_difference):
    actual_encoded_labels_test.pop()
true_labels = np.array(actual_encoded_labels_test)

print(np.shape(stackY))
stackY = stackY.reshape((stackY.shape[0], stackY.shape[1]*stackY.shape[2]))
print(np.shape(stackY))
predictions = rf.predict(stackY)

actual_labels1 = np.argmax(true_labels, axis = 1)
print(np.shape(predictsSector_1), np.shape(true_labels))
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
acc = accuracy_score(predictions, true_labels)

print(f"accuracy score combined {acc}")

print("evaluate on train_ds\n")


output_base = "/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_one"
if not os.path.exists(output_base):
    os.makedirs(output_base)
output_file_path_total = os.path.join(output_base, "predictions_rf.csv")

predicted_labels = np.argmax(predictions, axis = 1)
actual_labels = np.argmax(true_labels, axis = 1)
data_results = pd.DataFrame(list(zip(frames_list_test, actual_labels, predicted_labels)), columns=["File", "True Label", "Predicted Label"])
data_results.to_csv(output_file_path_total, index=False) 

pred_prob = rf.predict_proba(stackY)

# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_classes = num_classes

fpr = dict()
tpr = dict()
roc_auc = dict()


for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# for i in range(n_classes):
#   fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predictions[:, i])
#   plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
#   print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", 'red', 'blueviolet', 'grey', 'black', "teal", "bisque"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# plt.figure()
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

# colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", 'red', 'blueviolet'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=2,
#         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#     )

# plt.plot([0, 1], [0, 1], "k--", lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Some extension of Receiver operating characteristic to multiclass")
# plt.legend(loc="lower right")
# plt.show()