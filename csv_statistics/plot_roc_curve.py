import argparse
import csv, os
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from utility import find_max_mode, get_prediction_files
from sklearn.metrics import auc, roc_curve, roc_auc_score

# parser = argparse.ArgumentParser(
#     description='Make predictions with signature network',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# parser.add_argument('--main_path', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
# parser.add_argument('--csvs_folder', type=str, required=True)

if __name__ == "__main__":
    # args = parser.parse_args()
    # main_path = args.main_path
    # csvs_folder = args.csvs_folder

    # path_to_csv = os.path.join(main_path, csvs_folder, "initial_stats")

    # previous_files = get_prediction_files(path_to_csv)

    # output_folder = os.path.join(main_path, csvs_folder)

    # if not os.path.exists(output_folder):
    #     os.mkdir(output_folder)

    # files_path = os.path.join(output_folder, "fpr_tpr_stats")
    # if not os.path.exists(files_path):
    #     os.mkdir(files_path)

# True Positive (TP): True label = A and predicted label = A
# False Positive (FP): True label != A and predicted label = A
# False Negative (FN): True label = A and predicted label != A
# True Negative (TN): True label != A and predicted label != A
# accuracy (ACC) = (TP + TN)/(TP + FN +TN +FP)
# false positive rate (FPR) = FP/(FP +TN)
# true positive rate (TPR) = TP/(TP+FN)
    # DEVICE_TYPES = [
    #     "01SamsungGalaxyS3",
    #     "02HuaweiP9",
    #     "03AppleiPhone5c",
    #     "04AppleiPhone6",
    #     "05HuaweiP9Lite", 
    #     "06AppleiPhone6Plus",
    #     "07SamsungGalaxyS5",
    #     "08AppleiPhone5",
    #     "09HuaweiP8",
    #     "10SamsungGalaxyS4Mini"
    # ]
    DEVICE_TYPES = [
        "01SamsungGalaxyS3",
        "02HuaweiP9",
        "03AppleiPhone5c",
        "04AppleiPhone6",
        "05HuaweiP9Lite", 
        "06AppleiPhone6Plus",
        "07SamsungGalaxyS5",
        "08AppleiPhone5",
        "09HuaweiP8",
        "10SamsungGalaxyS4Mini"
    ]

    
    with open("/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_5_1/predictions/predicted_label.csv") as file_name:
        predicted_labels = np.loadtxt(file_name, delimiter=",")
        with open("/Users/marynavek/Projects/video_identification_cnn/ensemble_results/experiment_5_1/predictions/true_label.csv") as file_pred:
            true_labels = np.loadtxt(file_pred, delimiter=",")
            # array_difference = len(true_labels) - len(predicted_labels)
            # print(np.shape(true_labels))
            # for i in range(array_difference):
            #     true_labels = np.delete(true_labels, -1, axis=0)
            # print(np.shape(true_labels))
            n_classes = 6
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            print(true_labels[:, 1])
            print(true_labels.shape)
            print(predicted_labels.shape)
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predicted_labels[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), predicted_labels.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            print(roc_auc["micro"])
            roc_auc_new = roc_auc_score(true_labels.ravel(), predicted_labels.ravel())
            print(f'ROC: {roc_auc_new}')
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
            print(roc_auc["macro"])
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