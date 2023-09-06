# from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
# import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from concat_two_model_net import ConcatTwoModels
from constrained_net import ConstrainedNet
from contrained_net_PRNU_transfer import ConstrainedNetPRNU
from new_concat_model_net import ConcatModel
from prepare_csv_dataset_noise_patch import DataSetGeneratorNoisePatch
from preparing_csv_dataset import DataSetGenerator

from matplotlib import pyplot as plt
# from keras import datasets
parser = argparse.ArgumentParser(
    description='Train the constrained_net',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--fc_layers', type=int, default=2, required=False, help='Number of fully-connected layers [default: 2].')
parser.add_argument('--fc_size', type=int, default=1024, required=False, help='Number of neurons in Fully Connected layers')
parser.add_argument('--epochs', type=int, required=False, default=1, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size')
parser.add_argument('--model_name', type=str, required=False, help='Name for the model')
parser.add_argument('--model_path', type=str, required=False, help='Path to model to continue training (*.h5)')
parser.add_argument('--dataset', type=str, required=False, help='Path to dataset to train the constrained_net')
parser.add_argument('--height', type=int, required=False, default=480, help='Input Height [default: 480]')
parser.add_argument('--width', type=int, required=False, default=800, help='Width of CNN input dimension [default: 800]')
parser.add_argument('--constrained', type=int, required=False, default=1, help='Include constrained layer')

def run_locally():

    noise_dataset = "/Users/marynavek/Projects/files/new_equalized_noise_iframes_minimum_1st_sector_all"
    reg_patches_dataset = "/Users/marynavek/Projects/files/new_equalized_patches_iframes_minimum_1st_sector_all"

    return noise_dataset, reg_patches_dataset

if __name__ == "__main__":
    DEBUG = True

    # if DEBUG:
    noise_dataset, reg_patches_dataset = run_locally()
    
    #frames dataset creation
    data_factory = DataSetGeneratorNoisePatch(input_dir_patches=reg_patches_dataset,
                            input_dir_patches_noise=noise_dataset)

    num_classes = len(data_factory.get_class_names())
    
    train_dataset_dict = data_factory.create_train_dataset()
    valid_dataset_dict = data_factory.create_validation_dataset()

    lrate = 0.00001
    constr_net = ConcatModel(lrate = lrate, sector="sector_1")
    constr_net.create_model(lrate = lrate)
        # print(num_classes)
    constr_net.print_model_summary()
    history = constr_net.train(train_ds=train_dataset_dict, val_ds_test=valid_dataset_dict)
    

