# from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
# import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from concat_two_model_net import ConcatTwoModels
from constrained_net import ConstrainedNet
from contrained_net_PRNU_transfer import ConstrainedNetPRNU
from prepare_csv_dataset_noise_patch import DataSetGeneratorNoisePatch
from preparing_csv_dataset import DataSetGenerator
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
    # fc_size = 1024
    # fc_layers = 2
    # n_epochs = 1
    # cnn_height = 480
    # cnn_width = 800
    # batch_size = 32
    # use_constrained_layer = True
    # model_path = "/Users/marynavek/Projects/Video_Project/models/ccnn-FC2x1024-480_800-f3-k_s5/fm-e00003.h5"
    
    noise_dataset = "/Users/marynavek/Projects/files/noise_patches_iframes_minimum_1st_sector_all"
    reg_patches_dataset = "/Users/marynavek/Projects/files/patches_iframes_minimum_1st_sector_all"
    reg_patches_model_path = "/Users/marynavek/Projects/video_identification_cnn/bayar_models/reg_patches/sector_1/all/32/FC_Bayayr_model-sector_1/fm-e00043.h5"
    noise_patches_model_path = "/Users/marynavek/Projects/video_identification_cnn/bayar_models/noise/sector_1/all/32/FC_Bayayr_model-sector_1/fm-e00054.h5"

    return noise_dataset, reg_patches_dataset, reg_patches_model_path, noise_patches_model_path

if __name__ == "__main__":
    DEBUG = True

    # if DEBUG:
    noise_dataset, reg_patches_dataset, reg_patches_model_path, noise_patches_model_path = run_locally()
    # else:
    #     args = parser.parse_args()
    #     fc_size = args.fc_size
    #     fc_layers = args.fc_layers
    #     n_epochs = args.epochs
    #     cnn_height = args.height
    #     cnn_width = args.width
    #     batch_size = args.batch_size
    #     use_constrained_layer = args.constrained == 1
    #     model_path = args.model_path
    #     model_name = args.model_name
    #     dataset_path = args.dataset_path
    #     dataset_path_prnu = args.dataset_path_prnu
    #     dataset_path_noiseprints = args.dataset_path_noiseprints

    #frames dataset creation
    data_factory = DataSetGeneratorNoisePatch(input_dir_frames=noise_dataset,
                            input_dir_noiseprint=reg_patches_dataset)

    num_classes = len(data_factory.get_class_names())
    
    train_dataset_dict = data_factory.create_train_dataset()
    valid_dataset_dict = data_factory.create_validation_dataset()


    constr_net = ConcatTwoModels(reg_patches_model_path=reg_patches_model_path, noise_patches_model_path=noise_patches_model_path, batches=32, sector="sector_1")
    constr_net.create_model()
    # print(num_classes)
    constr_net.print_model_summary()
    history = constr_net.train(train_ds=train_dataset_dict, val_ds_test=valid_dataset_dict)

