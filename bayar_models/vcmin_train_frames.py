# from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
# import tensorflow_datasets as tfds
import tensorflow as tf
from prepate_csv_dataset_for_bayar import DataSetGeneratorBayar

from vcmi_constrained_net import ConstrainedNet
# from keras import datasets

parser = argparse.ArgumentParser(
    description='Train the constrained_net',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ds_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--tensor_flow_path', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_path = args.ds_path
    model_path = args.model_path
    tensor_flow_path = args.tensor_flow_path
    # dataset_path = "/Users/marynavek/Projects/files/round_2_ten_devices_frames_ds_1"
    


    data_factory = DataSetGeneratorBayar(input_dir_patchs=dataset_path)

    num_classes = len(data_factory.get_class_names())
    
    train_dataset_dict = data_factory.create_train_dataset()
    valid_dataset_dict = data_factory.create_validation_dataset()
    print(len(train_dataset_dict))
    print(len(valid_dataset_dict))
    
    constr_net = ConstrainedNet(model_path_name=model_path, tensor_flow_path=tensor_flow_path)
    
    constr_net.create_model(num_classes)
    
    print(num_classes)
    constr_net.print_model_summary()
    
    history = constr_net.train(train_ds=train_dataset_dict, val_ds_test=valid_dataset_dict, num_classes=num_classes)

