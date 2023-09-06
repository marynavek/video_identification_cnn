# from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
from ast import arg
# import tensorflow_datasets as tfds
import tensorflow as tf
from hpc_bayar_net_test_new import HPCBayarModelNet
from prepate_csv_dataset_for_bayar import DataSetGeneratorBayar
# from keras import datasets
parser = argparse.ArgumentParser(
    description='Train the constrained_net',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ds_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--tensor_flow_path', type=str, required=True)
parser.add_argument('--sector', type=str, required=True)
parser.add_argument('--patch_type', type=str, required=True)
parser.add_argument('--model_file_path', type=str, required=False)
parser.add_argument('--epoch_start', type=int, required=False)
if __name__ == "__main__":
    args = parser.parse_args()
    dataset_path = args.ds_path
    model_path = args.model_path
    tensor_flow_path = args.tensor_flow_path
    sector = args.sector
    patch_type = args.patch_type
    model_file_path = args.model_file_path
    epoch_start = args.epoch_start
    # dataset_path ="/Users/marynavek/Projects/files/experiment_1_mov_cleanup_2nd_sector"
    

    # sector = "sector_2"
    # patch_type = "experiment_1"
    # classes = [
    #     'D02_Apple_iPhone4s',
    #     'D04_Apple_iPhone5c',
    #     'D05_Apple_iPhone6',
    #     'D06_Apple_iPhone4']

#          'D08_Apple_iPhone5c' 'D09_Apple_iPhone6'
#  'D11_Apple_iPhone5c' 'D12_Apple_iPhone6Plus' 'D18_Apple_iPhone5'
    #frames dataset creation
    data_factory = DataSetGeneratorBayar(input_dir_patchs=dataset_path)

    num_classes = len(data_factory.get_class_names())
    
    train_dataset_dict = data_factory.create_train_dataset()
    valid_dataset_dict = data_factory.create_validation_dataset()
    
    constr_net = HPCBayarModelNet(sector=sector, patch_type=patch_type, model_path=model_path, tensor_flow_path=tensor_flow_path)
    
    constr_net.create_model(num_classes)
    # constr_net.load_model(model_path=model_file_path)
    
    print(num_classes)
    constr_net.print_model_summary()
    
    history = constr_net.train(train_ds=train_dataset_dict, val_ds_test=valid_dataset_dict, num_classes=num_classes)

