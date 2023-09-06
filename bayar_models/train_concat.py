# from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
# import tensorflow_datasets as tfds
import tensorflow as tf
# from prnu_singular_net import PRNUModelNet
from bayar_concat_prnu import BayarModelNet
from prepate_csv_dataset_for_bayar import DataSetGeneratorBayar
# from keras import datasets

if __name__ == "__main__":

    dataset_path ="/Users/marynavek/Projects/files/experiment_5/sector_2_patch"
    

    sector = "sector_2"
    patch_type = "experiment"
    
    data_factory = DataSetGeneratorBayar(input_dir_patchs=dataset_path)

    num_classes = len(data_factory.get_class_names())
    print(data_factory.get_class_names())
    train_dataset_dict = data_factory.create_train_dataset()
    valid_dataset_dict = data_factory.create_validation_dataset()
    
    constr_net = BayarModelNet(sector=sector, patch_type=patch_type)
    
    constr_net.create_model(num_classes)
    
    print(num_classes)
    constr_net.print_model_summary()
    
    history = constr_net.train(train_ds=train_dataset_dict, val_ds_test=valid_dataset_dict, num_classes=num_classes)

