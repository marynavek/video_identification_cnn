# from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
# import tensorflow_datasets as tfds
import tensorflow as tf
from efficient_net_audio import EfficientNetAudio
from prepare_efficient_net_audi_dataset import DataSetGeneratorAudioSecond
from efficient_net_model_image import EfficientNet
# from prnu_singular_net import PRNUModelNet
from prepate_csv_dataset_for_bayar import DataSetGeneratorBayar
# from keras import datasets

if __name__ == "__main__":

    dataset_path ="/Users/marynavek/Projects/files/audio_experiment/train_test_audio"
    
    data_factory = DataSetGeneratorAudioSecond(input_dir_patchs=dataset_path)
    num_classes = len(data_factory.get_class_names())
    print(data_factory.get_class_names())
    train_dataset_dict = data_factory.create_train_dataset()
    valid_dataset_dict = data_factory.create_validation_dataset()
    
    constr_net = EfficientNetAudio()
    
    constr_net.create_model(num_classes)
    # constr_net.load_model(model_path=model_path)
    # print(num_classes)
    constr_net.print_model_summary()
    
    # history = constr_net.train(train_ds=train_dataset_dict, val_ds_test=valid_dataset_dict, num_classes=num_classes)

