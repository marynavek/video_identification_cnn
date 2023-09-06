# from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
# import tensorflow_datasets as tfds
from prepare_efficient_net_audi_dataset import DataSetGeneratorAudioSecond
from hpc_efficient_net_model_audio import EfficientNetAudio
# from prnu_singular_net import PRNUModelNet
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
        
    # dataset_path ="/Users/marynavek/Projects/files/audio_experiment/train_test_audio"
    
    data_factory = DataSetGeneratorAudioSecond(input_dir_patchs=dataset_path)
    # model_path = "/Users/marynavek/Projects/video_identification_cnn/experiments_mov_models/cortiva/1/BayayrModel-1/fm-e00009.h5"
    data_factory.split_train_test_audio()

    train_dataset_data, train_labels = data_factory.create_train_dataset()
    valid_dataset_data, test_labels = data_factory.create_validation_dataset()
    
    constr_net = EfficientNetAudio(model_path=model_path, tensor_flow_path=tensor_flow_path)
    num_classes = len(data_factory.get_all_audio_names())
    
    constr_net.create_model(num_classes)
    # constr_net.load_model(model_path=model_path)
    # print(num_classes)
    constr_net.print_model_summary()
    
    history = constr_net.train(
            train_ds=train_dataset_data, train_labels=train_labels, val_ds_test=valid_dataset_data, test_labels=test_labels, num_classes=num_classes)

