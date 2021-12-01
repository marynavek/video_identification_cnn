 # from constrained_net.data.data_factory import DataFactory
# from constrained_net.constrained_net import ConstrainedNet
import argparse
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.models import Model
import tensorflow as tf

from constrained_net import Constrained3DKernelMinimal, ConstrainedNet
from data_factory import DataFactory

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
    fc_size = 1024
    fc_layers = 2
    n_epochs = 1
    cnn_height = 480
    cnn_width = 800
    batch_size = 32
    use_constrained_layer = True
    # model_path = "/Users/marynavek/Projects/Video_Project/models/ccnn-FC2x1024-480_800-f3-k_s5/fm-e00003.h5"
    model_path = None
    model_name = None
    dataset_path = "/Users/marynavek/Projects/Video_Project/dataset"
    dataset_path_prnu = "/Users/marynavek/Projects/Video_Project/prnu"

    return fc_size, fc_layers, n_epochs, cnn_height, cnn_width, batch_size, use_constrained_layer, model_path, model_name, dataset_path, dataset_path_prnu

if __name__ == "__main__":
    DEBUG = True

    if DEBUG:
        fc_size, fc_layers, n_epochs, cnn_height, cnn_width, batch_size, use_constrained_layer, model_path, model_name, dataset_path, dataset_path_prnu = run_locally()
    else:
        args = parser.parse_args()
        fc_size = args.fc_size
        fc_layers = args.fc_layers
        n_epochs = args.epochs
        cnn_height = args.height
        cnn_width = args.width
        batch_size = args.batch_size
        use_constrained_layer = args.constrained == 1
        model_path = args.model_path
        model_name = args.model_name
        dataset_path = args.dataset
        dataset_path_prnu = args.dataset_path_prnu


    #frames dataset creation
    data_factory_frames = DataFactory(input_dir=dataset_path,
                               batch_size=batch_size,
                               height=cnn_height,
                               width=cnn_width)

    num_classes_frames = len(data_factory_frames.get_class_names())
    train_ds_frames = data_factory_frames.get_tf_train_data()
    filename_ds_frames, test_ds_frames = data_factory_frames.get_tf_test_data()

    #PRNU dataset creation
    print("prnu dataset path")
    print(dataset_path_prnu)
    data_factory_prnu = DataFactory(input_dir=dataset_path_prnu,
                               batch_size=batch_size,
                               height=cnn_height,
                               width=cnn_width)

    num_classes_prnu = len(data_factory_prnu.get_class_names())
    train_ds_prnu = data_factory_prnu.get_tf_train_data()
    filename_ds_prnu, test_ds_prnu = data_factory_prnu.get_tf_test_data()

    #1.create PRNU model 
    prnu_model = ConstrainedNet(constrained_net=use_constrained_layer)
    # if model_path:
    #     prnu_model.set_model(model_path)
    # else:
    #     # Create new model
    prnu_model.create_model(num_classes_prnu, fc_layers, fc_size, cnn_height, cnn_width, model_name)

    prnu_model.print_model_summary()
    #2. train PRNU model
    #3. save PRNU model as base model
    history = prnu_model.train(train_ds=train_ds_prnu, val_ds=test_ds_prnu, epochs=n_epochs)
    
    prnu_model.summarize_model(history, train_ds=train_ds_prnu, val_ds=test_ds_prnu)

    prnu_model.save_trained_model('trained_prnu_model.h5')

    #4 loading pre-trained model
    
    # model = prnu_model.load_trained_model('trained_prnu_model.h5')
    # saved_model_path = '/Users/marynavek/Projects/Video_Project/models/ccnn-FC2x1024-480_800-f3-k_s5/fm-e00001.h5'
    # model = tf.keras.models.load_model(saved_model_path, custom_objects={
    #             'Constrained3DKernelMinimal': Constrained3DKernelMinimal})
    # tf.keras.models.load_model('/Users/marynavek/Projects/Video_Project/models/ccnn-FC2x1024-480_800-f3-k_s5/fm-e00001.h5')

    # base_model = model(include_top=False)

    frames_model = ConstrainedNet(model_name='transfer_model')
    frames_model.create_main_model_from_transfer(pre_trained_model_path='trained_prnu_model.h5', num_outputs=num_classes_frames, fc_size=fc_size)    
    frames_model.print_transfer_model_summary()
    history = frames_model.train_transfer_model(train_ds=train_ds_frames, val_ds=test_ds_frames, epochs=n_epochs)
    frames_model.summarize_model(history=history, train_ds=train_ds_frames, test_ds=test_ds_frames)

    frames_model.save_trained_model("trasnfer_model.h5")
