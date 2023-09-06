import os
import tensorflow as tf
from tensorflow.keras.layers import concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K
from data_generator_noise_patch import DataGeneratorNoisePatch
# from tensorflow.python.keras.layers.merge import average
from new_model import Constrained3DKernelMinimalTransfer
from data_generator import DataGenerator


class ConcatTwoModels:
    def __init__(self, reg_patches_model_path, noise_patches_model_path, batches, sector):
        self.reg_patches_model_path = reg_patches_model_path
        self.noise_patches_model_path = noise_patches_model_path
        self.batches = batches
        self.sector = sector
        self.model_path = None
        self.model=None
        self.global_save_model_dir = self.__generate_model_path()
        self.global_tensorboard_dir = self.__generate_tensor_path()

    def __generate_model_name(self):
        model_name = f"FC_Bayayr_model_Transfer-{self.sector}"

        return model_name

    def __generate_model_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/two_concat_models"
        new_path = os.path.join(path_base,  self.sector)
        return new_path

    def __generate_tensor_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/two_concat_models"
        new_path = os.path.join(path_base,  self.sector)
        return new_path

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path, custom_objects={
                'Constrained3DKernelMinimalTransfer': Constrained3DKernelMinimalTransfer})
        return model

    def create_model(self):
        noise_branch = self.load_model(self.noise_patches_model_path)
        reg_patches_branch = self.load_model(self.reg_patches_model_path)
        # noise_branch.summary()
        # "removing layers"
        #flatten layer -6
        #1 dense -5
        for layer in noise_branch.layers:
            layer.trainable = False
            layer._name = layer.name + str("_noise")
        
        for layer in reg_patches_branch.layers:
            layer.trainable = False
            layer._name = layer.name + str("_patches")
        
        noise_cutoff_branch = Model(inputs=noise_branch.inputs, outputs=noise_branch.layers[-5].output)
        reg_patches_cutoff_branch = Model(inputs=reg_patches_branch.inputs, outputs=reg_patches_branch.layers[-5].output)

        merge_layer = concatenate([noise_cutoff_branch.layers[-1].output, reg_patches_cutoff_branch.layers[-1].output])

        denseLayer = Dense(100, activation=tf.keras.activations.tanh)(merge_layer)
        denseLayer2 = Dense(100, activation=tf.keras.activations.tanh)(denseLayer)
        finalDense = Dense(6, activation=tf.keras.activations.tanh)(denseLayer2)

        final_model = Model(inputs=(noise_cutoff_branch.inputs, reg_patches_cutoff_branch.inputs), outputs=finalDense)

        # opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95, decay=0.0005)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # final_model.compile(loss=tf.keras.losses.categorical_crossentropy,
        #               optimizer=opt,
        #               metrics=tf.keras.metrics.Accuracy())
        final_model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=['accuracy'])
                    #   loss=[None,None,whatever], loss_weights=[None,None,1.]
        
        # final_model.summary()
        model_name = self.__generate_model_name()

        self.model_name = model_name
        self.model = final_model

        return final_model

    def train(self, train_ds, val_ds_test):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()

        self.model.fit(DataGeneratorNoisePatch(train_ds),
                            epochs=20,
                            verbose=1,
                            initial_epoch=0,
                            validation_data=DataGeneratorNoisePatch(val_ds_test, shuffle=False),
                            callbacks=callbacks)
        
    def get_tensorboard_path(self):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save TensorBoard log-files.")

        # Create directory if not exists
        path = os.path.join(self.global_tensorboard_dir, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def print_model_summary(self):
        if self.model is None:
            print("Can't print model summary, self.model is None!")
        else:
            print(f"\nSummary of model:\n{self.model.summary()}")

    def get_save_model_path(self, file_name):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save checkpoints.")

        # Create directory if not exists
        path = os.path.join(self.global_save_model_dir, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Append file name and return
        return os.path.join(path, file_name)
        
    def get_callbacks(self):
        default_file_name = "fm-e{epoch:05d}.h5"
        save_model_path = self.get_save_model_path(default_file_name)

        # self.model.save(save_model_path)
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                         verbose=1,
                                                         save_weights_only=False,
                                                         period=1)

        tensorboard_cb = TensorBoard(log_dir=self.get_tensorboard_path())
        print_lr_cb = PrintLearningRate()

        return [save_model_cb, tensorboard_cb, print_lr_cb]

class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f"Learning rate on_epoch_end epoch {epoch}: {K.eval(lr_with_decay)}")