import os
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.models import Model
from data_generator_singular import DataGeneratorSingular
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K
from keras.constraints import Constraint

import numpy as np 
class Constrained3DKernelMinimalTransfer(Constraint):
    def __call__(self, w):
        """
                This custom weight constraint implements the constrained convolutional layer for RGB-images.
                However, this is not as straightforward as it should be. Since TensorFlow prehibits us from
                assigning new values directly to the weight matrix, we need a trick to update its values.
                This trick consists of multiplying the weight matrix by so-called 'mask matrices'
                to get the desired results.

                For example, if we want to set the center values of the weight matrix to zero, we first create a
                mask matrix of the same size consisting of only ones, except at the center cells. The center cells
                will have as value one. After multiplying the weight matrix with this mask matrix,
                we obtain a 'new' weight matrix, where the center values are set to zero but with the remaining values
                untouched.

                More information about this problem:
                #https://github.com/tensorflow/tensorflow/issues/14132

                The incoming weight matrix 'w' is a 4D array of shape (x, y, z, n_filters) where (normally):
                x = 5
                y = 5
                z = 3, since we're using RGB images
                n_filters = 3

                This means there are 3 filters in total with each filter being 3-dimensional.

                This module includes the following (main) steps:
                Recall that each 3D-filter has three xy-planes (across the z-direction).
                For each (3D-)filter:
                    1. For each of the three xy-planes, set its center value to zero.
                    2. For each of the three xy-planes, normalize its values such that the sum adds up to 1
                    3. For each of the three xy-planes, set its center value to -1

               """
        w_original_shape = w.shape
        w = w * 10000  # scale by 10k to prevent numerical issues

        # 1. Reshaping of 'w'
        x, y, z, n_kernels = w_original_shape[0], w_original_shape[1], w_original_shape[2], w_original_shape[3]
        center = x // 2 # Determine the center cell on the xy-plane.
        new_shape = [n_kernels, z, y, x]
        w = tf.reshape(w, new_shape)

        # 2. Set center values of 'w' to zero by multiplying 'w' with mask-matrix
        center_zero_mask = np.ones(new_shape)
        center_zero_mask[:, :, center, center] = 0
        w *= center_zero_mask

        # 3. Normalize values w.r.t xy-planes
        xy_plane_sum = tf.reduce_sum(w, [2, 3], keepdims=True)  # Recall new shape of w: (n_kernels, z, y, x).
        w = tf.math.divide(w, xy_plane_sum)  # Divide each element by its corresponding xy-plane sum-value

        # 4. Set center values of 'w' to negative one by subtracting mask-matrix from 'w'
        center_one_mask = np.zeros(new_shape)
        center_one_mask[:, :, center, center] = 1
        w = tf.math.subtract(w, center_one_mask)

        # Reshape 'w' to original shape and return
        return tf.reshape(w, w_original_shape)

    def get_config(self):
        return {}

class BayarModel:

    def __init__(self, type_of_patches, data_name, model_path=None, constrained_net=True):
        self.use_TensorBoard = True
        self.model = None
        self.type_of_patches = type_of_patches
        self.batches = 32
        self.model_path = None
        if model_path is not None:
            self.set_model(model_path)

        self.verbose = False
        self.model_name = None
        self.data_name = data_name
        self.global_save_model_dir = self.__generate_model_path()
        self.global_tensorboard_dir = self.__generate_tensor_path()

    def __generate_model_name(self):
        model_name = f"FC_Bayayr_model-{self.data_name}"

        return model_name

    def __generate_model_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/bayar_models"
        new_path = os.path.join(path_base, self.type_of_patches, self.data_name)
        return new_path

    def __generate_tensor_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/bayar_tensors"
        new_path = os.path.join(path_base, self.type_of_patches, self.data_name)
        return new_path
# The callback to be applied at the end of each iteration. This is 
# used to constrain the layer's weights the same way Bayar and Stamm do
# at their paper. It's the core of the method, and the number one source
# for bugs and logic flaws.

    def create_model(self, model_name=None):
        input_shape = (128, 128, 3)

        input_layer = Input(shape=input_shape)


        # (5,5), 2*(3,3), (1,1)
        constrained_conv_layer = Conv2D(filters=3,
                                kernel_size=(5,5),
                                strides=(1, 1),
                                padding="valid", # Intentionally
                                kernel_constraint=Constrained3DKernelMinimalTransfer(),
                                name="constrained_layer")(input_layer)

        conv2d_1 = Conv2D(96, (5,5), strides=(2,2),padding='same')(constrained_conv_layer)
        batch_norm1 = BatchNormalization()(conv2d_1)
        activation1 = Activation(tf.keras.activations.tanh)(batch_norm1)
        max_pool1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation1)

        conv2d_2 = Conv2D(64, (3,3), strides=(1,1),padding='same')(max_pool1)
        batch_norm2 = BatchNormalization()(conv2d_2)
        activation2 = Activation(tf.keras.activations.tanh)(batch_norm2)
        max_pool2 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation2)

        conv2d_4 = Conv2D(64, (3,3), strides=(1,1),padding='same')(max_pool2)
        batch_norm4 = BatchNormalization()(conv2d_4)
        activation4 = Activation(tf.keras.activations.tanh)(batch_norm4)
        max_pool4 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation4)

        conv2d_3 = Conv2D(128, (1,1), strides=1,padding='same')(max_pool4)
        batch_norm3 = BatchNormalization()(conv2d_3)
        activation3 = Activation(tf.keras.activations.tanh)(batch_norm3)
        max_pool3 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(activation3)

        flatten = Flatten()(max_pool3)

        dense1 = Dense(100)(flatten)
        activation4 = Activation(tf.keras.activations.tanh)(dense1)

        dense2 = Dense(100)(activation4)
        activation4 = Activation(tf.keras.activations.tanh)(dense2)

        final_dense = Dense(6, activation=tf.keras.activations.softmax)(activation4)

        model = Model(input_layer, final_dense)

        model.summary()

        # sgd = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.95, decay=0.0005)
        # model.compile(
        #     optimizer=sgd, 
        #     loss='categorical_crossentropy', 
        #     metrics=['accuracy'])
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=['accuracy'])

        model_name = self.__generate_model_name()

        self.model_name = model_name
        self.model = model
        
        return model

    def train(self, train_ds, val_ds_test):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()
        
        self.model.fit(DataGeneratorSingular(train_ds, batch_size=self.batches),
                       epochs=60,
                       initial_epoch=0,
                       validation_data=DataGeneratorSingular(val_ds_test, batch_size=self.batches, shuffle=False),
                       callbacks=callbacks,
                       workers=12,
                       use_multiprocessing=True)

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