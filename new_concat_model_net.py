import tensorflow as tf
import numpy as np
import os, math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras.constraints import Constraint
from keras import backend as K
import tensorflow as tf



from data_generator import DataGenerator
from data_generator_noise_patch import DataGeneratorNoisePatch

class Constrained3DKernelMinimalPRNU(Constraint):
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


class ConcatModel:
    GLOBAL_SAVE_MODEL_DIR = "/Users/marynavek/Projects/video_identification_cnn/models_concat_transfer_noise/"
    GLOBAL_TENSORBOARD_DIR = "/Users/marynavek/Projects/video_identification_cnn/tensorboard_concat_transfer_noise/"

    def __init__(self, lrate, sector, model_path=None, constrained_net=True):
        self.use_TensorBoard = True
        self.model = None
        self.model_path = None
        self.sector = sector
        if model_path is not None:
            self.set_model(model_path)

        self.verbose = False
        self.model_name = None
        self.lrate = lrate

        # Constrained layer properties
        self.constrained_net = constrained_net
        self.constrained_n_filters = 3
        self.constrained_kernel_size = 5

    def set_constrained_params(self, n_filters=None, kernel_size=None):
        print(f"Setting constrained params: number of filters={n_filters}, kernel size={kernel_size}")
        if n_filters is not None:
            self.constrained_n_filters = n_filters
        if kernel_size is not None:
            self.constrained_kernel_size = kernel_size

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, value):
        self.model_name = value

    def get_model(self):
        return self.model

    def set_model(self, model_path):
        # Path is e.g. ~/constrained_net/fm-e00001.h5
        path_splits = model_path.split(os.sep)
        model_name = path_splits[-2]
        print(model_name)
        self.model_path = model_path
        self.model_name = model_name

        self.model = tf.keras.models.load_model(model_path, custom_objects={
                'Constrained3DKernelMinimalPRNU': Constrained3DKernelMinimalPRNU})
        # else:

        if self.model is None:
            raise ValueError(f"Model could not be loaded from location {model_path}")

    def set_useTensorBoard(self, value): self.use_TensorBoard = value
    def set_verbose(self, value): self.verbose = value

    def __generate_model_name(self, fc_layers, fc_size, cnn_height, cnn_width):
        model_name = f"FC{fc_layers}x{fc_size}-{cnn_height}_{cnn_width}"

        if self.constrained_net:
            n_filters = self.constrained_n_filters
            kernel_s = self.constrained_kernel_size
            model_name = f"ccnn-{model_name}-f{n_filters}-k_s{kernel_s}"
        else:
            model_name = f"cnn-{model_name}"

        return model_name

    def create_model(self,lrate,height=480, width=800, model_name=None):

        # input_shape = (height, width, 3)

        input_patches = (128, 128, 3)
        input_layer_patch_noise = Input(shape=input_patches)
        input_layer_patches = Input(shape=input_patches)
        # input_layer_patches = Input(dtype=input_shape)
        
        cons_layer_patch_noise = Conv2D(filters=3,
                                kernel_size=(5,5),
                                strides=(1, 1),
                                input_shape=input_patches,
                                padding="valid", # Intentionally
                                kernel_constraint=Constrained3DKernelMinimalPRNU(),
                                name="constrained_layer_patch_noise")(input_layer_patch_noise)

        
        cons_layer_patches = Conv2D(filters=self.constrained_n_filters,
                                kernel_size=self.constrained_kernel_size,
                                strides=(1, 1),
                                input_shape=input_patches,
                                padding="valid", # Intentionally
                                kernel_constraint=Constrained3DKernelMinimalPRNU(),
                                name="constrained_layer_patches")(input_layer_patches)

        # Determine whether to use the input shape parameter
        conv2d_patch_noise1 = Conv2D(filters=96, kernel_size=(5, 5), strides=(2, 2), padding="same")(cons_layer_patch_noise)
        batch1_patch_noise = BatchNormalization()(conv2d_patch_noise1)
        activation1_patch_noise = Activation(tf.keras.activations.tanh)(batch1_patch_noise)
        max_pool1_patch_noise = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation1_patch_noise)


        con2d_2_patch_noise = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(max_pool1_patch_noise)
        batch2_patch_noise = BatchNormalization()(con2d_2_patch_noise)
        activation2_patch_noise = Activation(tf.keras.activations.tanh)(batch2_patch_noise)
        max_pool2_patch_noise = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation2_patch_noise)
                        
        con2d_3_patch_noise = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(max_pool2_patch_noise)
        batch3_patch_noise = BatchNormalization()(con2d_3_patch_noise)
        activation3_patch_noise = Activation(tf.keras.activations.tanh)(batch3_patch_noise)
        max_pool3_patch_noise = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation3_patch_noise)


        con2d_4_patch_noise = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(max_pool3_patch_noise)
        batch4_patch_noise = BatchNormalization()(con2d_4_patch_noise)
        activation4_patch_noise = Activation(tf.keras.activations.tanh)(batch4_patch_noise)
        max_pool4_patch_noise = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation4_patch_noise)

        flatten_patch_noise = Flatten()(max_pool4_patch_noise)
        dense_patch_noise = Dense(100, activation=tf.keras.activations.tanh)(flatten_patch_noise)


        conv2d_patches1 = Conv2D(filters=96, kernel_size=(5, 5), strides=(2, 2), padding="same")(cons_layer_patches)
            
        batch1_patches = BatchNormalization()(conv2d_patches1)
        activation1_patches = Activation(tf.keras.activations.tanh)(batch1_patches)
        max_pool1_patches = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation1_patches)

        con2d_2_patches = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(max_pool1_patches)
        batch2_patches = BatchNormalization()(con2d_2_patches)
        activation2_patches = Activation(tf.keras.activations.tanh)(batch2_patches)
        max_pool2_patches = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation2_patches)

        con2d_3_patches = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(max_pool2_patches)
        batch3_patches = BatchNormalization()(con2d_3_patches)
        activation3_patches = Activation(tf.keras.activations.tanh)(batch3_patches)
        max_pool3_patches = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation3_patches)

        con2d_4_patches = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(max_pool3_patches)
        batch4_patches = BatchNormalization()(con2d_4_patches)
        activation4_patches = Activation(tf.keras.activations.tanh)(batch4_patches)
        max_pool4_patches = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(activation4_patches)


        flatten_patches = Flatten()(max_pool4_patches)
        dense_patches = Dense(100, activation=tf.keras.activations.tanh)(flatten_patches)

        merge_layer = concatenate([dense_patch_noise, dense_patches], axis=1)

      
        for i in range(2):
            if i == 0:
                output_layer_patches_temp = Dense(100, activation=tf.keras.activations.tanh)(merge_layer)
            else: 
                output_layer_patches_temp = Dense(100, activation=tf.keras.activations.tanh)(output_layer_patches_temp)
                   
        final_patches_output_layer = output_layer_patches_temp    

        dense_merged = Dense(6, activation=tf.keras.activations.softmax)(final_patches_output_layer)
        model = Model(inputs=(input_layer_patch_noise, input_layer_patches), outputs=dense_merged)

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=['accuracy'])

        if model_name is None:
            model_name = self.__generate_model_name(2, 100, height, width)

        self.model_name = model_name
        self.model = model
        
        return model
   
    def compile(self):
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95, decay=0.0005),
                      metrics=["acc"])


    def get_tensorboard_path(self):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save TensorBoard log-files.")

        # Create directory if not exists
        path = os.path.join(self.GLOBAL_TENSORBOARD_DIR, self.sector, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def get_save_model_path(self, file_name):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save checkpoints.")

        # Create directory if not exists
        path = os.path.join(self.GLOBAL_SAVE_MODEL_DIR, self.sector, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Append file name and return
        return os.path.join(path, file_name)

    def __get_initial_epoch(self):
        # Means we train from scratch
        if self.model_path is None:
            return 0

        path = self.model_path
        file_name = path.split(os.sep)[-1]

        if file_name is None:
            return 0

        file_name = file_name.split(".")[0]
        splits = file_name.split("-")
        for split in splits:
            if split.startswith("e"):
                epoch = split.strip("e")
                return int(epoch)

        return 0
       
    def train(self, train_ds, val_ds_test):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()

        return self.model.fit(DataGeneratorNoisePatch(train_ds),
                            epochs=70,
                            verbose=1,
                            initial_epoch=0,
                            validation_data=DataGeneratorNoisePatch(val_ds_test, shuffle=False),
                            callbacks=callbacks)

    def get_callbacks(self):
        default_file_name = "fm-e{epoch:05d}.h5"
        save_model_path = self.get_save_model_path(default_file_name)

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                         verbose=1,
                                                         save_weights_only=False,
                                                         period=1)

        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                       patience=5, min_lr=0.001)

        tensorboard_cb = TensorBoard(log_dir=self.get_tensorboard_path())
        print_lr_cb = PrintLearningRate()

        return [save_model_cb, print_lr_cb, tensorboard_cb]

    def evaluate(self, test_ds, model_path=None):
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        elif self.model is None:
            raise ValueError("No model available")

        test_loss, test_acc = self.model.evaluate(test_ds)

        return (test_acc, test_loss)

    def predict(self, dataset, load_model=None):
        if load_model is not None:
            self.model = tf.keras.models.load_model(load_model)
        elif self.model is None:
            raise ValueError("No model available")

        test_ds = dataset.get_test_data()
        predicted_labels = self.model.predict_class(test_ds)
        true_labels = dataset.get_labels(test_ds)

        return (true_labels, predicted_labels)

    def print_model_summary(self):
        if self.model is None:
            print("Can't print model summary, self.model is None!")
        else:
            print(f"\nSummary of model:\n{self.model.summary()}")


class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f"Learning rate on_epoch_end epoch {epoch}: {K.eval(lr_with_decay)}")