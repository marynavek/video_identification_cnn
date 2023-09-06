import os
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from data_generator_bayar import DataGeneratorBayar
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K

class PRNUModelNet:

    def __init__(self, sector, patch_type):
        self.sector = sector
        self.patch_type = patch_type
        self.model = None
        self.model_name = None
        self.global_save_model_dir = self.__generate_model_path()
        self.global_tensorboard_dir = self.__generate_tensor_path()

    def __generate_model_name(self):
        model_name = f"BayayrModel-{self.sector}"

        return model_name

    def __generate_model_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/PRNU_model_exp"
        new_path = os.path.join(path_base, self.patch_type, self.sector)
        return new_path

    def __generate_tensor_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/PRNU_model_exp"
        new_path = os.path.join(path_base, self.patch_type, self.sector)
        return new_path

    def create_model(self, num_classes, model_name=None):
        input_shape = (128, 128, 1)

        input_layer = Input(shape=input_shape)

        conv2d_1 = Conv2D(64, (3,3), strides=(1,1),padding='same')(input_layer)
        batch_norm1 = BatchNormalization()(conv2d_1)
        activation1 = Activation(tf.keras.activations.relu)(batch_norm1)

        conv2d_2 = Conv2D(64, (3,3), strides=(1,1),padding='same')(activation1)
        batch_norm2 = BatchNormalization()(conv2d_2)
        activation2 = Activation(tf.keras.activations.relu)(batch_norm2)
        max_pool1 = MaxPooling2D(pool_size=(2,2), strides=1, padding='same')(activation2)
        dropout1 = Dropout(0.5)(max_pool1)

        conv2d_3 = Conv2D(96, (3,3), strides=(1,1),padding='same')(dropout1)
        batch_norm3 = BatchNormalization()(conv2d_3)
        activation3 = Activation(tf.keras.activations.relu)(batch_norm3)

        conv2d_4 = Conv2D(96, (3,3), strides=(1,1),padding='same')(activation3)
        batch_norm4 = BatchNormalization()(conv2d_4)
        activation4 = Activation(tf.keras.activations.relu)(batch_norm4)
        max_pool2 = MaxPooling2D(pool_size=(2,2), strides=1, padding='same')(activation4)
        # dropout2 = Dropout(0.5)(max_pool2)

        flatten = Flatten()(max_pool2)

        dense1 = Dense(256)(flatten)
        activation5 = Activation(tf.keras.activations.relu)(dense1)
        # dropout3 = Dropout(0.5)(activation5)

        dense2 = Dense(256)(activation5)

        final_dense = Dense(num_classes, activation=tf.keras.activations.softmax)(dense2)

        model = Model(input_layer, final_dense)

        # opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95, decay=0.0005)

        opt = tf.keras.optimizers.Adadelta()
        # opt = tf.keras.optimizers.Adam()
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=['accuracy'])

        model_name = self.__generate_model_name()

        self.model_name = model_name
        self.model = model
        
        return model

    def train(self, train_ds, val_ds_test, num_classes):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()
        
        self.model.fit(DataGeneratorBayar(train_ds, num_classes=num_classes, batch_size=32),
                       epochs=50,
                       initial_epoch=0,
                       validation_data=DataGeneratorBayar(val_ds_test, num_classes=num_classes, batch_size=32, shuffle=False),
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
                                                            monitor='val_accuracy',
                                                            save_best_only=True,
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