from tensorflow.keras.applications import EfficientNetB0
import os
import tensorflow as tf
from tensorflow.keras.layers import RandomRotation, Dense, Input, RandomTranslation, RandomFlip, RandomContrast, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential
from data_generator_bayar import DataGeneratorBayar
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K
from keras.constraints import Constraint

import numpy as np 


class EfficientNet:

    def __init__(self, sector="1", patch_type="cortiva"):
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
        path_base = "/Users/marynavek/Projects/video_identification_cnn/experiments_mov_models"
        new_path = os.path.join(path_base, self.patch_type, self.sector)
        return new_path

    def __generate_tensor_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/experiments_mov_tensor"
        new_path = os.path.join(path_base, self.patch_type, self.sector)
        return new_path

    def create_model(self, num_classes, model_name=None):
        
        img_augmentation = Sequential(
            [
                RandomRotation(factor=0.15),
                RandomTranslation(height_factor=0.1, width_factor=0.1),
                RandomFlip(),
                RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )

        inputs = Input(shape=(224, 224, 3))
        # x = img_augmentation(inputs)
        model = EfficientNetB0(include_top=False,input_tensor=inputs, weights='imagenet')
        model.trainable = False

        x = GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = Dropout(top_dropout_rate, name="top_dropout")(x)
        
        outputs = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs, outputs, name="EfficientNet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model_name = self.__generate_model_name()
        # model.summary()
        self.model_name = model_name
        self.model = model
        return model


    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        self.model = model
        model_name = self.__generate_model_name()
        # model.summary()
        self.model_name = model_name

    def train(self, train_ds, val_ds_test, num_classes):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()
        
        self.model.fit(DataGeneratorBayar(train_ds, num_classes=num_classes, batch_size=32),
                       epochs=100,
                       initial_epoch=13,
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