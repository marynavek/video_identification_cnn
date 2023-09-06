import os
import tensorflow as tf
from tensorflow.keras.layers import concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K
from data_generator_ensemble import DataGeneratorEnsemble
from sklearn.ensemble import VotingClassifier
from bayar_singular_net import ConstrainedBayar


class EnsembleModelNet:
    def __init__(self, sector_1_model_path, sector_2_model_path, sector_3_model_path, sector_4_model_path):
        self.sector_1_model_path = sector_1_model_path
        self.sector_2_model_path = sector_2_model_path
        self.sector_3_model_path = sector_3_model_path
        self.sector_4_model_path = sector_4_model_path
        self.model_path = None
        self.model=None
        self.global_save_model_dir = self.__generate_model_path()
        self.global_tensorboard_dir = self.__generate_tensor_path()

    def __generate_model_name(self):
        model_name = "FC_Bayayr_model_Ensemble"

        return model_name

    def __generate_model_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/ensemble_model"
        new_path = os.path.join(path_base,  "ensemble")
        return new_path

    def __generate_tensor_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/ensemble_keras"
        new_path = os.path.join(path_base,  "ensemble")
        return new_path

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
        return model

    def create_model(self):
        sector_1_model = self.load_model(self.sector_1_model_path)
        sector_2_model = self.load_model(self.sector_2_model_path)
        sector_3_model = self.load_model(self.sector_3_model_path)
        sector_4_model = self.load_model(self.sector_4_model_path)
        
        for layer in sector_1_model.layers:
            layer.trainable = False
            layer._name = layer.name + str("_sector1")

        for layer in sector_2_model.layers:
            layer.trainable = False
            layer._name = layer.name + str("_sector2")

        for layer in sector_3_model.layers:
            layer.trainable = False
            layer._name = layer.name + str("_sector3")

        for layer in sector_4_model.layers:
            layer.trainable = False
            layer._name = layer.name + str("_sector4")
        
        # merge_layer = concatenate([sector_1_model.layers[-1].output, sector_2_model.layers[-1].output, 
        #             sector_3_model.layers[-1].output, sector_4_model.layers[-1].output])


        # dense_layer = Dense(6, activation=tf.keras.activations.tanh)(merge_layer)

        final_model = VotingClassifier(
             estimators=[('sect_1', sector_1_model),
                         ('sect_2', sector_2_model),
                         ('sect_3', sector_3_model),
                         ('sect_4',sector_3_model)], 
             voting='soft',
             flatten_transform=True)


        # final_model = Model(inputs=(sector_1_model.inputs, sector_2_model.inputs, 
        #                 sector_3_model.inputs, sector_4_model.inputs), outputs=dense_layer)

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        # final_model.compile(loss="categorical_crossentropy",
        #                     optimizer=opt,
        #                     metrics=['accuracy'])

        # final_model.summary()
        model_name = self.__generate_model_name()

        self.model_name = model_name
        self.model = final_model
        # self.print_model_summary()
        # tf.keras.utils.plot_model(final_model, "multi_input_and_output_model.png", show_shapes=True)

        return final_model

    def train(self, train_ds, val_ds_test):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()
        self.model.fit_transform(DataGeneratorEnsemble(train_ds))

        # return self.model.fit(DataGeneratorEnsemble(train_ds),
        #                     epochs=20,
        #                     verbose=1,
        #                     initial_epoch=0,
        #                     validation_data=DataGeneratorEnsemble(val_ds_test, shuffle=False),
        #                     callbacks=callbacks)
        
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