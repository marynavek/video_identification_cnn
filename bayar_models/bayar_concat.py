from tensorflow.keras import layers
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K
from data_generator_ensemble import DataGeneratorEnsemble
from sklearn.ensemble import VotingClassifier
from bayar_singular_net import ConstrainedBayar

class WeightedSum(layers.Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.a = self.add_weight(name='alpha',
                                 shape=(1),
                                 initializer=tf.keras.initializers.Constant(0.5),
                                 dtype='float32',
                                 trainable=True,
                                 constraint=tf.keras.constraints.min_max_norm(
                                     max_value=1, min_value=0))
        super(WeightedSum, self).build(input_shape)

    def call(self, model_outputs):
        return self.a * model_outputs[0] + (1 - self.a) * model_outputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class EnsembleWeigthedModelNet:
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
        path_base = "/Users/marynavek/Projects/video_identification_cnn/bayar_ensemble_model"
        new_path = os.path.join(path_base,  "ensemble")
        return new_path

    def __generate_tensor_path(self):
        path_base = "/Users/marynavek/Projects/video_identification_cnn/bayar_ensemble_keras"
        new_path = os.path.join(path_base,  "ensemble")
        return new_path

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path, custom_objects={
                                                     'ConstrainedBayar': ConstrainedBayar})
        return model

    def create_branch(self, input_layer_sector, num_classes, sector):
        name = 'Constrained_layer' + str(sector)
        constrained_conv_layer = Conv2D(filters=3,
                                kernel_size=(5,5),
                                strides=(1, 1),
                                padding="valid", # Intentionally
                                kernel_constraint=ConstrainedBayar(), 
                                name=name)(input_layer_sector)
        name = 'conv2d1_' + str(sector)
        conv2d_1 = Conv2D(96, (5,5), strides=(2,2),padding='same', name=name)(constrained_conv_layer)
        name = 'batch1_' + str(sector)
        batch_norm1 = BatchNormalization(name=name)(conv2d_1)
        name = 'activation1_' + str(sector)
        activation1 = Activation(tf.keras.activations.tanh, name=name)(batch_norm1)
        name = 'max_pool1_' + str(sector)
        max_pool1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name=name)(activation1)

        name = 'conv2d2_' + str(sector)
        conv2d_2 = Conv2D(64, (3,3), strides=(1,1),padding='same', name=name)(max_pool1)
        name = 'batch2_' + str(sector)
        batch_norm2 = BatchNormalization(name=name)(conv2d_2)
        name = 'activation2_' + str(sector)
        activation2 = Activation(tf.keras.activations.tanh, name=name)(batch_norm2)
        name = 'max_pool2_' + str(sector)
        max_pool2 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name=name)(activation2)  

        name = 'conv2d3_' + str(sector)
        conv2d_4 = Conv2D(64, (3,3), strides=(1,1),padding='same', name=name)(max_pool2)
        name = 'batch3_' + str(sector)
        batch_norm4 = BatchNormalization(name=name)(conv2d_4)
        name = 'activation3_' + str(sector)
        activation4 = Activation(tf.keras.activations.tanh, name=name)(batch_norm4)
        name = 'max_pool3_' + str(sector)
        max_pool4 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name=name)(activation4) 

        name = 'conv2d4_' + str(sector)
        conv2d_3 = Conv2D(128, (1,1), strides=1,padding='same', name=name)(max_pool4)
        name = 'batch4_' + str(sector)
        batch_norm3 = BatchNormalization(name=name)(conv2d_3)
        name = 'activation4_' + str(sector)
        activation3 = Activation(tf.keras.activations.tanh, name=name)(batch_norm3)
        name = 'max_pool4_' + str(sector)
        max_pool3 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name=name)(activation3) 

        name = 'flatten_' + str(sector)
        flatten = Flatten(name=name)(max_pool3)   

        name = 'dense1_' + str(sector)
        dense1 = Dense(200, name=name)(flatten)
        name = 'activation5_' + str(sector)
        activation4 = Activation(tf.keras.activations.tanh, name=name)(dense1)  

        name = 'dense2_' + str(sector)
        dense2 = Dense(200, name=name)(activation4)
        name = 'cactivation6_' + str(sector)
        activation4 = Activation(tf.keras.activations.tanh, name=name)(dense2)    

        name = 'dense3_' + str(sector)
        final_dense = Dense(num_classes, activation=tf.keras.activations.softmax, name=name)(activation4)   

        return final_dense


    def create_model(self, num_classes):
        input_shape = (128, 128, 3)
        # sector_1_model = self.load_model(self.sector_1_model_path)
        # sector_2_model = self.load_model(self.sector_2_model_path)
        # sector_3_model = self.load_model(self.sector_3_model_path)
        # sector_4_model = self.load_model(self.sector_4_model_path)
        input_layer_sector_1 = Input(shape=input_shape)
        input_layer_sector_2 = Input(shape=input_shape)
        input_layer_sector_3 = Input(shape=input_shape)
        input_layer_sector_4 = Input(shape=input_shape)
        sector_1_dense = self.create_branch(input_layer_sector_1, num_classes, sector='sector_1')
        sector_2_dense = self.create_branch(input_layer_sector_2, num_classes, sector='sector_2')
        sector_3_dense = self.create_branch(input_layer_sector_3, num_classes, sector='sector_3')
        sector_4_dense = self.create_branch(input_layer_sector_4, num_classes, sector='sector_4')
        
        # for layer in sector_1_dense.layers:
        #     # layer.trainable = False
        #     layer._name = layer.name + str("_sector1")

        # for layer in sector_2_dense.layers:
        #     # layer.trainable = False
        #     layer._name = layer.name + str("_sector2")

        # for layer in sector_3_dense.layers:
        #     # layer.trainable = False
        #     layer._name = layer.name + str("_sector3")

        # for layer in sector_4_dense.layers:
        #     # layer.trainable = False
        #     layer._name = layer.name + str("_sector4")
        
        weigth_sum = WeightedSum()([sector_1_dense, sector_2_dense, 
                    sector_3_dense, sector_4_dense])

        # weigth_sum = WeightedSum()([sector_1_model.layers[-1].output, sector_2_model.layers[-1].output, 
        #             sector_3_model.layers[-1].output, sector_4_model.layers[-1].output])


        dense_layer = Dense(num_classes, activation=tf.keras.activations.tanh)(weigth_sum)

        final_model = Model(inputs=(input_layer_sector_1, input_layer_sector_2, input_layer_sector_3, input_layer_sector_4), outputs=dense_layer)
        # final_model = Model(inputs=(sector_1_dense.inputs, sector_2_model.inputs, 
        #                 sector_3_model.inputs, sector_4_model.inputs), outputs=dense_layer)
        print([(l, l.name) for l in final_model.layers])
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        final_model.compile(loss="categorical_crossentropy",
                            optimizer=opt,
                            metrics=['accuracy'])

        model_name = self.__generate_model_name()

        self.model_name = model_name
        self.model = final_model
        final_model.summary()

        return final_model

    def train(self, train_ds, val_ds_test, num_classes):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        callbacks = self.get_callbacks()
        # self.model.fit_transform(DataGeneratorEnsemble(train_ds))

        return self.model.fit(DataGeneratorEnsemble(train_ds, sector="1", num_classes=num_classes),
                            epochs=70,
                            verbose=1,
                            initial_epoch=0,
                            validation_data=DataGeneratorEnsemble(val_ds_test, sector="1", num_classes=num_classes, shuffle=False),
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
        # save_model_cb = self.model.save(save_model_path)
        # self.model.save(save_model_path)
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