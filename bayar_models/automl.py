import numpy as np
import tensorflow as tf
# from tensorflow.keras.datasets import mnist

import autokeras as ak
from data_generator_bayar import DataGeneratorBayar

from prepate_csv_dataset_for_bayar import DataSetGeneratorBayar

dataset_path ="/Users/marynavek/Projects/files/ten_homo_patches_1st_sector"

data_factory = DataSetGeneratorBayar(input_dir_patchs=dataset_path)

num_classes = len(data_factory.get_class_names())

train_ds = data_factory.create_train_dataset()
val_ds_test = data_factory.create_validation_dataset()

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Only search ResNet architectures.
    # Normalize the dataset.
    normalize=False,
    # Do not do data augmentation.
    augment=False,
)(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=5
)

# print(DataGeneratorBayar(train_ds,  num_classes=num_classes, batch_size=32).shape)  # (60000, 28, 28, 1)
# x, y = DataGeneratorBayar(train_ds,  num_classes=num_classes, batch_size=32)
# print(x.shape)
# clf = ak.ImageClassifier(overwrite=True, max_trials=5)
def callable_iterator(generator):
  for img_batch, targets_batch in generator:
    # print("hello")
    # if img_batch.shape[0] == expected_batch_size:
      yield img_batch, targets_batch
train_dataset = tf.data.Dataset.from_generator(
    lambda: callable_iterator(DataGeneratorBayar(train_ds,  num_classes=num_classes, batch_size=32)),
    output_types=(tf.float32, tf.float32), 
    output_shapes=(tf.TensorShape([None, 128, 128, 3]),
                   tf.TensorShape([None, num_classes])))
val_dataset = tf.data.Dataset.from_generator(
    lambda: callable_iterator(DataGeneratorBayar(val_ds_test,  num_classes=num_classes, batch_size=32)),
    output_types=(tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([None, 128, 128, 3]),
                   tf.TensorShape([None, num_classes])))

# Autokeras data_utils gets confused by the generator.
# Just let it know that the data is indeed batched.
ak.utils.data_utils.batched = lambda _: True

clf.fit(train_dataset, 
    validation_data = val_dataset,
    epochs=10)

predicted_y = clf.predict(val_dataset)
print(predicted_y)

clf.summary()
