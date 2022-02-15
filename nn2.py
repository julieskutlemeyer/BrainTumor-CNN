import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#import the directory where the dataset are
datadir = "./Brain Tumor Data Set"
data_dir = pathlib.Path(datadir)

image_count = len(list(data_dir.glob('*/*')))
healthy = len(list(data_dir.glob('healthy/*')))
brain_tumor = len(list(data_dir.glob('brain tumor/*')))

print(image_count, healthy, brain_tumor)

# image = PIL.Image.open(str(brain_tumor[0]))
# image.show()

#picture-frame
batch_size = 32
img_height = 180
img_width = 180

#lager en tf.data.Dataset objekt som inneholder datasettet. enklere å jobbe med tensorflow når det er på det formatet
#returnerer to "lister"/tensors, en med (batch, høyde, bredde, kanaler) , og en med (batch,)
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, #80%training, 20% testing
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  #har to enheter med disse kanalene:
  # (32, 180, 180, 3)
  # (32,)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names #["Brain Tumor, Healthy"]


# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

AUTOTUNE = tf.data.AUTOTUNE #når input pipelinen kjører, tracker tf.data hvor mye tid hver iterasjon tar
# denne tiden blir matet inn i optimaliseringsalgoritmen. 
#optimaliseringsalgoritmen optimaliserer allkokasjonen for cpu-en
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #.cache() henter datasettet fra disk og til minnet. .shuffle(X) lager en buffer med X samples, og shuffler de random. .prefetch() prefetcher fra train.ds tallet som er inni .prefetch
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#lage selve modellen
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'), #et lag. 16 filtre, 3x3 kernel, relu activation
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) #training + validation accuracy

model.summary() #se alle lagene av nettverket.

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#data augmentation:
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

######################---TESTING ON PICS---################################################################

# #PIC1
brainT1_path = "./BrainT1.jpg"
img1 = tf.keras.utils.load_img(
    brainT1_path, target_size=(180, 180)
 )
img1_array = tf.keras.utils.img_to_array(img1)
img1_array = tf.expand_dims(img1_array, 0) # Create a batch
predictions1 = model.predict(img1_array)
score1 = tf.nn.softmax(predictions1[0])

print(
    "This image1 most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score1)], 100 * np.max(score1))
)

 #PIC2
brainNT1_path = "./BrainNT1.jpg"
img2 = tf.keras.utils.load_img(
     brainNT1_path, target_size=(180, 180)
 )
# img2.show()
img2_array = tf.keras.utils.img_to_array(img2)
img2_array = tf.expand_dims(img2_array, 0) # Create a batch
predictions2 = model.predict(img2_array)
score2 = tf.nn.softmax(predictions2[0])

print(
    "This image2 most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score2)], 100 * np.max(score2))
)

#PIC3
brainT2_path = "./BrainT2.jpg"
img3 = tf.keras.utils.load_img(
    brainT2_path, target_size=(180, 180)
 )
# img3.show()
img3_array = tf.keras.utils.img_to_array(img3)
img3_array = tf.expand_dims(img3_array, 0) # Create a batch
predictions3 = model.predict(img3_array)
score3 = tf.nn.softmax(predictions3[0])

print(
    "This image2 most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score3)], 100 * np.max(score3))
)



