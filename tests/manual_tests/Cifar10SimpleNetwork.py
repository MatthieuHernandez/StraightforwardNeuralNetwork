# https://www.tensorflow.org/tutorials/images/cnn
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
# Encode labels
one_hot_train_labels = tf.one_hot(train_labels, 10)
one_hot_test_labels = tf.one_hot(test_labels, 10)

model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(16, (3, 3), padding="valid", activation='gelu', use_bias=False))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding="valid", activation='gelu', use_bias=False))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(150, activation='gelu'))
model.add(layers.Dense(10, activation='sigmoid'))

model.summary()

model.compile(optimizer=optimizers.SGD(learning_rate=0.002, momentum=0.8),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

history = model.fit(train_images, one_hot_train_labels, epochs=50, batch_size=1,
                    validation_data=(test_images, one_hot_test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)