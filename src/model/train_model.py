from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input, AveragePooling2D, Activation,Conv2D, MaxPooling2D, BatchNormalization,Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
# the cifar-10 dataset has 32x32 images from 10 differente classes with 6000 images per class, giving us 50k for training and 10k for validation
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# normalizing the data
x_train = x_train / 255
x_test = x_test / 255
# since we have no ordinary relationship between the classes we need to encode some of the data so we don't get unexpected results
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape =  (32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# we use softmax because we are working with 10 different classes
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# since we needed more images because the previous models were overfitting I'm using keras to flip and shift the image a little bit
# so we can have more data for the training
ImgDataGen = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.2, vertical_flip = True, rotation_range = 15)
it_train = ImgDataGen.flow(x_train, y_train_cat)
steps = int(x_train.shape[0] / 64)

#gives us a summary of the model layers, trainable and non-trainable parameters as well as the total parameters
model.summary()
history = model.fit(it_train, epochs = 55, steps_per_epoch = steps, validation_data = (x_test, y_test_cat))
# prints the final acc of the model
evaluation = model.evaluate(x_test, y_test_cat)
print('Final accuracy: {}'.format(evaluation[1]))

model_name = "cifar_model_" + epochs + "epoch.h5"
model.save(model_name)
