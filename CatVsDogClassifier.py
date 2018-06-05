from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3),  input_shape = (256, 256, 3),  activation = 'relu',))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu',))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu',))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(1, activation = 'sigmoid', ))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
train_gen = ImageDataGenerator()
test_gen = ImageDataGenerator()
validation_gen = ImageDataGenerator()


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_set = train_datagen.flow_from_directory(
    'data/train',
    target_size=(256, 256),
    batch_size=10,
    class_mode='binary')

validation_set = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(256, 256),
    batch_size=10,
    class_mode='binary')


classifier.fit_generator(train_set, steps_per_epoch = 2500, epochs = 25, validation_data = validation_set, validation_steps = 210)


