#----------------------------import required modules--------------------------#

#import sequential for constructing the model
from keras.models import Sequential

#import load_model to save and load the model later
from keras.models import load_model

#import Conv2D to construct the convolutional layer.
from keras.layers import Conv2D

#import MaxPooling2D to construct the max pooling layer
from keras.layers import MaxPooling2D

#import Flatten to flatten the input before sending it to fully connected layer
from keras.layers import Flatten

#import Dense to construct the fully connected layer
from keras.layers import Dense

#import Dropout as a regularization technique to prevent overfitting
from keras.layers import Dropout

#import ImageDataGenerator to initialize the data generator
from keras.preprocessing.image import ImageDataGenerator

#import image to read the input images
from keras.preprocessing import image

#---------------------------Construct the Model-------------------------------#

classifier = Sequential()

#construct the network with 3 convolutional layers, each followed by a Max
#Pooling layer, and these followed by two fully connected layers.

classifier.add(Conv2D(32, (3, 3), 
                     input_shape = (256, 256, 3), 
                     activation = 'relu',))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), 
                    activation = 'relu',))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3),
                     activation = 'relu',))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(64, 
                    activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, 
                    activation = 'sigmoid', ))

classifier.compile(optimizer = 'rmsprop', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])


#---------------------------Read the input------------------------------------#

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_set = train_datagen.flow_from_directory('data/train',
                                               target_size=(256, 256),
                                               batch_size=10,
                                               class_mode='binary')

validation_set = validation_datagen.flow_from_directory('data/validation',
                                                        target_size=(256, 256),
                                                        batch_size=10,
                                                        class_mode='binary')

#---------------------------Train the model-----------------------------------#

classifier.fit_generator(train_set,
                         steps_per_epoch = 2500, 
                         epochs = 25, 
                         validation_data = validation_set, 
                         validation_steps = 210)

#---------------------------Save the model and weights------------------------#

model.save('Classifier.h5')
model.save_weights('Classifier_weights.h5')