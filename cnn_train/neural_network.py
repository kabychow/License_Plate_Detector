from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(20, 20), color_mode='grayscale', batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('dataset/test', target_size=(20, 20), color_mode='grayscale', batch_size=32, class_mode='categorical')

# model = load_model('model.h5')

model = Sequential()
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 1)))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(34, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=453,
        epochs=10,
        validation_data=test_generator,
        validation_steps=134)

model.save('model.h5')
