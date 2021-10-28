import io
import json
import logging
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from tools.helper import get_dir

SCRAPER_DIR = get_dir('assets\pics')
TRAIN_FOLDER = get_dir('assets\pics', "training_cards")
VALIDATE_FOLDER = get_dir('assets\pics', "validate_cards")
TEST_FOLDER = get_dir('assets\tests', "test_cards")

log = logging.getLogger(__name__)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)


def binary_pil_to_cv2(img):
    return cv2.cvtColor(np.array(Image.open(io.BytesIO(img))), cv2.COLOR_BGR2RGB)



img_height = 45#45
img_width = 35#32#


class CardNeuralNetwork():

    @staticmethod
    def create_augmented_images():
        shutil.rmtree(TRAIN_FOLDER, ignore_errors=True)
        shutil.rmtree(VALIDATE_FOLDER, ignore_errors=True)

        log.info("Augmenting data with random pictures based on templates")

        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen_normal = ImageDataGenerator(
            brightness_range=[0.1,1],
            rotation_range=1,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest')
        datagen_fantasy = ImageDataGenerator(
            brightness_range=[0.1,1],
            rotation_range=30,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0,
            zoom_range=0.25,
            horizontal_flip=False,
            fill_mode='nearest')
        datagen_fantasy_2 = ImageDataGenerator(
            brightness_range=[0.1,1],
            rotation_range=30,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest')

        cardImage = cv2.imread("assets\cards.jpg")
        for folder in [TRAIN_FOLDER, VALIDATE_FOLDER]:
            card_ranks_original = '23456789TJQKAN'
            original_suits = 'SHDC'
            namelist = []
            for c in card_ranks_original:
                for s in original_suits:
                    namelist.append(c + s)
            x_i = 0
            y_j = 0
            back = np.zeros((img_height,img_width,3),np.uint8)
            back[:,:] = (103,148,55)
            for name in tqdm(namelist):
                img1 = cardImage[int(y_j*352/4.0):int(y_j*352/4.0+45),int(x_i*845/13.0):int(x_i*845/13.0+35)]
                img2 = cardImage[int(y_j*352/4.0+1):int(y_j*352/4.0+33),int(x_i*845/13.0+1):int(x_i*845/13.0+33)]
                img3 = cardImage[int(y_j*352/4.0+1):int(y_j*352/4.0+33),int(x_i*845/13.0+1):int(x_i*845/13.0+23)]
                if name[0] == "K" or name[0] == "T" or name[0] == "Q":
                    img3 = np.copy(img2) 
                y_j+=1
                if y_j==4:
                    x_i += 1
                    y_j = 0
               
                x = np.copy(img1)
                x = x.reshape((1,) + x.shape)   
                directory = os.path.join(SCRAPER_DIR, folder, name)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                i = 0
                cnt = 15000
                for _ in datagen_normal.flow(x, save_to_dir=directory,
                                      save_prefix=name,
                                      save_format='bmp',
                                      ):
                    i += 1
                    if i > cnt:
                        break  # otherwise the generator would loop indefinitely

                back[1:,1:] = img1[1:,1:]
                x = np.copy(back)
                x = x.reshape((1,) + x.shape)   
                i = 0
                for _ in datagen_normal.flow(x, save_to_dir=directory,
                                      save_prefix=name,
                                      save_format='bmp',
                                      ):
                    i += 1
                    if i > cnt:
                        break  # otherwise the generator would loop indefinitely


                img2 = cv2.resize(img2,(35,45))
                x = img2
                x = x.reshape((1,) + x.shape)
                i = 0
                for _ in datagen_fantasy.flow(x, save_to_dir=directory,
                                      save_prefix=name,
                                      save_format='bmp',
                                      ):
                    i += 1
                    if i > cnt:
                        break  # otherwise the generator would loop indefinitely
                img3 = cv2.resize(img3,(35,45))
                x = img3
                x = x.reshape((1,) + x.shape)
                i = 0
                for _ in datagen_fantasy_2.flow(x, save_to_dir=directory,
                                      save_prefix=name,
                                      save_format='bmp',
                                      ):
                    i += 1
                    if i > cnt:
                        break  # otherwise the generator would loop indefinitely

    def train_neural_network(self):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        self.train_generator = ImageDataGenerator(
            rescale=0.05,
            shear_range=0.01,
            zoom_range=0.05,
            horizontal_flip=False).flow_from_directory(
            directory=os.path.join(SCRAPER_DIR, TRAIN_FOLDER),
            target_size=(img_height, img_width),
            batch_size=512,
            class_mode='binary',
            color_mode='rgb')

        self.validation_generator = ImageDataGenerator(
            rescale=0.06,
            shear_range=0.01,
            zoom_range=0.05,
            horizontal_flip=False).flow_from_directory(
            directory=os.path.join(SCRAPER_DIR, VALIDATE_FOLDER),
            target_size=(img_height, img_width),
            batch_size=512,
            class_mode='binary',
            color_mode='rgb')

        num_classes = 56
        input_shape = (img_height, img_width, 3)
        epochs = 100
        from tensorflow.keras.callbacks import TensorBoard
        from tensorflow.keras.constraints import MaxNorm
        from tensorflow.keras.layers import Conv2D, MaxPooling2D
        from tensorflow.keras.layers import Dropout, Flatten, Dense
        from tensorflow.keras.models import Sequential
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(2048, activation='relu', kernel_constraint=MaxNorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu', kernel_constraint=MaxNorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        from tensorflow.keras.losses import sparse_categorical_crossentropy
        from tensorflow.keras import optimizers
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])

        log.info(model.summary())

        print("model compile")

        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=1,
                                   verbose=1, mode='auto')
        tb = TensorBoard(log_dir='c:/tensorboard/pb',
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         embeddings_freq=1,
                         embeddings_layer_names=False,
                         embeddings_metadata=False)

        print("tensorboard")


        model.fit(self.train_generator,
                  epochs=epochs,
                  verbose=1,
                  validation_data=self.validation_generator,
                  callbacks=[early_stop])
        print("model fit")
        self.model = model
        score = model.evaluate(self.validation_generator, steps=56)
        print('Validation loss:', score[0])
        print('Validation accuracy:', score[1])

    def save_model_to_disk(self):
        # serialize model to JSON
        log.info("Save model to disk")
        class_mapping = self.train_generator.class_indices
        class_mapping = dict((v, k) for k, v in class_mapping.items())
        with open(SCRAPER_DIR + "/model_classes.json", "w") as json_file:
            json.dump(class_mapping, json_file)
        model_json = self.model.to_json()
        with open(SCRAPER_DIR + "/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(SCRAPER_DIR + "/model.h5")
        log.info("Done.")

    def load_model(self):
        log.info("Loading model from disk")
        # load json and create model
        with open(SCRAPER_DIR + '/model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        from keras.models import model_from_json
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(SCRAPER_DIR + "/model.h5")
        with open(SCRAPER_DIR + "/model_classes.json") as json_file:
            self.class_mapping = json.load(json_file)

    def predict(self, file):
        print(file)
        img = cv2.imread(file)
        prediction = predict(img, self.loaded_model, self.class_mapping)
        return prediction


import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
# actual predict function
def predict(pil_image, nn_model, mapping):
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    img = cv2.resize(pil_image, (img_width, img_height))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x = x * 0.02
    prediction = np.argmax(nn_model.predict(x))
    card = mapping[str(prediction)]
   
    if card == "NH" or card == "ND" or card == "NS":
        card = "NC"
    return card
