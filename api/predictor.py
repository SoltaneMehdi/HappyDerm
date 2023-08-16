from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from .Mobilenet_classifier import classify


def predict(image):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array.reshape((1,) + img_array.shape)
    generator = datagen.flow(img_array)
    prediction = classify(generator)
    return prediction
