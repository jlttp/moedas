# USO
# python train.py --dataset dataset --model moedas.model --labelbin lb.pickle

# Codigo adaptado do tutorial: https://www.dobitaobyte.com.br/como-criar-um-dataset-para-deep-learning/


import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from modelo.minivggnet import miniVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path do dataset (diretorio contendo imagens para treinar a rede)")
ap.add_argument("-m", "--model", required=True,
    help="path para salvar o modelo treinado")
ap.add_argument("-l", "--labelbin", required=True,
    help="path para labels")
ap.add_argument("-p", "--plot", type=str, default="results.png",
    help="path do grafico accuracy/loss")
args = vars(ap.parse_args())

EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

data = []
labels = []

print("[INFO] carregando imagens...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv.imread(imagePath)
    image = cv.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
 
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] matriz dos dados: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

print("[INFO] compilando modelo...")
model = miniVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("[INFO] treinando rede...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

print("[INFO] salvando modelo...")
model.save(args["model"])

print("[INFO] salvando labels...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])