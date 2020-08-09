# USO
# python classify.py --model moedas.model --labelbin lb.pickle --image ./exemplos/img1.jpg

# Codigo adaptado do tutorial: https://www.dobitaobyte.com.br/como-criar-um-dataset-para-deep-learning/


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2 as cv
import os
from deteccao import deteccaoMoedas
import matplotlib.pyplot as plt
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path do modelo treinado")
ap.add_argument("-l", "--labelbin", required=True,
    help="path dos labels")
ap.add_argument("-i", "--image", required=True,
    help="path da imagem")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
listamoedas = deteccaoMoedas.detect(img)
#count = 1
sum_moedas = 0
for image in listamoedas:
    output = image.copy()
    image = cv.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("[INFO] carregando rede...")
    model = load_model(args["model"])
    lb = pickle.loads(open(args["labelbin"], "rb").read())

    print("[INFO] classificando imagem...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]

    filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
    label = "{}: {:.2f}%".format(label, proba[idx] * 100)
    output = imutils.resize(output, width=400)
    cv.putText(output, label, (25, 40),  cv.FONT_HERSHEY_SIMPLEX,
        1, (255, 0, 0), 3)

    print("[INFO] {}".format(label))
    if "5centavos" in label:
        sum_moedas+=0.05
    if "10centavos" in label:
        sum_moedas+=0.1
    if "50centavos" in label:
        sum_moedas+=0.5
    cv.imwrite("moedas_detectadas/{}.jpg".format(uuid.uuid4()), output)

scale_percent = 23 # por cento do tamanho original
width = int(img.shape[1] * scale_percent/100)
height = int(img.shape[0] * scale_percent/100)
dim = (width, height)
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
cv.putText(resized, "Valor Calculado: R$ {:.2f}".format(sum_moedas), (50, 70),  cv.FONT_HERSHEY_SIMPLEX,
        2, (255, 0, 0), 3)
cv.imwrite("moedas_detectadas/{}.jpg".format(uuid.uuid4()), resized)
cv.imshow('Valor Calculado', resized)
print("[INFO] SOMA MOEDAS: R$ {:.2f}".format(sum_moedas))
cv.waitKey(0)
cv.destroyAllWindows()