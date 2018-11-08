# -------------------------- TREINADOR PARA TODOS OS ALGORITMOS EM RECONHECIMENTO FACIAL-------------------------------------------

import os                                               # Importando o OS para passar uma pasta como caminho
import cv2                                              # Importando o OpenCV 
import numpy as np                                      # Importando o Numpy 
from PIL import Image                                   # Importando o Image 

LBPHFace = cv2.face.LBPHFaceRecognizer_create(1, 1, 7,7) # CRIA O MÉTODO LBPH DE RECONHECIMENTO FACIAL

path = 'C:/Users/joaob/Documents/Github/face-recog/Python/dataSet'   # Caminho para a pasta que contém as fotos
def getImageWithID (path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')          # Abre a imagem e converte para cinza
        faceImage = faceImage.resize((110,110))                 # Redimenciona a imagem para que o método reconhecedor LBPH possa ser treinado
        faceNP = np.array(faceImage, 'uint8')                   # Converte a imagem em um array Numpy 
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # Recupera o ID do array
        FaceList.append(faceNP)                                 # Acrescenta o array Numpy para a lista
        IDs.append(ID)                                          # Acrescenta o ID para a lista de IDs
        cv2.imshow('Training Set', faceNP)                      # Mostra as imagens na lista
        cv2.waitKey(125)
    return np.array(IDs), FaceList                              # Os IDs serão convertidos em um array Numpy
IDs, FaceList = getImageWithID(path)

# ------------------------------------ TREINANDO O RECONHECIMENTO FACIAL ----------------------------------------
print('TREINANDO OS ARQUIVOS......')
LBPHFace.train(FaceList, IDs)
print('MÉTODO LBPH RECONHECIDOR FACIAL COMPLETADO...')
LBPHFace.save('Recogniser/trainingDataLBPH.xml')
print ('TODOS ARQUIVOS XML FORAM SALVOS...')

cv2.destroyAllWindows()