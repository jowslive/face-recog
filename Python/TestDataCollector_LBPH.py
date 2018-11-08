# ------------ COLETA A DATA USANDO O MÉTODO LBPH EM UMA CERTA IMAGEM E SALVA A DATA EM UM ARQUIVO DE TEXTO -------------
# ---------------------- SALVA A DATA GERADA EM 3 ARQUIVOS DE TEXTO E OS "PREENCHE" --------------------------------------


import os               # importando the OS for path
import cv2              # importando the OpenCV library
import numpy as np      # importando Numpy library
from PIL import Image   # importando Image library
import matplotlib.pyplot as plt # Importando Plot library
import NameFind

face_cascade = cv2.CascadeClassifier('C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_frontalcatface.xml')

path = 'C:/Users/joaob/Documents/Github/face-recog/Python/dataSet/'        # Caminho para as fotos que serão comparadas

img = cv2.imread('C:/Users/joaob/Documents/Github/face-recog/Python/dataSet/User.2.1.jpg')        # Caminho da imagem para ser checada

def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []

    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')          # Abre a imagem e a converte para cinza
        print(str((faceImage.size)))
        faceImage = faceImage.resize((110, 110))                # Redimenciona a imagem
        faceNP = np.array(faceImage, 'uint8')                   # Converte a imagem para um array Numpy - Valor de 0 a 255
        print(str((faceNP.shape)))
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # Pega o ID do array
        FaceList.append(faceNP)                                 # Acrescenta o array Numpy para a lista
        IDs.append(ID)                                          # Acrescenta o ID para a lista de IDs

    return np.array(IDs), FaceList                              # Os IDs são convertidos para uma lista de array Numpy 

face_number = 1
IDs, FaceList = getImageWithID(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                   # Converte a imagem para cinza
faces = face_cascade.detectMultiScale(gray, 1.3, 4)            # Detecta as faces e as posições
radTrain = open("C:/Users/joaob/Documents/Github/face-recog/Python/SaveData/LBPHLBPH_PIXEL_RADIUS.txt", "w+")   # Abre o arquivo para salvar os dados
neiTrain = open("C:/Users/joaob/Documents/Github/face-recog/Python/SaveData/LBPHLBPH_NEIGHBOURS.txt", "w+")     # Abre o arquivo para salvar os dados
cellTrain = open("C:/Users/joaob/Documents/Github/face-recog/Python/SaveData/LBPHLBPH_CELLS.txt", "w+")         # Abre o arquivo para salvar os dados

for (x, y, w, h) in faces:
    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))
    radPix = 1
    rad_tabal_ID = []
    rad_tabal_conf = []
    # --------------------------- Executa os testes para o raio a partir do centro ----------------
    for _ in range(54):
        recog = cv2.face.LBPHFaceRecognizer_create(radPix)     # Criando o método LBPH FACE
        print('Treinando o pixel  ' + str(radPix) + ' em relação ao centro')
        recog.train(FaceList, IDs)                            # O reconhecedor é treinado usando as imagens
        print('O Método LBPH de reconhecimento facial está sendo treinado,por favor aguarde!')
        ID, conf = recog.predict(Face)
        rad_tabal_ID.append(ID)
        rad_tabal_conf.append(conf)
        radTrain.write(str(ID) + "," + str(conf) + "\n")
        print ("No raio: " + str(radPix) + " o ID sendo treinado é: " + str(ID) + " A confiança é: " + str(conf))
        radPix = radPix + 1
    # ---------------------------------------- 1° PAINEL -----------------------------------------------------
    plt.subplot(2, 1, 1)
    plt.plot(rad_tabal_ID)
    plt.title('ID contrário ao raio dos pixels', fontsize=10)
    plt.axis([0, radPix, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Raio (em pixels)', fontsize=8)
    p2 = plt.subplot(2, 1, 2)
    plt.plot(rad_tabal_conf, 'red')
    plt.title('Confiança', fontsize=10)
    p2.set_xlim(xmin=0)
    p2.set_xlim(xmax=radPix)
    plt.ylabel('Confiança (em relação ao raio dos pixels)', fontsize=8)
    plt.xlabel('Raio (em pixels)', fontsize=8)
    plt.tight_layout()
    plt.show()
    # ---------------------------  Executa os testes para os vizinhos -----------------------------
    radPixel = input('Digite o raio de pixels ideal: ')    # ALTERA O RAIO DO PIXEL CASO UM VALOR MELHOR SEJA ACHADO
    neighbour = 1
    nei_ID = []
    nei_conf = []
    for _ in range(13):
        recog = cv2.face.LBPHFaceRecognizer_create(int(radPixel), int(neighbour))  # Criando o método de reconhecimento facial
        print('TREINANDO PARA OS  ' + str(neighbour) + ' VIZINHOS')
        recog.train(FaceList, IDs)                                      # O reconhecedor é treinado usando as imagens fornecidas
        print('MÉTODO LBPH DE RECONHECIMENTO FACIAL TREINADO COM SUCESSO!')
        ID, conf = recog.predict(Face)
        nei_ID.append(ID)
        nei_conf.append(conf)
        neiTrain.write(str(ID) + "," + str(conf) + '\n')
        print ('PARA O RAIO: ' + str(radPixel) + " E O " + str(neighbour) + "VIZINHO, O ID É : " + str(ID) + " CONFIANÇA: " + str(conf))
        neighbour = neighbour + 1
    # ---------------------------------------- 2° PAINEL -----------------------------------------------------
    plt.subplot(2, 1, 1)
    plt.plot(nei_ID)
    plt.title('ID contra o número de vizinhos', fontsize=10)
    plt.axis([0, neighbour, 10, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Número de vizinhos', fontsize=8)
    p2 = plt.subplot(2, 1, 2)
    plt.plot(nei_conf, 'red')
    plt.title('ID contra o número de vizinhos', fontsize=10)
    p2.set_xlim(xmin=0)
    p2.set_xlim(xmax=neighbour)
    plt.ylabel('Confiança', fontsize=8)
    plt.xlabel('Número de vizinhos', fontsize=8)
    plt.tight_layout()
    plt.show()
    # ---------------------------  Executa certos testes para o número de Células fornecidas -----------------------------
    neighbour = input('DIGITE O NÚMERO IDEAL DE VIZINHOS: ')   # Muda o valor do vizinho caso algum valor melhor seja achado
    cellVal = 1
    cell_ID = []
    cell_conf = []
    for _ in range(50):
        recog = cv2.face.LBPHFaceRecognizer_create(int(radPixel), int(neighbour), int(cellVal), int(cellVal))  # Criando o método de reconhecimento facial
        print('TREINANDO AS CELULAS  ' + str(cellVal) )
        recog.train(FaceList, IDs)                                          # O reconhecedor é treinado usando as imagens fornecidas
        print('MÉTODO LBPH DE RECONHECIMENTO FACIAL TREINADO COM SUCESSO!')
        ID, conf = recog.predict(Face)
        cell_ID.append(ID)
        cell_conf.append(conf)
        cellTrain.write(str(ID) + "," + str(conf) + "\n")
        print ('PARA OS RAIOS: ' + str(radPixel) + " , " + str(neighbour) + "VIZINHOS E VALOR DA CÉULAR " + str(cellVal) + ", O ID É: " + str(ID) + " A CONFIANÇA É: " + str(conf))
        cellVal = cellVal + 1
    NameFind.tell_time_passed()


    # ------------------------------------------- TODOS PAINÉIS -----------------------------------------------------------
    plt.subplot(3, 2, 1)
    plt.plot(rad_tabal_ID)
    plt.title('ID contra o raio do Pixel', fontsize=10)
    plt.axis([0, 53, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Raio (em Pixels)', fontsize=8)
    plt.subplot(3, 2, 2)
    plt.plot(rad_tabal_conf, 'red')
    plt.title('Confiança contra o Raio do Pixel', fontsize=10)
    plt.ylabel('Confiança', fontsize=8)
    plt.xlabel('Raio (em Pixels)', fontsize=8)
    plt.subplot(3, 2, 3)
    plt.plot(nei_ID)
    plt.title('ID contra o número de vizinhos', fontsize=10)
    plt.axis([0, 12, 10, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Número de vizinhos', fontsize=8)
    plt.subplot(3, 2, 4)
    plt.plot(nei_conf, 'red')
    plt.title('ID contra o número de vizinhos', fontsize=10)
    plt.ylabel('Confiança', fontsize=8)
    plt.xlabel('Número de vizinhos', fontsize=8)
    plt.subplot(3, 2, 5)
    plt.plot(cell_ID)
    plt.title('ID contra o número de células', fontsize=10)
    plt.axis([0, 49, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Número de células', fontsize=8)
    plt.subplot(3, 2, 6)
    plt.plot(cell_conf, 'red')
    plt.title('ID contra o número de células', fontsize=10)
    plt.ylabel('Confiança', fontsize=8)
    plt.xlabel('Número de células', fontsize=8)
    plt.tight_layout()
    plt.show()

    face_number = face_number + 1

radTrain.close()
neiTrain.close()
cellTrain.close()
cv2.destroyAllWindows()