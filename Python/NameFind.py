#     ----------- FUNÇÃO PARA LER O ARQUIVO E ADICIONAR OS NOMES + IDS EM "VASILHAS"  -----------

import cv2
import math
import time

now_time = time.perf_counter()

face = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
glass_cas = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

WHITE = [255, 255, 255]

def FileRead():
    Info = open("C:/Users/joaob/Documents/Github/face-recog/Python/Names.txt", "r")         # Abrindo o arquivo Names.txt em modo de leitura
    NAME = []                                      # "Vasilha" para guardar os nomes
    while (True):                                  # Lê todas as linhas no arquivo e guarda elas em duas "vasilhas"
        Line = Info.readline() 
        if Line == '':
            break
        NAME.append (Line.split(",")[1].rstrip())

    return NAME                                    # Retorna as duas "vasilhas"

Names = FileRead()                                 # Rodando a função acima , obtemos o ID e os nomes contidos na Vasilha

#     ------------------- FUNÇAÕ PARA ENCONTRAR O NOME  -------------------


def ID2Name(ID, conf):
    if ID > 0:
        NameString = "--"        # Encontra o nome usando o ID do Index
    else:
        NameString = " Face não reconhecida! " 

    return NameString


#     ------------------- DESENHA O QUADRADO EM VOLTA DA FACE  -------------------


def DispID(x, y, w, h, NAME, Image):

    #  ------------------- POSIÇÃO DA CAIXA DE IDENTIFICAÇÃO  -------------------

    Name_y_pos = y - 10
    Name_X_pos = x + w/2 - (len(NAME)*7/2)

    if Name_X_pos < 0:
        Name_X_pos = 0
    elif (Name_X_pos +10 + (len(NAME) * 7) > Image.shape[1]):
          Name_X_pos= Name_X_pos - (Name_X_pos +10 + (len(NAME) * 7) - (Image.shape[1]))
    if Name_y_pos < 0:
        Name_y_pos = Name_y_pos = y + h + 10

 #  -------------------   DESENHO DA CAIXA + ID  -------------------

    draw_box(Image, x, y, w, h)


    cv2.rectangle(Image, (int(Name_X_pos-10), int(Name_y_pos-25)), (int(Name_X_pos +10 + (len(NAME) * 7)), int(Name_y_pos-1)), (0,0,0), -2)   # Desenha um retangulo
    cv2.rectangle(Image, (int(Name_X_pos-10), int(Name_y_pos-25)), (int(Name_X_pos +10 + (len(NAME) * 7)), int(Name_y_pos-1)), WHITE, 1)
    cv2.putText(Image, NAME, (int(Name_X_pos), int(Name_y_pos - 10)), cv2.FONT_HERSHEY_DUPLEX, int(1), WHITE)  # "Printa" o nome do ID encontrado


def draw_box(Image, x, y, w, h):
    cv2.line(Image, (x, y), (x + (w//5) ,y), WHITE, 2)
    cv2.line(Image, (x+((w//5)*4), y), (x+w, y), WHITE, 2)
    cv2.line(Image, (x, y), (x, y+(h//5)), WHITE, 2)
    cv2.line(Image, (x+w, y), (x+w, y+(h//5)), WHITE, 2)
    cv2.line(Image, (x, (y+(h//5*4))), (x, y+h), WHITE, 2)
    cv2.line(Image, (x, (y+h)), (x + (w//5) ,y+h), WHITE, 2)
    cv2.line(Image, (x+((w//5)*4), y+h), (x + w, y + h), WHITE, 2)
    cv2.line(Image, (x+w, (y+(h//5*4))), (x+w, y+h), WHITE, 2)
