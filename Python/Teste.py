import cv2
import numpy as np
from matplotlib import pyplot as plt
#Função para pegar o valor do pixel
def valor_pixel(crop_img, center, x, y):
    n_valor = 0
    try:
        if img[x][y] >= center:
            n_valor = 1
    except:
        pass
    return n_valor

#Calcula a vizinhança dos pixels
def lbp_calc_pixel(crop_img, x, y):

    center = crop_img[x][y]
    val_ar = []
    val_ar.append(valor_pixel(crop_img, center, x-1, y+1))     # top_right
    val_ar.append(valor_pixel(crop_img, center, x, y+1))       # right
    val_ar.append(valor_pixel(crop_img, center, x+1, y+1))     # bottom_right
    val_ar.append(valor_pixel(crop_img, center, x+1, y))       # bottom
    val_ar.append(valor_pixel(crop_img, center, x+1, y-1))     # bottom_left
    val_ar.append(valor_pixel(crop_img, center, x, y-1))       # left
    val_ar.append(valor_pixel(crop_img, center, x-1, y-1))     # top_left
    val_ar.append(valor_pixel(crop_img, center, x-1, y))       # top
    for i in range(1,10):
        print(val_ar[i])


for i in range(0, l):
        for j in range(0, c):
             img_lbp[i, j] = lbp_calculated_pixel(crop_img, i, j)
