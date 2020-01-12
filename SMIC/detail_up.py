import cv2
import os
from numpy import *
patth='../../datasets/SMIC_TIM10/'
to_patth='../../datasets/SMIC_DETAIL_TIM10/'
emotionlisting=os.listdir(patth)
def multiScaleSharpen(img ,radius):
    h,w,chan = img.shape
    GaussBlue1 = zeros(img.shape,dtype = uint8)
    GaussBlue2 = zeros(img.shape, dtype=uint8)
    GaussBlue3 = zeros(img.shape, dtype=uint8)
    Dest_float_img = zeros(img.shape, dtype=float32)
    Dest_img = zeros(img.shape, dtype=uint8)
    w1 = 0.5
    w2 = 0.5
    w3 = 0.25
    GaussBlue1 = cv2.GaussianBlur(img,(radius,radius),1)
    GaussBlue2 = cv2.GaussianBlur(img,(radius*2-1,radius*2-1),2)
    GaussBlue3 = cv2.GaussianBlur(img,(radius*4-1,radius*4-1),4)
    for i in range(0,h):
        for j in range(0,w):
            for k in range(0,chan):
                Src = img.item(i,j,k)
                D1 = Src-GaussBlue1.item(i,j,k)
                D2 = GaussBlue1.item(i,j,k) - GaussBlue2.item(i,j,k)
                D3 = GaussBlue2.item(i,j,k) - GaussBlue3.item(i,j,k)
                if(D1 > 0):
                    sig = 1
                else:
                    sig = -1
                Dest_float_img.itemset((i,j,k),(1-w1*sig)*D1+w2*D2+w3*D3+Src)
    Dest_img = cv2.convertScaleAbs(Dest_float_img)
    return Dest_img

if __name__ == '__main__':
    for emotiondata in emotionlisting:
        negativepath = patth + emotiondata + "/"
        to_negativepath = to_patth + emotiondata + "/"
        directorylisting = os.listdir(negativepath)
        for video in directorylisting:
            videopath = negativepath + video
            to_videopath = to_negativepath + video
            os.mkdir(to_videopath)
            frames = []
            framelisting = os.listdir(videopath)
            framerange = [x for x in range(10)]
            for frame in framerange:
                imagepath = videopath + "/" + framelisting[frame]
                img = cv2.imread(imagepath)
                multiScaleSharpen_out = zeros(img.shape, dtype=uint8)
                multiScaleSharpen_out = multiScaleSharpen(img, 5)
                rotate_img_path = to_videopath + "/" + framelisting[frame]
                cv2.imwrite(rotate_img_path, multiScaleSharpen_out)



