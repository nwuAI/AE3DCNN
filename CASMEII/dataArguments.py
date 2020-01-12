
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
patth='C:/Users/HP/Desktop/dataa/CASME2_TIM26/'
emotionlisting=os.listdir(patth)
for emotiondata in emotionlisting:
    negativepath = patth +emotiondata+"/"
    directorylisting = os.listdir(negativepath)
    for video in directorylisting:
        videopath = negativepath + video
        os.mkdir(negativepath + 'rotate5_' + video)
        os.mkdir(negativepath + 'rotate10_' + video)
        os.mkdir(negativepath + 'rotate15_' + video)
        os.mkdir(negativepath + 'rotater5_' + video)
        os.mkdir(negativepath + 'rotater10_' + video)
        os.mkdir(negativepath + 'rotater15_' + video)
        frames = []
        framelisting = os.listdir(videopath)  
        framerange = [x for x in range(26)] 
        for frame in framerange:
            imagepath = videopath + "/" + framelisting[
                frame]  
            image = cv2.imread(imagepath)
          
            (height, width) = image.shape[:2]  
            center = (height // 2, width // 2)  
            
            matrix5 = cv2.getRotationMatrix2D(center, angle=5, scale=1)  
            rotate_img = cv2.warpAffine(image, matrix5, (width, height))  
            rotate_img_path = negativepath + 'rotate5_' + video + "/" + framelisting[
                frame]  
            cv2.imwrite(rotate_img_path, rotate_img)
            #
            matrix10 = cv2.getRotationMatrix2D(center, angle=10, scale=1)                   
            rotate_img = cv2.warpAffine(image, matrix10, (width, height))
            rotate_img_path = negativepath + 'rotate10_' + video + "/" + framelisting[frame]
            cv2.imwrite(rotate_img_path, rotate_img)
            # 
            matrix15 = cv2.getRotationMatrix2D(center, angle=15, scale=1)
            rotate_img = cv2.warpAffine(image, matrix15, (width, height))
            rotate_img_path = negativepath + 'rotate15_' + video + "/" + framelisting[frame]
            cv2.imwrite(rotate_img_path, rotate_img)
            # 
            matrixr5 = cv2.getRotationMatrix2D(center, angle=-5, scale=1)
            rotate_img = cv2.warpAffine(image, matrixr5, (width, height))
            rotate_img_path = negativepath + 'rotater5_' + video + "/" + framelisting[frame]
            cv2.imwrite(rotate_img_path, rotate_img)
            # 
            matrixr10 = cv2.getRotationMatrix2D(center, angle=-10, scale=1)
            rotate_img = cv2.warpAffine(image, matrixr10, (width, height))
            rotate_img_path = negativepath + 'rotater10_' + video + "/" + framelisting[frame]
            cv2.imwrite(rotate_img_path, rotate_img)
            # 
            matrixr15 = cv2.getRotationMatrix2D(center, angle=-15, scale=1)
            rotate_img = cv2.warpAffine(image, matrixr15, (width, height))
            rotate_img_path = negativepath + 'rotater15_' + video + "/" + framelisting[frame]
            cv2.imwrite(rotate_img_path, rotate_img)

