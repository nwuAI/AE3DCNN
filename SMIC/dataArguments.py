
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
negativepath = 'C:/Users/HP/Desktop/Dataset/Negative/'
positivepath = 'C:/Users/HP/Desktop/Dataset/Positive/'
surprisepath = 'C:/Users/HP/Desktop/Dataset/Surprise/'

directorylisting = os.listdir(negativepath)
for video in directorylisting:
    videopath = negativepath + video
    os.mkdir(negativepath + 'rotate5_'+video)
    os.mkdir(negativepath + 'rotate10_' + video)
    os.mkdir(negativepath + 'rotate15_' + video)
    os.mkdir(negativepath + 'rotater5_'+video)
    os.mkdir(negativepath + 'rotater10_' + video)
    os.mkdir(negativepath + 'rotater15_' + video)
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(26)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           
           (height,width)=image.shape[:2]
           center=(height//2,width//2)
           #5
           matrix5=cv2.getRotationMatrix2D(center,angle=5,scale=1)
           rotate_img=cv2.warpAffine(image,matrix5,(width,height))
           rotate_img_path=negativepath + 'rotate5_'+video+"/" +framelisting[frame]
           cv2.imwrite(rotate_img_path,rotate_img)
           # 10
           matrix10 = cv2.getRotationMatrix2D(center, angle=10, scale=1)
           rotate_img = cv2.warpAffine(image, matrix10, (width, height))
           rotate_img_path = negativepath + 'rotate10_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # 15
           matrix15 = cv2.getRotationMatrix2D(center, angle=15, scale=1)
           rotate_img = cv2.warpAffine(image, matrix15, (width, height))
           rotate_img_path = negativepath + 'rotate15_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -5
           matrixr5 = cv2.getRotationMatrix2D(center, angle=-5, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr5, (width, height))
           rotate_img_path = negativepath + 'rotater5_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -10
           matrixr10 = cv2.getRotationMatrix2D(center, angle=-10, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr10, (width, height))
           rotate_img_path = negativepath + 'rotater10_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -15
           matrixr15 = cv2.getRotationMatrix2D(center, angle=-15, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr15, (width, height))
           rotate_img_path = negativepath + 'rotater15_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)

directorylisting = os.listdir(positivepath)
for video in directorylisting:
    videopath = positivepath + video
    os.mkdir(positivepath + 'rotate5_'+video)
    os.mkdir(positivepath + 'rotate10_' + video)
    os.mkdir(positivepath + 'rotate15_' + video)
    os.mkdir(positivepath + 'rotater5_'+video)
    os.mkdir(positivepath + 'rotater10_' + video)
    os.mkdir(positivepath + 'rotater15_' + video)
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(26)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)

           (height,width)=image.shape[:2]
           center=(height//2,width//2)
           #5
           matrix5=cv2.getRotationMatrix2D(center,angle=5,scale=1)
           rotate_img=cv2.warpAffine(image,matrix5,(width,height))
           rotate_img_path=positivepath + 'rotate5_'+video+"/" +framelisting[frame]
           cv2.imwrite(rotate_img_path,rotate_img)
           # 10
           matrix10 = cv2.getRotationMatrix2D(center, angle=10, scale=1)
           rotate_img = cv2.warpAffine(image, matrix10, (width, height))
           rotate_img_path = positivepath + 'rotate10_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # 15
           matrix15 = cv2.getRotationMatrix2D(center, angle=15, scale=1)
           rotate_img = cv2.warpAffine(image, matrix15, (width, height))
           rotate_img_path = positivepath + 'rotate15_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -5
           matrixr5 = cv2.getRotationMatrix2D(center, angle=-5, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr5, (width, height))
           rotate_img_path = positivepath + 'rotater5_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -10
           matrixr10 = cv2.getRotationMatrix2D(center, angle=-10, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr10, (width, height))
           rotate_img_path = positivepath + 'rotater10_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -15
           matrixr15 = cv2.getRotationMatrix2D(center, angle=-15, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr15, (width, height))
           rotate_img_path = positivepath + 'rotater15_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)


directorylisting = os.listdir(surprisepath)
for video in directorylisting:
    videopath = surprisepath + video
    os.mkdir(surprisepath + 'rotate5_'+video)
    os.mkdir(surprisepath + 'rotate10_' + video)
    os.mkdir(surprisepath + 'rotate15_' + video)
    os.mkdir(surprisepath + 'rotater5_'+video)
    os.mkdir(surprisepath + 'rotater10_' + video)
    os.mkdir(surprisepath + 'rotater15_' + video)
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(26)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           (height,width)=image.shape[:2]
           center=(height//2,width//2)
           #5
           matrix5=cv2.getRotationMatrix2D(center,angle=5,scale=1)
           rotate_img=cv2.warpAffine(image,matrix5,(width,height))
           rotate_img_path=surprisepath + 'rotate5_'+video+"/" +framelisting[frame]
           cv2.imwrite(rotate_img_path,rotate_img)
           # 10
           matrix10 = cv2.getRotationMatrix2D(center, angle=10, scale=1)
           rotate_img = cv2.warpAffine(image, matrix10, (width, height))
           rotate_img_path = surprisepath + 'rotate10_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # 15
           matrix15 = cv2.getRotationMatrix2D(center, angle=15, scale=1)
           rotate_img = cv2.warpAffine(image, matrix15, (width, height))
           rotate_img_path = surprisepath + 'rotate15_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -5
           matrixr5 = cv2.getRotationMatrix2D(center, angle=-5, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr5, (width, height))
           rotate_img_path = surprisepath + 'rotater5_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -10
           matrixr10 = cv2.getRotationMatrix2D(center, angle=-10, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr10, (width, height))
           rotate_img_path = surprisepath + 'rotater10_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
           # -15
           matrixr15 = cv2.getRotationMatrix2D(center, angle=-15, scale=1)
           rotate_img = cv2.warpAffine(image, matrixr15, (width, height))
           rotate_img_path = surprisepath + 'rotater15_' + video + "/" + framelisting[frame]
           cv2.imwrite(rotate_img_path, rotate_img)
