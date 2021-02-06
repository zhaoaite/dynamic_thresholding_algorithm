# -*- coding: utf-8 -*-
import cv2,os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from removehighlight import remove_connectivity





#Binary processing

#Set the threshold threshold, the pixel value is less than the threshold, the value is 0, the pixel value is greater than the threshold, the value is 1

#The specific threshold needs to be tried many times, and the effect of different thresholds is different
def get_table(threshold=115):
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    return table

def calcGrayHist(image):
    rows,clos = image.shape
    grahHist = np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(clos):
            grahHist[image[r][c]] +=1
    return grahHist



#gray image generation
def tif2gray(img_path):
#    img_path = '/home/zat/project/STED-20200915T020146Z-001/STED/STED (20200909)/Sample 1/Cell 3/Image_Area01_Channel_tubulin.tif'
    uint16_img = cv2.imread(img_path, -1)
    uint16_img -= uint16_img.min()
    uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
    uint16_img *= 255
    new_uint16_img = uint16_img.astype(np.uint8)
    #cv2.imshow('UINT8', uint8_img)
#    cv2.imshow('UINT16', new_uint16_img)
    #hsv=cv2.cvtColor(new_uint16_img,cv2.COLOR_RGB2HSV)
    #cv2.imshow('hsv', hsv)

    
    name = img_path[-15:]
    cv2.imwrite(name, new_uint16_img)
    
#    new_path=os.path.join('/home/zat/project/STED_NEW/',path,subpath,subsubpath)
#    print('---------------')
#    if not os.path.exists(new_path):
#        print(new_path)
#        os.makedirs(new_path)
#    cv2.imwrite(new_path+'/'+file, new_uint16_img)
    return new_uint16_img

def denoising(new_img_path):
    new_img = cv2.imread(new_img_path,cv2.IMREAD_GRAYSCALE)
    Maximg = np.max(new_img)
    Minimg = np.min(new_img)
    Omin,Omax = 0,260
    a = float(Omax - Omin)/(Maximg - Minimg)
    b = Omin - a*Minimg
    O = a*new_img + b
    O = O.astype(np.uint8)
#    a=1
#    O = float(a)*new_img
#    O[0>255] = 255
#    O = np.round(O)
#    O = O.astype(np.uint8)
#    binary_im=cv2.GaussianBlur(new_img, (7, 7), 0)
#    binary_im = cv2.medianBlur(O,5)
#    binary_im = cv2.blur(O,(3,3))
    return O
    
def coloring_local(new_img_path,file_name):
    new_img = cv2.imread(new_img_path)
    HSV=cv2.cvtColor(new_img,cv2.COLOR_BGR2HSV)
    HSV_ex_high=cv2.cvtColor(new_img,cv2.COLOR_BGR2HSV)
    HSV_ex_low=cv2.cvtColor(new_img,cv2.COLOR_BGR2HSV)
        

    
    retn_img=HSV
    h,w,_=HSV.shape

    
    thresh_point=np.array([0,0,0])
    z=range(0,h)
    d=range(0,w)
    thresh_point_num=0
    for x in z:
        for y in d:
            b = HSV[x,y]
            if b.any()!=np.array([0,0,0]).any():
                thresh_point_num+=1
                thresh_point=thresh_point+b
    thresh_point=thresh_point/thresh_point_num 
    print('thresh_point',thresh_point)

    for x in z:
        for y in d:
            b = HSV[x,y]
            if 'vash' in file_name:
                if b[2] >= round(thresh_point[2])+30 and b[2] <= 255:
                    HSV_ex_low[x,y]=[0 ,0 ,221]
                else:
                    HSV_ex_low[x,y]=[0 ,0 ,0]
                
            else:    
                if b[2] >= round(thresh_point[2])+30 and b[2] <= 255:
                    HSV_ex_high[x,y]=[34 ,255 ,255]
                else:
                    HSV_ex_high[x,y]=[0 ,0 ,0]


    if 'vash' in file_name:
#        HSV_ex_low = cv2.GaussianBlur(HSV_ex_low, (3, 3), 0)
        thresh, binary = cv2.threshold(HSV_ex_low, round(thresh_point[2]), 255,cv2.THRESH_BINARY)
        retn_img=binary
        
    else:
#        HSV_ex_high = cv2.GaussianBlur(HSV_ex_high, (3, 3), 0)
        thresh, binary = cv2.threshold(HSV_ex_high, round(thresh_point[2]), 255,cv2.THRESH_BINARY)
        retn_img=binary
    return retn_img
    


def coloring_whole(new_img_path,file_name):
    #remove noise
#    new_img_path = 'Image_Area02_Channel_tubulin.jpg'
    new_img = cv2.imread(new_img_path)
    #dst = cv2.fastNlMeansDenoisingColored(new_img,None,20,20,9,21)
    #cv2.imwrite('no_noise.jpg', dst)
    
    HSV=cv2.cvtColor(new_img,cv2.COLOR_BGR2HSV)
    HSV_ex_high=cv2.cvtColor(new_img,cv2.COLOR_BGR2HSV)
    HSV_ex_low=cv2.cvtColor(new_img,cv2.COLOR_BGR2HSV)
    retn_img=HSV
    h,w,_=HSV.shape
    
    name = file_name[-10:]
    cv2.imwrite(name, HSV)
    
    
    thresh_point=np.array([0,0,0])
    z=range(0,h)
    d=range(0,w)
    thresh_point_num=0
    for x in z:
        for y in d:
            b = HSV[x,y]
            if b.any()!=np.array([0,0,0]).any():
                thresh_point_num+=1
                thresh_point=thresh_point+b
    thresh_point=thresh_point/thresh_point_num 
    print('thresh_point',thresh_point)


    #    remove highlight
    for x in z:
        for y in d:
            b = HSV[x,y]
##            04/09
#            if 'Channel_vash' in file_name:
#                if b[0]>=0 and b[0]<=180 and b[1] >= 0 and b[1] <= 40 and b[2] >= 60 and b[2] <= 220:
#                    HSV_ex_low[x,y]=[0 ,0 ,221]
#                else:
#                    HSV_ex_low[x,y]=[0 ,0 ,0]
#                
#            else:    
#                if b[0]>=0 and b[0]<=180 and b[1] >= 0 and b[1] <= 43 and b[2] >= 90 and b[2] <= 220:
#                    HSV_ex_high[x,y]=[34 ,255 ,255]
#                else:
#                    HSV_ex_high[x,y]=[0 ,0 ,0]


##            09/09
#            if 'Channel_vash' in file_name:
#                if b[0]>=0 and b[0]<=180 and b[1] >= 0 and b[1] <= 43 and b[2] >= 100 and b[2] <= 220:
#                    HSV_ex_low[x,y]=[0 ,0 ,221]
#                else:
#                    HSV_ex_low[x,y]=[0 ,0 ,0]
#                
#            else:    
#                if b[0]>=0 and b[0]<=180 and b[1] >= 0 and b[1] <= 43 and b[2] >= 100 and b[2] <= 220:
#                    HSV_ex_high[x,y]=[34 ,255 ,255]
#                else:
#                    HSV_ex_high[x,y]=[0 ,0 ,0]
                    
            if 'vash' in file_name:
                if b[0]>=0 and b[0]<=180 and b[1] >= 0 and b[1] <= 43 and b[2] >= round(thresh_point[2])+30 and b[2] <= 220:
                    HSV_ex_low[x,y]=[0 ,0 ,221]
                else:
                    HSV_ex_low[x,y]=[0 ,0 ,0]
                
            else:    
                if b[0]>=0 and b[0]<=180 and b[1] >= 0 and b[1] <= 43 and b[2] >= round(thresh_point[2])+30 and b[2] <= 220:
                    HSV_ex_high[x,y]=[34 ,255 ,255]
                else:
                    HSV_ex_high[x,y]=[0 ,0 ,0]

    if 'vash' in file_name:
        name = file_name[-9:]
        cv2.imwrite(name, HSV_ex_low)
        HSV_ex_low = cv2.GaussianBlur(HSV_ex_low, (3, 3), 0)
        name = file_name[-8:]
        cv2.imwrite(name, HSV_ex_low)
#        cv2.imshow("imageHSV",HSV_ex_low)
#        cv2.imwrite(new_img_path+'.tif', HSV_ex_low)
#        HSV_ex_low = cv2.fastNlMeansDenoisingColored(HSV_ex_high,None,10,10,7,21)
#        HSV_ex_low = cv2.medianBlur(HSV_ex_low,3)
        thresh, binary = cv2.threshold(HSV_ex_low, round(thresh_point[2])+30, 255,cv2.THRESH_BINARY)
        retn_img=binary
        
    else:
        if round(thresh_point[2])+30>=65:
            name = file_name[-7:]
            cv2.imwrite(name, HSV_ex_high)
#            HSV_ex_high = cv2.bilateralFilter(HSV_ex_high, int(round(thresh_point[2]))+30, 75, 75)
            HSV_ex_high = cv2.GaussianBlur(HSV_ex_high, (5, 5), 0)
            name = file_name[-6:]
            cv2.imwrite(name, HSV_ex_high)
#            HSV_ex_high = cv2.medianBlur(HSV_ex_high,3)
#            HSV_ex_high = cv2.cvtColor(HSV_ex_high, cv2.COLOR_BGR2GRAY)
            thresh, binary = cv2.threshold(HSV_ex_high, round(thresh_point[2])+30, 255,cv2.THRESH_BINARY)
            retn_img=binary
        else:
            name = file_name[-7:]
            cv2.imwrite(name, HSV_ex_high)
            HSV_ex_high = cv2.GaussianBlur(HSV_ex_high, (3, 3), 0)
            name = file_name[-6:]
            cv2.imwrite(name, HSV_ex_high)
#            HSV_ex_high = cv2.bilateralFilter(HSV_ex_high, int(round(thresh_point[2]))+30, 75, 75)
#            HSV_ex_high = cv2.medianBlur(HSV_ex_high,3)
#            HSV_ex_high = cv2.cvtColor(HSV_ex_high, cv2.COLOR_BGR2GRAY)
            thresh, binary = cv2.threshold(HSV_ex_high, round(thresh_point[2])+30, 255,cv2.THRESH_BINARY)
            retn_img=binary
#            retn_img = cv2.dilate(retn_img,np.ones((2,1),np.uint8),iterations = 2)            
                       
   

        
        # remove isolated points       
#        num_labels,labels,stats,centers = cv2.connectedComponentsWithStats(retn_img, connectivity=8,ltype=cv2.CV_32S)
#        new_image = retn_img.copy()
#        for label in range(num_labels):
#            if stats[label,cv2.CC_STAT_AREA] == 1:
#                new_image[labels == label] = 0
#        retn_img=new_image
#        retn_img = cv2.medianBlur(retn_img,3)
        
        
#        retn_img = cv2.fastNlMeansDenoising(retn_img,3,3,7,21)
#        cv2.imshow('HSV_ex',HSV_ex_high)
#        cv2.imwrite(new_img_path+'.tif', HSV_ex_high)
    return retn_img


def merge(path1,path2):
#    path1='Image_Area02_Channel_tubulin.jpg.tif'
#    path2='Image_Area02_Channel_vash2.jpg.tif'
    new_img1 = cv2.imread(path1)
    new_img2 = cv2.imread(path2)
    htich = cv2.addWeighted(new_img1,1, new_img2, 0.5, 1)
#    htich = cv2.add(new_img1,new_img2)
#    cv2.imshow("merged_img", htich)
    return htich
#    cv2.imshow("merged_img", htich)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()



if __name__ == '__main__':
   data_path = './STED-20200915T020146Z-001/STED'  
   fileList = os.listdir(data_path)
   for path in fileList:
       for subpath in os.listdir(os.path.join(data_path,path)):
           for subsubpath in os.listdir(os.path.join(data_path,path,subpath)):
               for file in os.listdir(os.path.join(data_path,path,subpath,subsubpath)):
                   if 'Channel' in file:
                       print(file)
                       img_gray=tif2gray(os.path.join(data_path,path,subpath,subsubpath,file))
                       img_color=coloring_whole(img_gray,file)
                       new_path=os.path.join('./STED_COLOR/',path,subpath,subsubpath)
                       print('---------------')
                       if not os.path.exists(new_path):
                           print(new_path)
                           os.makedirs(new_path)
                       cv2.imwrite(new_path+'/'+file, img_color)


   data_path = './1'  
   fileList = os.listdir(data_path)
   for path in fileList:
       for subpath in os.listdir(os.path.join(data_path,path)):
           for subsubpath in os.listdir(os.path.join(data_path,path,subpath)):
               for file in os.listdir(os.path.join(data_path,path,subpath,subsubpath)):
                   if 'Ch' in file:
                       print(file)
                       img_gray=tif2gray(os.path.join(data_path,path,subpath,subsubpath,file))
                       img_denoise=denoising(os.path.join(data_path,path,subpath,subsubpath,file))
                       new_path=os.path.join('./STED_DENOISE/1',path,subpath,subsubpath)
                       print('---------------')
                       if not os.path.exists(new_path):
                           print(new_path)
                           os.makedirs(new_path)
                       cv2.imwrite(new_path+'/'+file, img_gray)
    
    data_path = './STED_DENOISE/1'  
    fileList = os.listdir(data_path)
    for path in fileList:
        for subpath in os.listdir(os.path.join(data_path,path)):
            for subsubpath in os.listdir(os.path.join(data_path,path,subpath)):
                for file in os.listdir(os.path.join(data_path,path,subpath,subsubpath)):
                    if 'merge' not in file:
                        if '局部' not in path:
    #                        filename=file.split('_')
                            img_denoise=coloring_whole(os.path.join(data_path,path,subpath,subsubpath,file),file)
                            new_path=os.path.join('./',path,subpath,subsubpath)
                            print(new_path)
                            if not os.path.exists(new_path):
                                os.makedirs(new_path)
                            cv2.imwrite(new_path+'/'+file, img_denoise)
                       else:
   #                        filename=file.split('_')
                           img_denoise=coloring_local(os.path.join(data_path,path,subpath,subsubpath,file),file)
                           new_path=os.path.join('./STED_COLOR/20201202',path,subpath,subsubpath)
                           print(new_path)
                           if not os.path.exists(new_path):
                               os.makedirs(new_path)
                           cv2.imwrite(new_path+'/'+file, img_denoise)   
    
#   merge
   data_path = './STED_COLOR1/'  
   fileList = os.listdir(data_path)
   for path in fileList:
       for subpath in os.listdir(os.path.join(data_path,path)):
           for subsubpath in os.listdir(os.path.join(data_path,path,subpath)):
               file_list=sorted(os.listdir(os.path.join(data_path,path,subpath,subsubpath)))
               for index,file in enumerate(file_list):
                   if index%2==0 and ('merge' not in file):
                       print(index,file)
                       img=merge(os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]),os.path.join(data_path,path,subpath,subsubpath,file_list[index]))
                       new_path=os.path.join('./STED_COLOR1/',path,subpath,subsubpath)
                       cv2.imwrite(new_path+'/merge_'+file, img)
   
    



