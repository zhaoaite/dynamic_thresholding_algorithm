import cv2
import numpy as np
import os

def remove_connectivity(path1,path2):
    tubulin=cv2.imread(path1,0)
    vash2=cv2.imread(path2,0)
    img_copy=tubulin.copy()
    ret, binary = cv2.threshold(img_copy, 75, 255, cv2.THRESH_BINARY) 
    print(binary.shape)
    h,w=binary.shape
    z=range(0,h)
    d=range(0,w)
    kernel = np.ones((21,21),np.uint8) 
    kernel_1 = np.ones((89,89),np.uint8) 

    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_1)

    opening_x = opening.shape[0]
    opening_y = opening.shape[1]
    opening[:,0] = 255
    opening[:,opening_y-1] = 255
    opening[0,:] = 255
    opening[opening_x-1,:] = 255
    
    for x in z:
        for y in d:
            if opening[x,y].any()!=np.array([0,0,0]).any():
                tubulin[x,y]=0
                vash2[x,y]=0
    

    return tubulin,vash2




if __name__ == '__main__':
    data_path = './STED_COLOR1/'  
    fileList = os.listdir(data_path)
    record_pixel_rate=[]
    for path in fileList:
        for subpath in os.listdir(os.path.join(data_path,path)):
            for subsubpath in os.listdir(os.path.join(data_path,path,subpath)):
                file_list=sorted(os.listdir(os.path.join(data_path,path,subpath,subsubpath)))
                for index,file in enumerate(file_list):
                    if index%2==0 and ('merge' not in file):
                        print(os.path.join(data_path,path,subpath,subsubpath,file))
                        tubulin,vash2 = remove_connectivity(os.path.join(data_path,path,subpath,subsubpath,file_list[index]),os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]))
                        new_path=os.path.join('./STED_COLOR2/',path,subpath,subsubpath)
                        if not os.path.exists(new_path):
                            os.makedirs(new_path)
                        cv2.imwrite(new_path+'/'+file_list[index], tubulin)   
                        cv2.imwrite(new_path+'/'+file_list[index+1], vash2)
