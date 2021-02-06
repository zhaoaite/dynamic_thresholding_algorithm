import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.stats.stats import pearsonr



def getColors(n):
    colors = np.zeros((n, 3))
    colors[:, 0] = np.random.permutation(np.linspace(0, 256, n))
    colors[:, 1] = np.random.permutation(colors[:, 0])
    colors[:, 2] = np.random.permutation(colors[:, 1])
    return colors


def connectivity_clump_detect(path1,path2):
    tublin=cv2.imread(path1)
    tublin=cv2.cvtColor(tublin,cv2.COLOR_BGR2GRAY)
    tublin = cv2.GaussianBlur(tublin, (5, 5), 0)
    thresh_tublin,binary_tublin=cv2.threshold(tublin, 30, 255,cv2.THRESH_BINARY)

    vash2=cv2.imread(path2)
    vash2=cv2.cvtColor(vash2,cv2.COLOR_BGR2GRAY)
    vash2 = cv2.GaussianBlur(vash2, (3, 3), 0)
    thresh_vash2,binary_vash2=cv2.threshold(vash2, 30, 255,cv2.THRESH_BINARY)
   
    
    connectivity=4
    num_labels_tublin, labels_tublin, stats_tublin, centroids_tublin = cv2.connectedComponentsWithStats(binary_tublin, connectivity, cv2.CV_8U)
    num_labels_vash2, labels_vash2, stats_vash2, centroids_vash2 = cv2.connectedComponentsWithStats(binary_vash2, connectivity, cv2.CV_8U)
    
    colors = getColors(num_labels_vash2)
    dst_tublin = np.ones((binary_tublin.shape[0], binary_tublin.shape[1], 3), dtype=np.uint8) * 0
    dst_vash2 = np.ones((binary_vash2.shape[0], binary_vash2.shape[1], 3), dtype=np.uint8) * 0
#    for i in range(num_labels):
#        dst_vash2[labels == i] = colors[i]
    
    
    num_tublin=0
    
    
    
    cross_pixel=0
    num_vash=0
    num_vash_pixel=0
    cross_num=0
    
    for i in range(num_labels_tublin):
        if stats_tublin[i,4]<6000 and stats_tublin[i,4]>5: 
            num_tublin+=1
            dst_tublin[labels_tublin == i] = [255,70,90]
    
    
    vash_list=[]
    for i in range(num_labels_vash2):
#        print(num_labels_vash2)
        if stats_vash2[i,4]>100 and stats_vash2[i,4]<50000: 
            dst_vash2[labels_vash2 == i] =  [255,70,90]
            num_vash+=1
            cv2.rectangle(vash2, (stats_vash2[i,0],stats_vash2[i,1]), (stats_vash2[i,0]+stats_vash2[i,2],stats_vash2[i,1]+stats_vash2[i,3]), (255,100,100), 1)
            vash_list.append(i)
 
    
    
    for i in vash_list:
        cross=0
        temp=0
        for pixel_x in range(stats_vash2[i,1],stats_vash2[i,1]+stats_vash2[i,3]):
            for pixel_y in range(stats_vash2[i,0],stats_vash2[i,0]+stats_vash2[i,2]):
                if dst_vash2[pixel_x,pixel_y].any()!=np.array([0,0,0]).any():
                    num_vash_pixel+=1
                    temp+=1
                if dst_tublin[pixel_x,pixel_y].any()!=np.array([0,0,0]).any() and dst_vash2[pixel_x,pixel_y].any()!=np.array([0,0,0]).any():
                    cross+=1
#                    cv2.rectangle(dst_tublin, (stats_vash2[i,0],stats_vash2[i,1]), (stats_vash2[i,0]+stats_vash2[i,2],stats_vash2[i,1]+stats_vash2[i,3]), (255,255,255), 1)
                    cv2.rectangle(vash2, (stats_vash2[i,0],stats_vash2[i,1]), (stats_vash2[i,0]+stats_vash2[i,2],stats_vash2[i,1]+stats_vash2[i,3]), (255,255,255), 1)
        if cross!=0:
            cross_pixel+=temp
            cross_num+=1
            


            
    new_path=os.path.join('./clumps/',path,subpath,subsubpath)
    print(new_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    cv2.imwrite(new_path+'/'+file, vash2)      
    
#    cv2.imwrite('1.jpg',dst_vash2)  
#    cv2.imwrite('2.jpg',dst_tublin) 
    
    
    
    vash2_in = cross_num/num_vash 
    vash2_out = (num_vash-cross_num)/num_vash 
    
    
    a=open('a.txt', 'a')
#    a.write("--------------------\n")
    a.write('path:'+str(path1)+'\n')
    a.write('tubulin:'+str(num_tublin)+'\n')
    a.write('clump:'+str(num_vash)+'\n')
    a.write('clump in tubulin:'+str(cross_num)+'\n')
    a.write('clump out of tubulin:'+str(num_vash-cross_num)+'\n')
    a.write("--------------------\n")
    a.write("clump_in/clump_total:"+str(vash2_in)+'\n')
    a.write("clump_out/clump_total:"+str(vash2_out)+'\n')
    a.close()
    
    if 'Sample 1' in path1 or 'Sample 2' in path1:
        WT.append([vash2_in,vash2_out])
    else:
        KO.append([vash2_in,vash2_out])
        
        
    return vash2_in,vash2_out



def PCC(path1,path2):
    tublin=cv2.imread(path1)
    tublin=cv2.cvtColor(tublin,cv2.COLOR_BGR2GRAY)
#    500*500 local
    tublin=cv2.resize(tublin, (1024,1024), interpolation = cv2.INTER_AREA)
    vash2=cv2.imread(path2)
    vash2=cv2.cvtColor(vash2,cv2.COLOR_BGR2GRAY)
    vash2=cv2.resize(vash2, (1024,1024), interpolation = cv2.INTER_AREA)
    
    tublin=np.asarray(tublin)
    vash2=np.asarray(vash2)
    print(vash2.shape)
    
    co=pearsonr(vash2.reshape(1024*1024), tublin.reshape(1024*1024))
    a=open('20201202_cell4_pixel.txt', 'a')
    a.write("--------------------\n")
    a.write('Pearson:'+str(co[0])+'\n')
    a.close()
    return co


def vash_num_detect(path1,path2):
    tublin=cv2.imread(path1)
    tublin=cv2.cvtColor(tublin,cv2.COLOR_BGR2GRAY)
    thresh_tublin,binary_tublin=cv2.threshold(tublin, 30, 255,cv2.THRESH_BINARY)
    vash2=cv2.imread(path2)
    vash2=cv2.cvtColor(vash2,cv2.COLOR_BGR2GRAY)
    thresh_vash2,binary_vash2=cv2.threshold(vash2, 30, 255,cv2.THRESH_BINARY)
    connectivity=4
    num_labels_tublin, labels_tublin, stats_tublin, centroids_tublin = cv2.connectedComponentsWithStats(binary_tublin, connectivity, cv2.CV_8U)
    num_labels_vash2, labels_vash2, stats_vash2, centroids_vash2 = cv2.connectedComponentsWithStats(binary_vash2, connectivity, cv2.CV_8U)
    
    colors = getColors(num_labels_vash2)
    dst_tublin = np.ones((binary_tublin.shape[0], binary_tublin.shape[1], 3), dtype=np.uint8) * 0
    dst_vash2 = np.ones((binary_vash2.shape[0], binary_vash2.shape[1], 3), dtype=np.uint8) * 0
#    for i in range(num_labels):
#        dst_vash2[labels == i] = colors[i]
    num_tublin=0
    
    
    cross_pixel=0
    num_vash=0
    num_vash_pixel=0
    cross_num=0
    
    
    
    for i in range(num_labels_tublin):
        if stats_tublin[i,4]<5000 and stats_tublin[i,4]>10: 
            num_tublin+=1
            dst_tublin[labels_tublin == i] = [255,70,90]
    
    
    vash_list=[]
    for i in range(num_labels_vash2):
        if stats_vash2[i,4]<100 and stats_vash2[i,4]>3: 
            dst_vash2[labels_vash2 == i] =  [255,70,90]
            num_vash+=1
            vash_list.append(i)
 
    
    
    for i in vash_list:
        cross=0
        temp=0
        for pixel_x in range(stats_vash2[i,1],stats_vash2[i,1]+stats_vash2[i,3]):
            for pixel_y in range(stats_vash2[i,0],stats_vash2[i,0]+stats_vash2[i,2]):
                if dst_vash2[pixel_x,pixel_y].any()!=np.array([0,0,0]).any():
                    num_vash_pixel+=1
                    temp+=1
                if dst_tublin[pixel_x,pixel_y].any()!=np.array([0,0,0]).any() and dst_vash2[pixel_x,pixel_y].any()!=np.array([0,0,0]).any():
                    cross+=1
                    cv2.rectangle(dst_tublin, (stats_vash2[i,0],stats_vash2[i,1]), (stats_vash2[i,0]+stats_vash2[i,2],stats_vash2[i,1]+stats_vash2[i,3]), (255,255,255), 1)
                    cv2.rectangle(dst_vash2, (stats_vash2[i,0],stats_vash2[i,1]), (stats_vash2[i,0]+stats_vash2[i,2],stats_vash2[i,1]+stats_vash2[i,3]), (255,255,255), 1)
        if cross!=0:
            cross_pixel+=temp
            cross_num+=1
                
                
                
    cv2.imwrite('1.jpg',dst_vash2)  
    cv2.imwrite('2.jpg',dst_tublin)      
#            x,y=centroids_vash2[i,:]
#            if dst_tublin[int(round(x)),int(round(y))].any()!=np.array([0,0,0]).any():
#                 cross_pixel+=1
#                 dst_vash2[labels_vash2 == i] = colors[i]
                 
    vash2_in=cross_pixel/num_vash_pixel 
    vash2_out= (num_vash_pixel-cross_pixel)/num_vash_pixel
#    vash2_in = cross_num/num_vash 
#    vash2_out = (num_vash-cross_num)/num_vash 
#    
    
    a=open('a.txt', 'a')
#    a.write("--------------------\n")
    a.write('path:'+str(path1)+'\n')
    a.write('tubulin:'+str(num_tublin)+'\n')
    a.write('vash2:'+str(num_vash)+'\n')
    a.write('vash2 in tubulin:'+str(cross_num)+'\n')
    a.write('vash2 out of tubulin:'+str(num_vash-cross_num)+'\n')
    a.write("--------------------\n")
    a.write("vash2_in/vash2_total:"+str(vash2_in)+'\n')
    a.write("vash2_out/vash2_total:"+str(vash2_out)+'\n')
    a.close()
    
    if 'Sample 1' in path1 or 'Sample 2' in path1:
        WT.append([vash2_in,vash2_out])
    else:
        KO.append([vash2_in,vash2_out])
        
           
    return vash2_in,vash2_out

          

    
    
    
def pixel_detect(path1,path2):
    tubulin=cv2.imread(path1)
    vash2=cv2.imread(path2)
#    vash2 = cv2.GaussianBlur(vash2, (3,3), 0)
#    tubulin = cv2.GaussianBlur(tubulin, (5, 5), 0)
   
    tubulin_hsv=cv2.cvtColor(tubulin,cv2.COLOR_BGR2HSV)
    vash2__hsv=cv2.cvtColor(vash2,cv2.COLOR_BGR2HSV)
    h,w,_=vash2__hsv.shape
    z=range(0,h)
    d=range(0,w)
    num_yellow_pixel=0
    num_black_pixel=0
    num_red_pixel=0
    cross_pixel=0
    pixels=0
    for x in z:
        for y in d:
            pixels+=1
            if tubulin_hsv[x,y].any()!=np.array([0,0,0]).any():
                num_yellow_pixel+=1
            if vash2__hsv[x,y].any()!=np.array([0,0,0]).any():
                num_red_pixel+=1
                if x<h-1 and y<w-1 and x>0 and y>0:
                    if tubulin_hsv[x,y].any()!=np.array([0,0,0]).any():
#                    (tubulin_hsv[x+1,y].any()!=np.array([0,0,0]).any() and vash2__hsv[x+1,y].any()!=np.array([0,0,0]).any()) or \
#                    (tubulin_hsv[x,y+1].any()!=np.array([0,0,0]).any() and vash2__hsv[x,y+1].any()!=np.array([0,0,0]).any()) or \
#                    (tubulin_hsv[x-1,y].any()!=np.array([0,0,0]).any() and vash2__hsv[x-1,y].any()!=np.array([0,0,0]).any()) or \
#                    (tubulin_hsv[x,y-1].any()!=np.array([0,0,0]).any() and vash2__hsv[x,y-1].any()!=np.array([0,0,0]).any()):
                        cross_pixel+=1
            
#            if tubulin_hsv[x,y].any()!=np.array([0,0,0]).any() and vash2__hsv[x,y].any()!=np.array([0,0,0]).any():
#                cross_pixel+=1
#            if tubulin_hsv[x,y].any()==np.array([0,0,0]).any() and vash2__hsv[x,y].any()!=np.array([0,0,0]).any():
#                out_tubulin_pixel+=1
            if tubulin_hsv[x,y].any()==np.array([0,0,0]).any() and vash2__hsv[x,y].any()==np.array([0,0,0]).any():
                num_black_pixel+=1


#    if 'Sample 4' in path1:
#        cross_pixel=cross_pixel+int(num_red_pixel*0.07)
#    else:
#        cross_pixel=cross_pixel-int(num_red_pixel*0.07)
        
                
                
                
#    vash2_in=cross_pixel/num_yellow_pixel  
    red_vash2_in=cross_pixel/num_red_pixel 
    red_vash2_out= (num_red_pixel-cross_pixel)/num_red_pixel
    
    a=open('20201202_cell4_pixel.txt', 'a')
#    a.write("--------------------\n")
    a.write('path:'+str(path1)+'\n')
#    a.write('cross_pixel:'+str(cross_pixel)+'\n')
#    a.write('tubulin:'+str(num_yellow_pixel)+'\n')
#    a.write('vash2:'+str(num_red_pixel)+'\n')
    a.write('map4 overlaping tubulin:'+str(cross_pixel)+'\n')
    a.write('map4 out of tubulin:'+str(num_red_pixel-cross_pixel)+'\n')
#    a.write("black:"+str(num_black_pixel)+'\n')
#    a.write("cross_pixel/tubulin:"+str(vash2_in)+'\n')
    a.write("--------------------\n")
    a.write("map4_overlap/vash_total:"+str(red_vash2_in)+'\n')
    a.write("map4_out/vash_total:"+str(red_vash2_out)+'\n')
    a.close()



    
    
    if 'Sample 5' in path1:
#    if 'Sample 1' in path1 or 'Sample 2' in path1:
        WT.append([red_vash2_in,red_vash2_out])
    else:
        KO.append([red_vash2_in,red_vash2_out])
    
    
        
#    print('cross_pixel',cross_pixel) 
#    print('tubulin',num_yellow_pixel)      
#    print("cross_vash2/tubulin:",vash2_in) 
#    print("outoftubl_vash2/black:",vash2_out)  
    return red_vash2_in,red_vash2_out
    
#    #if HSV[2,3]==[178 ,255 ,204]:
#    #    print("红色")
#    cv2.imshow("ex_HSV",ex_HSV)
#    cv2.imshow("HSV",HSV)
#    cv2.imshow('image',image)#显示img
#    #cv2.setMouseCallback("imageHSV",getpos)#
#    cv2.waitKey(0)
#    
    
    
if __name__ == '__main__':
    data_path = './STED_COLOR/20201202/'  
    WT=[]
    KO=[]
    fileList = os.listdir(data_path)
    record_pixel_rate=[]
    for path in fileList:
        paths_list=sorted(os.listdir(os.path.join(data_path,path)))
        for subpath in paths_list:
            subpaths=sorted(os.listdir(os.path.join(data_path,path,subpath)),key= lambda x:int(x[5:]))
            print(subpaths)
            for subsubpath in subpaths:
                file_list=sorted(os.listdir(os.path.join(data_path,path,subpath,subsubpath)))
                for index,file in enumerate(file_list):
                    if index%2==0 and ('merge' not in file):
                        print(os.path.join(data_path,path,subpath,subsubpath,file))
#                        connect->pixel_detect
#                        dst_tublin,dst_vash2 = connectivity_clump_detect(os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]),os.path.join(data_path,path,subpath,subsubpath,file_list[index]))

#                        dst_tublin,dst_vash2 = connectivity_detect(os.path.join(data_path,path,subpath,subsubpath,file_list[index]),os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]))
                        PCC(os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]),os.path.join(data_path,path,subpath,subsubpath,file_list[index]))
                        vash2_in,vash2_out = pixel_detect(os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]),os.path.join(data_path,path,subpath,subsubpath,file_list[index]))
#                        vash2_in,vash2_out = pixel_detect(os.path.join(data_path,path,subpath,subsubpath,file_list[index]),os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]))
                        
#                        vash2_in,vash2_out = pixel_detect(os.path.join(data_path,path,subpath,subsubpath,file_list[index]),os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]),tubulin,vash2)
##                        vash_num_calculate
#                        vash2_in,vash2_out = vash_num_detect(os.path.join(data_path,path,subpath,subsubpath,file_list[index]),os.path.join(data_path,path,subpath,subsubpath,file_list[index+1]))
                        
                        
                        
                        
                        
    np.savetxt('1213_s5_local_pixel.txt',WT)
    np.savetxt('1213_s6_local_pixel.txt',KO)
   
    WT=np.loadtxt('1213_s5_local_pixel.txt')
    KO=np.loadtxt('1213_s6_local_pixel.txt')
    
    WT=np.array(WT)
    KO=np.array(KO)
    
    
    
    plt.plot(range(0,len(WT),1),WT[:,0],'o',label = 's5 in')
    plt.plot(50,np.mean(WT[:,0]),'o',label = 's5 in (mean)')
    print(np.mean(WT[:,0]))
    plt.text(50, np.mean(WT[:,0])+0.02, round(np.mean(WT[:,0]),2), ha='center', va='bottom', fontsize=10)
    
    plt.plot(range(100,100+len(KO),1),KO[:,0],'p',label = 's6 in')
    plt.plot(150,np.mean(KO[:,0]),'p',label = 's6 in (mean)')
    print(np.mean(KO[:,0]))
    plt.text(150, np.mean(KO[:,0])+0.02, round(np.mean(KO[:,0]),2), ha='center', va='bottom', fontsize=10)
    
    plt.plot(range(200,200+len(WT),1),WT[:,1],'>',label = 's5 out')
    plt.plot(250,np.mean(WT[:,1]),'>',label = 's5 out (mean)')
    print(np.mean(WT[:,1]))
    plt.text(250, np.mean(WT[:,1])+0.02, round(np.mean(WT[:,1]),2), ha='center', va='bottom', fontsize=10)
    
    
    plt.plot(range(300,300+len(KO),1),KO[:,1],'*',label = 's6 out')
    plt.plot(350,np.mean(KO[:,1]),'p',label = 's6 out (mean)')
    print(np.mean(KO[:,1]))
    plt.text(350, np.mean(KO[:,1])+0.02, round(np.mean(KO[:,1]),2), ha='center', va='bottom', fontsize=10)
    
    plt.legend(loc='upper left',ncol=2)
    plt.show()
    
    
#    wt=np.loadtxt('wt.txt')
#    ko=np.loadtxt('ko.txt')
#    plt.plot(range(0,len(wt),1),wt[:,0],'o')
#    plt.plot(range(500,500+len(ko),1),ko[:,0],'p')
#    plt.plot(range(1000,1000+len(wt),1),wt[:,1],'>')
#    plt.plot(range(1500,1500+len(ko),1),ko[:,1],'*')
#    plt.show()