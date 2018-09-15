# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:51:06 2018
旋转
@author: zxl
"""

import cv2
import os
import math
import random

import voc_xml
from voc_xml import CreateXML
import utils



#标注框坐标旋转
def rot_xy(rot_cter_x,rot_cter_y,x,y,seta,scale=1.0):
    '''
    Args:
        rot_cter_x,rot_cter_y:旋转中心x,y坐标
        x,y:待旋转点x,y坐标
        seta:旋转角度,顺时针，与opencv图像旋转相反
        scale:放缩尺寸
    return:
        rotx,roty:旋转后的坐标x,y
    '''
    rad_seta = math.radians(-seta)
    rotx = rot_cter_x + (x-rot_cter_x)*scale*math.cos(rad_seta) - (y-rot_cter_y)*scale*math.sin(rad_seta)
    roty = rot_cter_y + (x-rot_cter_x)*scale*math.sin(rad_seta) + (y-rot_cter_y)*scale*math.cos(rad_seta)
    return int(rotx),int(roty)

def rot_box(box,cterxy,imgwh,rot_angle,scale=1.0,correction=True):
    '''
     Args:
         box:边框坐标[xmin,ymin,xmax,ymax]
         cterxy:旋转中心点坐标 [cter_x,cter_y]
         imgwh:图片宽高[w,h]
         rot_angle:旋转角
         scale:放缩尺度
         correction: bool,修正旋转后的目标框为正常左上右下坐标 
    return:
        box:边框坐标[x1,y1,x2,y2,x3,y3,x4,y4]，左上开始，逆时针
    '''
    result_box = []
    xmin,ymin,xmax,ymax = box[0],box[1],box[2],box[3]
    complete_coords = [xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]
    for i in range(int(len(complete_coords)/2)):
        rotx,roty = rot_xy(cterxy[0],cterxy[1],complete_coords[2*i],complete_coords[2*i+1],rot_angle,scale)
        result_box.append(rotx)
        result_box.append(roty)
    if correction:
        xmin = min(result_box[0:len(result_box):2])
        xmax = max(result_box[0:len(result_box):2])
        ymin = min(result_box[1:len(result_box):2])
        ymax = max(result_box[1:len(result_box):2])
        
        xmin_v = utils.confine(xmin,0,imgwh[0]-1)
        ymin_v = utils.confine(ymin,0,imgwh[1]-1)
        xmax_v = utils.confine(xmax,0,imgwh[0]-1)
        ymax_v = utils.confine(ymax,0,imgwh[1]-1)
        #使用阈值剔除边缘截断严重的目标
        if utils.calc_iou([xmin,ymin,xmax,ymax],[xmin_v,ymin_v,xmax_v,ymax_v]) < 0.5:
            xmin_v,ymin_v,xmax_v,ymin_v=0,0,0,0
        return [xmin_v,ymin_v,xmin_v,ymax_v,xmax_v,ymax_v,xmax_v,ymin_v]
    else:
        return complete_coords

def rot_xml(rot_img_name,xml_tree,cterxy,rot_angle,scale=1.0,correction=True):
    '''
    旋转xml文件
    Args:
        xml_tree: 待旋转xml  ET.parse()
        cterxy: 旋转中心坐标[cter_x,cter_y]
        rot_img_name: 旋转后图片保存名字
        rot_angle：旋转角度
        scale:放缩尺度
        correction: bool,修正旋转后的目标框为正常左上右下坐标 
    return:
        createdxml : 创建的xml CreateXML对象
    '''
    root = xml_tree.getroot()
    size = root.find('size')
    imgw,imgh,depth = int(size.find('width').text),int(size.find('height').text),int(size.find('depth').text) 

    createdxml = CreateXML(rot_img_name,imgw,imgh,depth)
    
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)       
        #边框坐标[x1,y1,x2,y2,x3,y3,x4,y4]，左上开始，逆时针
        box=rot_box([xmin,ymin,xmax,ymax],cterxy,[imgw,imgh],rot_angle,scale,correction)
        rxmin,rymin,rxmax,rymax = utils.confine(box[0],0,imgw-1),utils.confine(box[1],0,imgh-1),utils.confine(box[4],0,imgw-1),utils.confine(box[5],0,imgh-1)
        if (rxmin >= rxmax) or (rymin >= rymax):
            continue
        createdxml.add_object_node(obj_name,box[0],box[1],box[4],box[5])
    
    return createdxml    
        
        
#旋转图片，并使用背景图填充四个角
def rot_img_and_padding(img,bk_img,cterxy,rot_angle,scale=1.0):
    '''
    以图片中心为原点旋转
    Args:
        img:待旋转图片
        bk_img:背景填充图片
        cterxy: 旋转中心[x,y]
        rot_angle:旋转角度，逆时针
        scale:放缩尺度
    return:
        imgRotation:旋转后的cv图片
    '''
    img_rows,img_cols = img.shape[:2]
    bk_rows,bk_cols = bk_img.shape[:2]
    
    #背景填充图块选择偏移
    r_offset = bk_rows-int(bk_rows/random.randint(1,5))
    c_offset = bk_cols-int(bk_cols/random.randint(1,5))
    matRotation=cv2.getRotationMatrix2D((cterxy[0],cterxy[1]),rot_angle,scale)
    imgRotation=cv2.warpAffine(img,matRotation,(int(img_cols),int(img_rows)),borderValue=(0,0,0))
    
    rot_img_rows,rot_img_cols = imgRotation.shape[:2]
    for r in range(0,rot_img_rows):
        left_done,right_done = False,False
        for c in range(0,rot_img_cols):
            left_c,right_c = c,rot_img_cols-1-c
            if left_c > right_c:
                break
            if not left_done:
                if not imgRotation[r,left_c].any():
                    bk_r,bk_c = r%(bk_rows-r_offset)+r_offset,left_c%(bk_cols-c_offset)+c_offset
                    imgRotation[r,left_c] = bk_img[bk_r,bk_c]
                else:
                    left_done=True
            if not right_done:
                if not imgRotation[r,right_c].any():
                    bk_r,bk_c = r%(bk_rows-r_offset)+r_offset,right_c%(bk_cols-c_offset)+c_offset
                    imgRotation[r,right_c] = bk_img[bk_r,bk_c]
            if left_done and right_done:
                break  
    return imgRotation
    
 
def generate_rotImg_xml(img,bk_img,xml_tree,cterxy,rot_img_name,rot_angle,scale=1.0,correction=True):
    '''
    旋转图片和对应的xml
    Args:
        img: 待旋转图片路径
        bk_img: 背景图片路径
        xml_tree: img对应的标注文件，ET.parse()
        cterxy:旋转中心[x,y]
        rot_img_name:旋转后图片保存名字
        rot_angle: 旋转角度
        scale: 放缩尺度
        correction: bool,修正旋转后的目标框为正常左上右下坐标
    return:
        imgRotation:旋转后的图片
        xmlRotation:旋转后的xml文件
    '''   
    imgRotation = rot_img_and_padding(img,bk_img,cterxy,rot_angle,scale)
    xmlRotation = rot_xml(rot_img_name,xml_tree,cterxy,rot_angle,scale,correction)
    return imgRotation,xmlRotation
    
def rotImg_xml_centre_from_dirs(imgs_dir,bk_imgs_dir,xmls_dir,rot_img_save_dir,rot_xmls_save_dir,img_suffix,
                                name_suffix,rot_angles,randomAngleRange=[0,360],random_num=1,randomRotation=False,scale=1.0,correction = True):
    '''
    旋转指定路径下的所有图片和xml,以每张图片中心点为旋转中心，并存储到指定路径
    Args:
        imgs_dir,bk_imgs_dir,xmls_dir: 待旋转图片、背景图片、原始xml文件存储路径
        rot_img_save_dir,rot_xmls_save_dir：旋转完成的图片、xml文件存储路径
        img_suffix: 图片可能的后缀名['.jpg','.png','.bmp',..]
        name_suffix：旋转完成的图片、xml的命名后缀标识
        rot_angles: 指定旋转角度[ang1,ang2,ang3,...]
        randomAngleRange: 随机旋转上下限角度[bottom_angle,top_angle]
        random_num: 随机旋转角度个数，randomRotation=True时生效
        randomRotation: 使能随机旋转
        scale: 放缩尺度
        correction: bool,修正旋转后的目标框为正常左上右下坐标       
    '''
    for root,dirs,files in os.walk(xmls_dir):
        for xml_name in files:
            xml_file = os.path.join(xmls_dir,xml_name)
            img_file = None
            for suffix in img_suffix:
                #print(os.path.join(imgs_dir,xml_name.split('.')[0]+suffix))
                if os.path.exists(os.path.join(imgs_dir,xml_name.split('.')[0]+suffix)):
                    img_file = os.path.join(imgs_dir,xml_name.split('.')[0]+suffix)
                    break
            if img_file is None:
                print("there has no image for ",xml_name)
                continue
            img = cv2.imread(img_file)
            imgh,imgw,n_channels = img.shape
            
            rot_num = random_num
            if not randomRotation:
                rot_num = len(rot_angles)
            for i in range(rot_num):
                r_angle = 0
                if randomRotation:
                    r_angle = random.randint(randomAngleRange[0],randomAngleRange[1])
                else:
                    r_angle = rot_angles[i]
                
                bk_img = cv2.imread(os.path.join(bk_imgs_dir,utils.randomChoiceIn(bk_imgs_dir)))
                rot_img_name = xml_name.split('.')[0]+'_'+name_suffix+str(r_angle)+'.'+img_file.split('.')[-1]
                imgRotation,xmlRotation=generate_rotImg_xml(img,bk_img,voc_xml.get_xml_tree(xml_file),[int(imgw/2),int(imgh/2)],rot_img_name,r_angle,scale,correction)
                cv2.imwrite(os.path.join(rot_img_save_dir,rot_img_name),imgRotation)
                xmlRotation.save_xml(rot_xmls_save_dir,rot_img_name.split('.')[0]+'.xml')
                
            
def main():
    imgs_dir ='C:/Users/zxl/Desktop/test/JPEGImages/'
    bk_imgs_dir ='C:/Users/zxl/Desktop/test/back/'
    xmls_dir = 'C:/Users/zxl/Desktop/test/Annotations/'
    
    rot_imgs_save_dir= 'C:/Users/zxl/Desktop/test/rot_imgs/'
    if not os.path.exists(rot_imgs_save_dir):
        os.makedirs(rot_imgs_save_dir)
    rot_xmls_save_dir='C:/Users/zxl/Desktop/test/rot_xmls/'
    if not os.path.exists(rot_xmls_save_dir):
        os.makedirs(rot_xmls_save_dir)
    img_suffix=['.jpg','.png','.bmp']
    name_suffix='rot' #命名标识
    rot_angles=[] #指定旋转角度，当randomRotation=False时有效
    random_num=3 #随机旋转角度个数
    randomRotation=True #使用随机旋转
    
    rotImg_xml_centre_from_dirs(imgs_dir,bk_imgs_dir,xmls_dir,rot_imgs_save_dir,rot_xmls_save_dir,img_suffix,\
                                name_suffix,rot_angles,random_num=random_num,randomRotation=randomRotation,scale=0.8)
    
                
if __name__=='__main__':
    main()            
    

                        
