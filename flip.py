# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 09:29:48 2018
翻转
@author: zxl
"""
import cv2
import os
import random
import voc_xml
from voc_xml import CreateXML

def flip_img(src,flip_type):
    '''翻转图像
    Args:
        src:输入图像
        flip_type:翻转类型，1水平翻转，0垂直翻转，-1水平垂直翻转
    return:
        fliped_img:翻转后的图像
    '''
    fliped_img = cv2.flip(src,flip_type)
    return fliped_img

def flip_xy(x,y,imgw,imgh,flip_type):
    '''翻转坐标点
    Args:
        x,y：坐标点
        imgw,imgh:翻转图像宽高
        flip_type:翻转类型，1水平翻转，0垂直翻转，-1水平垂直翻转
    return:
        fliped_x,fliped_y:翻转后坐标 
    '''
    if 1 == flip_type:
        fliped_x = imgw - x
        fliped_y = y
    elif 0 == flip_type:
        fliped_x = x
        fliped_y = imgh - y
    elif -1 == flip_type:
        fliped_x = imgw - x
        fliped_y = imgh - y
    else:
        print('flip type err')
        return
    return fliped_x,fliped_y
        
def flip_box(box,imgw,imgh,flip_type):
    '''翻转目标框
    Args:
        box:目标框坐标[xmin,ymin,xmax,ymax]
        imgw,imgh:图像宽高
        flip_type:翻转类型,1水平翻转，0垂直翻转，-1水平垂直翻转
    return:
        fliped_box:翻转后的目标框
    '''
    x1,y1 = flip_xy(box[0],box[1],imgw,imgh,flip_type)
    x2,y2 = flip_xy(box[2],box[3],imgw,imgh,flip_type)
    xmin,xmax = min(x1,x2),max(x1,x2)
    ymin,ymax = min(y1,y2),max(y1,y2)
    fliped_box = [xmin,ymin,xmax,ymax]
    return fliped_box

def flip_xml(flip_img_name,xml_tree,flip_type):
    '''翻转xml
    Args:
        flip_img_name:翻转后图片保存名
        xml_tree:待翻转的xml ET.parse()
        flip_type:翻转类型,1水平翻转，0垂直翻转，-1水平垂直翻转
    return:
        createdxml : 创建的xml CreateXML对象   
    '''
    root = xml_tree.getroot()
    size = root.find('size')
    imgw,imgh,depth = int(size.find('width').text),int(size.find('height').text),int(size.find('depth').text)
    createdxml = CreateXML(flip_img_name,int(imgw),int(imgh),depth)
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)
        box = flip_box([xmin,ymin,xmax,ymax],imgw,imgh,flip_type)       
        if (box[0] >= box[2]) or (box[1] >= box[3]):
            continue
        createdxml.add_object_node(obj_name,box[0],box[1],box[2],box[3])
    return createdxml
    

def flip_img_xml(img,xml_tree,flip_img_name,flip_type):
    '''翻转图像和xml目标框
    Args:
        img：源图像
        xml_tree：待crop的xml ET.parse()
        crop_img_name:翻转图片命名
        flip_type:翻转类型
    return:
        fliped_img,fliped_xml : 裁剪完成的图像和xml文件        
    '''  
    fliped_img = flip_img(img,flip_type)
    fliped_xml = flip_xml(flip_img_name,xml_tree,flip_type)
    return fliped_img,fliped_xml

def flip_img_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,img_suffix,name_suffix,\
                          flip_types,random_flip=False):
    '''翻转指定路径下所有图片和xml
    Args:
        imgs_dir,xmls_dir:待翻转图片和xml路径
        imgs_save_dir,xmls_save_dir：图片和xml保存路径
        img_suffix:图片可能的后缀名['.jpg','.png','.bmp',..]
        name_suffix: 处理完成的图片、xml的命名标识
        flip_types: 每张图执行的翻转类型[type1,type2,...],翻转类型共三种,1水平翻转，0垂直翻转，-1水平垂直翻转
        random_flip:是否随机选择翻转类型,与flip_type互斥     
    '''
    for root,dirs,files in os.walk(xmls_dir):
        for xml_name in files:
            xml_file = os.path.join(xmls_dir,xml_name)
            #print(xml_file)
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
            types = flip_types
            if random_flip:
                types = [random.randint(-1,1)]
            for tp in types:
                flip_img_name = xml_name.split('.')[0]+'_'+name_suffix+'_type'+str(tp)+'.'+img_file.split('.')[-1]
                imgflip,xmlflip = flip_img_xml(img,voc_xml.get_xml_tree(xml_file),flip_img_name,tp)
                cv2.imwrite(os.path.join(imgs_save_dir,flip_img_name),imgflip)
                xmlflip.save_xml(xmls_save_dir,flip_img_name.split('.')[0]+'.xml')


def main():
    imgs_dir ='C:/Users/zxl/Desktop/test/JPEGImages/'
    xmls_dir = 'C:/Users/zxl/Desktop/test/Annotations/'
    
    imgs_save_dir= 'C:/Users/zxl/Desktop/test/flip_imgs/'
    if not os.path.exists(imgs_save_dir):
        os.makedirs(imgs_save_dir)
    xmls_save_dir='C:/Users/zxl/Desktop/test/flip_xmls/'
    if not os.path.exists(xmls_save_dir):
        os.makedirs(xmls_save_dir)
    img_suffix=['.jpg','.png','.bmp']
    name_suffix='flip' #命名标识
    flip_types = [1,0,-1] #指定每张图翻转类型 1水平翻转，0垂直翻转，-1水平垂直翻转
    random_flip = False # 随机翻转 与flip_types指定类型互斥

    flip_img_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,img_suffix,name_suffix,flip_types,random_flip)
    
if __name__=='__main__':
    main()  



         