# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:33:03 2018
缩放
@author: zxl
"""

import cv2
import voc_xml
from voc_xml import CreateXML
import os


def resize_xy(x,y,fx,fy):
    '''
    放缩点坐标
    Args:
        x,y:待放缩点坐标
        fx,fy:放缩比例
    return:
        x,y:放缩后坐标点
    '''
    return int(x*fx),int(y*fy)
    
def resize_box(box,fx,fy):
    '''
    放缩目标框：
    Args:
        box: 目标框 [xmin,ymin,xmax,ymax]
        fx,fy: x,y坐标轴放缩比例
    return:
        rsize_box: 放缩后的坐标框 [xmin,ymin,xmax,ymax]
    '''
    xmin,ymin = resize_xy(box[0],box[1],fx,fy)
    xmax,ymax = resize_xy(box[2],box[3],fx,fy)
    return [xmin,ymin,xmax,ymax]


def resize_img(src,dsize=(0,0),fx=1.0,fy=1.0):
    '''
    放缩图片
    Args:
        src:源图片
        dsize:指定放缩大小(w,h)
        fx,fy:比例放缩
    return：
        sized_img:放缩后的图像
    '''
    sized_img = cv2.resize(src,dsize,fx=fx,fy=fy)
    return sized_img




def resize_xml(resized_img_name,xml_tree,dsize=(0,0),fx=1.0,fy=1.0):
    '''
    xml目标框放缩变换
    Args:
        resized_img_name: resize图片保存名
        xml_tree:  待resize xml  ET.parse()
        dsize:指定放缩大小(w,h)
        fx,fy:比例放缩
    return:
        createdxml : 创建的xml CreateXML对象        
    '''
    root = xml_tree.getroot()
    size = root.find('size')
    imgw,imgh,depth = int(size.find('width').text),int(size.find('height').text),int(size.find('depth').text)
    resize_imgw,resize_imgh = imgw,imgh
    if dsize == (0,0):
        resize_imgw = int(imgw*fx)
        resize_imgh = int(imgh*fy)
    else:
        resize_imgw,resize_imgh = dsize
    rsize_fx,resize_fy = resize_imgw/imgw,resize_imgh/imgh
                           
    createdxml = CreateXML(resized_img_name,resize_imgw,resize_imgh,depth)
    
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)
        box = resize_box([xmin,ymin,xmax,ymax],rsize_fx,resize_fy)
        if (box[0] >= box[2]) or (box[1] >= box[3]):
            continue
        createdxml.add_object_node(obj_name,box[0],box[1],box[2],box[3])
    return createdxml

def generate_resizeImg_xml(img,xml_tree,resized_img_name,dsize=(0,0),fx=1.0,fy=1.0):
    '''
    生成旋转后的图片和xml文件
    Args:
        img:源图片
        xml_tree：待resizexml  ET.parse()
        resized_img_name: resize图片保存名
        dsize:指定放缩大小(w,h)
        fx,fy:比例放缩
    return:
        resized_img,resized_xml       
    '''
    resized_img = resize_img(img,dsize,fx,fy)
    resized_xml = resize_xml(resized_img_name,xml_tree,dsize,fx,fy)
    return resized_img,resized_xml
    

def resizeImg_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,img_suffix,name_suffix,dsize=(0,0),fx=1.0,fy=1.0):
    '''
    放缩指定路径下的图片和xml
    Args:
        imgs_dir,xmls_dir: 待放缩图片、原始xml文件存储路径
        imgs_save_dir，xmls_save_dir: 处理完成的图片、xml文件存储路径
        img_suffix: 图片可能的后缀名['.jpg','.png','.bmp',..]
        name_suffix: 处理完成的图片、xml的命名标识
        dsize: 指定放缩大小(w,h)
        fx,fy: 比例放缩       
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
            
            imgh,imgw,n_channels = img.shape
            resize_imgw,resize_imgh = imgw,imgh
            if dsize == (0,0):
                resize_imgw = imgw*fx
                resize_imgh = imgh*fy
            else:
                resize_imgw,resize_imgh = dsize
                       
            resized_img_name = xml_name.split('.')[0]+'_'+name_suffix+str(resize_imgw)+'x'+str(resize_imgh)+'.'+img_file.split('.')[-1]
            imgResize,xmlResize = generate_resizeImg_xml(img,voc_xml.get_xml_tree(xml_file),resized_img_name,dsize,fx,fy)
            cv2.imwrite(os.path.join(imgs_save_dir,resized_img_name),imgResize)
            xmlResize.save_xml(xmls_save_dir,resized_img_name.split('.')[0]+'.xml')

def main():
    imgs_dir ='C:/Users/zxl/Desktop/test/JPEGImages/'
    xmls_dir = 'C:/Users/zxl/Desktop/test/Annotations/'
    
    imgs_save_dir= 'C:/Users/zxl/Desktop/test/resize_imgs/'
    if not os.path.exists(imgs_save_dir):
        os.makedirs(imgs_save_dir)
    xmls_save_dir='C:/Users/zxl/Desktop/test/resize_xmls/'
    if not os.path.exists(xmls_save_dir):
        os.makedirs(xmls_save_dir)
    img_suffix=['.jpg','.png','.bmp']
    name_suffix='rsize' #命名标识
    dsize=(400,200)  #指定放缩大小(w,h)
    fx=1.0
    fy=1.0 #放缩比例
    
    resizeImg_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,img_suffix,name_suffix,dsize,fx,fy)
    
                
if __name__=='__main__':
    main()  














