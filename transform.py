# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:21:18 2018
随机组合变换
@author: zxl
"""
import cv2
import os
import numpy as np

import utils
import mosaic
import voc_xml
from voc_xml import CreateXML

def transform_img_xml(src_imgpath,src_xmlpath,transforms,img_save_name):
    '''按transforms中的转换操作变换img和xml
    Args:
        src_imgpath: 待变换的图片路径
        src_xmlpath: xml标注文件路径
        transforms：转换操作
        img_save_name: 图片保存名
    return:
        transformed_img:转换完成的图片
        createdxml:转换生成的新标签
    '''
    src_img = cv2.imread(src_imgpath)
    src_xml = voc_xml.get_xml_tree(src_xmlpath)
    
    transformed_img,certain_transforms = mosaic.transform_img(src_img,transforms)
    
    imgh,imgw,n_channels = transformed_img.shape
    createdxml = CreateXML(img_save_name,imgw,imgh,n_channels)
    createdxml = mosaic.transform_xml(src_xml,createdxml,certain_transforms,0,0)
    return transformed_img,createdxml


def transform_onefile(src_imgpath,src_xmlpath,imgs_save_dir,xmls_save_dir,transforms,N=1):
    '''对一张图进行转换，并生成转换后的图片和xml文件
    Args:
        src_imgpath: 待变换的图片路径
        src_xmlpath: xml标注文件路径
        imgs_save_dir：图片文件保存目录
        xmls_save_dir：xml文件保存目录   
        transforms：转换操作
        N:每张原图生成N张转换图
    '''
    for n in range(1,N+1):
        imgname = os.path.basename(src_imgpath).split('.')[0]
        new_imgname = imgname+'_trans'+str(n).zfill(3)
        img_save_name = new_imgname+'.jpg'
        transformed_img,createdxml=transform_img_xml(src_imgpath,src_xmlpath,transforms,img_save_name)
        cv2.imwrite(os.path.join(imgs_save_dir,img_save_name),transformed_img)
        createdxml.save_xml(xmls_save_dir,img_save_name.split('.')[0]+'.xml')

def transform_file_from_dirs(imgs_xmls_dirs,imgs_save_dir,xmls_save_dir,transforms,N=1):
    '''对文件夹中所有图片进行转换，并生成转换后的图片和xml文件
    Args:
        imgs_xmls_dirs:待转换的图片、xml、背景图片目录
        imgs_save_dir：图片文件保存目录
        xmls_save_dir：xml文件保存目录 
        transforms：转换操作
        N:每张原图生成N张转换图    
    '''
    for i in range(len(imgs_xmls_dirs)):
        imgs_dir = imgs_xmls_dirs[i]['imgs_dir']
        xmls_dir = imgs_xmls_dirs[i]['xmls_dir']
        bk_imgs_dir = imgs_xmls_dirs[i]['bk_imgs_dir']
        for trans in transforms:
            if trans['opt'] == 'rotate':
                trans['bk_imgs_dir'] = bk_imgs_dir
    
        fileCount = utils.fileCountIn(imgs_dir)
        count = 0
        for root,dirs,files in os.walk(imgs_dir):
            for imgname in files:
                src_imgpath = os.path.join(imgs_dir,imgname)
                src_xmlpath = os.path.join(xmls_dir,imgname.split('.')[0]+'.xml')
                count += 1
                if count%10 == 0:
                    print('[%d | %d]%d%%'%(fileCount,count,count*100/fileCount))
                if not os.path.exists(src_xmlpath):
                    print(src_xmlpath,' not exist!')
                    continue
                transform_onefile(src_imgpath,src_xmlpath,imgs_save_dir,xmls_save_dir,transforms,N)
            
def main():
    imgs_xmls_dirs = {0:{'imgs_dir':'C:/Users/zxl/Desktop/test/JPEGImages/',\
                    'bk_imgs_dir':'C:/Users/zxl/Desktop/test/back/',\
                    'xmls_dir':'C:/Users/zxl/Desktop/test/Annotations/'},
                        
                     }
    
    imgs_save_dir = 'C:/Users/zxl/Desktop/test/trans_imgs/'
    if not os.path.exists(imgs_save_dir):
        os.makedirs(imgs_save_dir)
    xmls_save_dir ='C:/Users/zxl/Desktop/test/trans_xmls/'
    if not os.path.exists(xmls_save_dir):
        os.makedirs(xmls_save_dir)
        
    N = 5
    
    transforms = [{'opt':'resize','fx':0.5,'fy':0.5,'dsize':(1024,1024),'imgwh':[]},  
                      {'opt':'rotate','cterxy':[],'imgwh':[],'rot_angle':0,'randomRotation':True,\
                        'randomAngleRange':[0,360],'scale':0.3,'correction':True,'bk_imgs_dir':''},\
                       {'opt':'flip','flip_type':-1,'random_flip':True,'imgwh':[]},
                        {'opt':'crop','crop_type':'RANDOM_CROP','dsize':(500,500),'top_left_x':0,'top_left_y':0,\
                         'crop_w':0,'crop_h':0,'fw':0.6,'fh':0.6,'random_wh':False,'iou_thr':0.5,'imgwh':[]}] 
    
    transform_file_from_dirs(imgs_xmls_dirs,imgs_save_dir,xmls_save_dir,transforms,N)
    
if __name__=='__main__':
    main()    