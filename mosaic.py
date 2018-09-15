# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:04:52 2018
拼接图片
@author: zxl
"""
import cv2
import os
import random
import copy
import numpy as np

import rotate
import resize
import flip
import crop
import voc_xml
import utils
from voc_xml import CreateXML

def mosaic_img(img,part_img,start_row,start_col):
    '''嵌入子图
    Args:
        img:大图
        part_img:待嵌入子图
        start_row,start_col:子图嵌入起始行列
    return:
        img:嵌入结果图
    '''
    rows,cols,n_channel=part_img.shape
    img[start_row:start_row+rows,start_col:start_col+cols] = part_img
    return img


def translational_box(box,start_row,start_col):
    '''平移box坐标
    Args:
        box:边框坐标[xmin,ymin,xmax,ymax]
        start_row,start_col:子图嵌入起始行列
    return:
        trans_box:平移后边框坐标
    '''
    trans_box = [box[0]+start_col,box[1]+start_row,box[2]+start_col,box[3]+start_row]
    return trans_box


def transform_box(box,transforms):
    '''目标框坐标转换
    Args:
        box:目标框[xmin,ymin,xmax,ymax]
        transforms:转换操作[{'opt':'rotate','cterxy':[],'imgwh':[],'rot_angle':0,'randomRotation':False,\
                            'randomAngleRange':[0,360],'scale':1.0,'correction':True,'bk_imgs_dir':'xxx'},
                               {'opt':'crop','crop_type':RANDOM_CROP,'dsize':(0,0),'top_left_x':0,'top_left_y':0,'fw':0.5,'fh':0.7,'random_wh':False ,'iou_thr':0.5},
                               {'opt':'flip','flip_type':-1,'random_flip':True,'imgwh':[]},
                               {'opt':'resize','fx':0.5,'fy':0.5,'dsize':(0,0),'imgwh':[]}]
    return:
        transformed_box:转换后目标框坐标[xmin,ymin,xmax,ymax]        
    '''
    transformed_box = box
    for operate in transforms:
        if [0,0,0,0]==transformed_box:
            break
        if transformed_box[2]>operate['imgwh'][0] or transformed_box[3]>operate['imgwh'][1]:
            print(operate['opt'])
            print(operate['imgwh'])
            print(transformed_box)
        
        if 'resize' == operate['opt']:
            transformed_box = resize.resize_box(transformed_box,operate['fx'],operate['fy'])
        elif 'rotate' == operate['opt']:
            #box,cterxy,imgwh,rot_angle,scale=1.0,correction=True
            tmp_box = rotate.rot_box(transformed_box,operate['cterxy'],operate['imgwh'],operate['rot_angle'],operate['scale'],operate['correction'])
            imgw,imgh = operate['imgwh'][0],operate['imgwh'][1]
            transformed_box = [utils.confine(tmp_box[0],0,imgw-1),utils.confine(tmp_box[1],0,imgh-1),utils.confine(tmp_box[4],0,imgw-1),utils.confine(tmp_box[5],0,imgh-1)]
        elif 'crop' == operate['opt']:
            transformed_box=crop.crop_box(transformed_box,operate['top_left_x'],operate['top_left_y'],operate['crop_w'],operate['crop_h'],operate['iou_thr'])
        elif 'flip' == operate['opt']:
            transformed_box = flip.flip_box(transformed_box,operate['imgwh'][0],operate['imgwh'][1],operate['flip_type'])
    return transformed_box 


def transform_xml(part_xml_tree,createdxml,transforms,start_row,start_col):
    '''将子图的标注框添加到总图的xml中
    Args:
        part_xml_tree:子图xml ET.parse()
        createdxml:总图创建的xml CreateXML对象
        transforms:转换操作
        start_row,start_col:子图嵌入起始行列
    return:
        createdxml: 总图创建的xml CreateXML对象   
    '''
    root = part_xml_tree.getroot()
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)
        box = transform_box([xmin,ymin,xmax,ymax],transforms)        
        if (box[0] >= box[2]) or (box[1] >= box[3]):
            continue
        box = translational_box(box,start_row,start_col)
        createdxml.add_object_node(obj_name,box[0],box[1],box[2],box[3])            
    return createdxml

        
def transform_img(src_img,transforms):
    '''图像变换
    Args:
        src_img:源图片
        transforms:转换操作[{'opt':'rotate','cterxy':[],'imgwh':[],'rot_angle':0,'randomRotation':False,\
                            'randomAngleRange':[0,360],'scale':1.0,'correction':True,'bk_imgs_dir':'xxx'},
                               {'opt':'crop','crop_type':RANDOM_CROP,'dsize':(0,0),'top_left_x':0,'top_left_y':0,'fw':0.5,'fh':0.7,'random_wh':False ,'iou_thr':0.5},
                               {'opt':'flip','flip_type':-1,'random_flip':True,'imgwh':[]},
                               {'opt':'resize','fx':0.5,'fy':0.5,'dsize':(0,0),'imgwh':[]}]
    return：
        transformed_img:变换后的图片
        certain_transforms:实际变换操作参数
    '''
    certain_transforms = copy.deepcopy(transforms)   
    imgh,imgw,depth = src_img.shape
    imgwh = [imgw,imgh]
    transformed_img = src_img
    for operate in certain_transforms:
        operate['imgwh'] = imgwh #每一种操作的输入图片宽高
        if 'rotate' == operate['opt']:
            bk_img = cv2.imread(os.path.join(operate['bk_imgs_dir'],utils.randomChoiceIn(operate['bk_imgs_dir'])))
            cterxy = [int(imgw/2),int(imgh/2)]
            rot_angle = operate['rot_angle']
            if operate['randomRotation']:
                rot_angle=random.randint(operate['randomAngleRange'][0],operate['randomAngleRange'][1])               
            transformed_img=rotate.rot_img_and_padding(transformed_img,bk_img,cterxy,rot_angle,operate['scale'])
            operate['cterxy'] = cterxy
            operate['rot_angle'] = rot_angle
            
        elif 'resize' == operate['opt']:
            resize_imgw,resize_imgh = imgwh[0],imgwh[1]
            if (0,0)==operate['dsize']:
                resize_imgw = imgw*operate['fx']
                resize_imgh = imgh*operate['fy']
            else:
                resize_imgw,resize_imgh = operate['dsize']
            transformed_img = resize.resize_img(transformed_img,operate['dsize'],operate['fx'],operate['fy'])
            imgwh = [resize_imgw,resize_imgh]
            operate['fx'] = resize_imgw/operate['imgwh'][0]
            operate['fy'] = resize_imgh/operate['imgwh'][1]
        elif 'crop' == operate['opt']:
            crop_imgw,crop_imgh = operate['dsize']
            if (0,0)==operate['dsize'] and not operate['random_wh']:
                crop_imgw = int(operate['imgwh'][0]*operate['fw'])
                crop_imgh = int(operate['imgwh'][1]*operate['fh'])
            elif operate['random_wh']:
                crop_imgw = int(operate['imgwh'][0]*(operate['fw']+ random.random()*(1-operate['fw'])))
                crop_imgh = int(operate['imgwh'][1]*(operate['fh'] + random.random()*(1-operate['fh'])))           

            if 'CENTER_CROP' == operate['crop_type']:
                top_left_x,top_left_y = int(operate['imgwh'][0]/2-crop_imgw/2),int(operate['imgwh'][1]/2-crop_imgh/2)
            elif 'RANDOM_CROP' == operate['crop_type']:
                top_left_x,top_left_y =  random.randint(0,operate['imgwh'][0]-crop_imgw-1),random.randint(0,operate['imgwh'][1]-crop_imgh-1)
            else:
                top_left_x,top_left_y = operate['top_left_x'],operate['top_left_y']
            
            transformed_img = crop.crop_img(transformed_img,top_left_x,top_left_y,crop_imgw,crop_imgh)
            imgwh = [crop_imgw,crop_imgh]
            operate['top_left_x'],operate['top_left_y'] = top_left_x,top_left_y
            operate['crop_w'],operate['crop_h'] = crop_imgw,crop_imgh

        elif 'flip' == operate['opt']:
            flip_type = operate['flip_type']
            if operate['random_flip']:
                flip_type = random.randint(-1,1)
            transformed_img=flip.flip_img(transformed_img,flip_type)
            operate['flip_type'] = flip_type
    return transformed_img,certain_transforms       
    
    
def mosaic_img_xml(img,part_img,createdxml,part_xml_tree,transforms,start_row,start_col):
    '''子图和xml嵌入
    Args:
        img:总图
        part_img:嵌入图
        createdxml：总图创建的xml CreateXML对象
        part_xml_tree：嵌入图xml,ET.parse()
        transforms:转换操作
        start_row,start_col:子图嵌入起始行列
    return:
        img:总图
        createdxml:总图创建的xml CreateXML对象
    '''
    transformed_img,certain_transforms = transform_img(part_img,transforms)
    img = mosaic_img(img,transformed_img,start_row,start_col)
    createdxml= transform_xml(part_xml_tree,createdxml,certain_transforms,start_row,start_col)
    return img,createdxml
    

def generate_img_xml(img_save_name,imgw,imgh,part_imgw,part_imgh,transforms,imgs_dir,xmls_dir):
    '''生成拼接图和拼接xml
    Args:
        img_save_name:
        imgw,imgh:生成总图宽高
        transforms:转换操作
        imgs_dir:图源目录
        xmls_dir:图源对应的xml目录
    return:
        img:总图
        createdxml：总图创建的xml,ET.parse()
    '''
    createdxml = CreateXML(img_save_name,imgw,imgh,3)
    img = np.zeros((imgh, imgw, 3), dtype=np.uint8)
    
    part_cols = int(imgw / part_imgw)
    part_rows = int(imgh / part_imgh)
    
    for row in range(part_rows):
        for col in range(part_cols):
            start_row = row*part_imgh
            start_col = col*part_imgw
            
            part_img_file = utils.randomChoiceIn(imgs_dir)
            part_img = cv2.imread(os.path.join(imgs_dir,part_img_file))
            
            part_xml_file = os.path.join(xmls_dir,part_img_file.split('.')[0]+'.xml') 
            part_xml_tree = voc_xml.get_xml_tree(part_xml_file)
            
            img,createdxml = mosaic_img_xml(img,part_img,createdxml,part_xml_tree,transforms,start_row,start_col)
    return img,createdxml
            

def generate_img_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,name_suffix,\
                              count,imgw,imgh,part_imgw,part_imgh,transforms):
    '''批量拼接图片和xml
    Args:
        imgs_dir,xmls_dir:源图片和xml路径
        imgs_save_dir,xmls_save_dir：图片和xml保存路径
        name_suffix: 处理完成的图片、xml的命名标识
        count:生成图片数量
        imgw,imgh:目标拼接图片宽高
        part_imgw,part_imgh:拼接子图宽高
        transforms:转换操作[{'opt':'rotate','cterxy':[],'imgwh':[],'rot_angle':0,'randomRotation':False,\
                            'randomAngleRange':[0,360],'scale':1.0,'correction':True,'bk_imgs_dir':'xxx'},
                        {'opt':'crop','crop_type':RANDOM_CROP,'dsize':(0,0),'top_left_x':0,'top_left_y':0,'fw':0.5,'fh':0.7,'random_wh':False ,'iou_thr':0.5},
                        {'opt':'flip','flip_type':-1,'random_flip':True,'imgwh':[]},
                        {'opt':'resize','fx':0.5,'fy':0.5,'dsize':(0,0),'imgwh':[]}]        
    '''
    for n in range(count):
        img_save_name  = name_suffix+'_'+str(n)+'.jpg'
        img,createdxml = generate_img_xml(img_save_name,imgw,imgh,part_imgw,part_imgh,transforms,imgs_dir,xmls_dir)
        cv2.imwrite(os.path.join(imgs_save_dir,img_save_name),img)
        createdxml.save_xml(xmls_save_dir,img_save_name.split('.')[0]+'.xml')
        
    
def main():
    imgs_dir ='C:/Users/zxl/Desktop/test/JPEGImages/'
    bk_imgs_dir ='C:/Users/zxl/Desktop/test/back/'
    xmls_dir = 'C:/Users/zxl/Desktop/test/Annotations/'
    
    imgs_save_dir= 'C:/Users/zxl/Desktop/test/mosaic_imgs/'
    if not os.path.exists(imgs_save_dir):
        os.makedirs(imgs_save_dir)
    xmls_save_dir='C:/Users/zxl/Desktop/test/mosaic_xmls/'
    if not os.path.exists(xmls_save_dir):
        os.makedirs(xmls_save_dir)
    
    name_suffix='mosaic' #命名标识
    count = 10 #拼接100张图片
    imgw,imgh = 800,600 #每张拼接图的大小
    part_imgw,part_imgh = int(imgw/4),int(imgh/3)
    
#    transforms = [{'opt':'rotate','cterxy':[],'imgwh':[],'rot_angle':0,'randomRotation':False,\
#                        'randomAngleRange':[0,360],'scale':1.0,'correction':True,'bk_imgs_dirs':bk_imgs_dir},
#                  {'opt':'crop','crop_type':'RANDOM_CROP','dsize':(0,0),'top_left_x':0,'top_left_y':0,\
#                          'fw':0.7,'fh':0.7,'random_wh':False ,'iou_thr':0.5,'imgwh':[]},
#                        {'opt':'flip','flip_type':-1,'random_flip':True,'imgwh':[]},
#                        {'opt':'resize','fx':0.5,'fy':0.5,'dsize':(0,0),'imgwh':[]}]  

    transforms = [     {'opt':'rotate','cterxy':[],'imgwh':[],'rot_angle':0,'randomRotation':True,\
                        'randomAngleRange':[0,360],'scale':1.0,'correction':True,'bk_imgs_dir':bk_imgs_dir},
                        {'opt':'crop','crop_type':'RANDOM_CROP','dsize':(0,0),'top_left_x':0,'top_left_y':0,\
                         'crop_w':0,'crop_h':0,'fw':0.6,'fh':0.6,'random_wh':True,'iou_thr':0.5,'imgwh':[]},
                        {'opt':'flip','flip_type':-1,'random_flip':True,'imgwh':[]},
                        {'opt':'resize','fx':0.5,'fy':0.5,'dsize':(part_imgw,part_imgh),'imgwh':[]}] 
    generate_img_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,name_suffix,\
                              count,imgw,imgh,part_imgw,part_imgh,transforms)
if __name__=='__main__':
    main()