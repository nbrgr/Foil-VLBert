__author__ = 'licheng'

"""
This interface provides access to one dataset:
1) foil
split by unc and google
The following API functions are defined:
FOIL      - FOIL api class
getfoilIds  - get foil ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadfoils   - load foils with the specified foil ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getfoilBox  - get foil's bounding box [x, y, w, h] given the foil_id
showfoil    - show image, segmentation or box of the foilerred object with the foil
getMask    - get mask and area of the foilerred object given foil
showMask   - show mask of the foilerred object given foil
"""

import sys
import os.path as osp
import json
import _pickle as pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np
# from .external import mask


# import cv2
# from skimage.measure import label, regionprops

class FOIL:

    def __init__(self, data_root, split='train'):
        # provide data_root folder which contains foilclef, foilcoco, foilcoco+ and foilcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'foilcoco', splitBy = 'unc'
        print('loading dataset foil into memory...')
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, 'foil')
        self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')

        # load foils from data/dataset/foils(dataset).json
        self.data = {}

        tic = time.time()
        """
        foil_file = osp.join(self.DATA_DIR, 'foils(' + splitBy + ').p')
        self.data['dataset'] = dataset
        self.data['foils'] = pickle.load(open(foil_file, 'rb'))
        """
        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'foilv1.0_' + split + '_2017.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  foils: 	 	{foil_id: foil}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgTofoils: 	{image_id: foils}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  foilToAnn:  	{foil_id: ann}
        # 9)  annTofoil:  	{ann_id: foil}
        # 10) catTofoils: 	{category_id: foils}
        # 11) sentTofoil: 	{sent_id: foil}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, imgToAnns = {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = Anns.get(ann['id'],[]) + [ann]
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img

        # fetch info from foils
        Foils, imgToFoils, annToFoils = {}, {}, {}
        for ann in self.data['annotations']:
            # ids
            foil_id = ann['foil_id']
            ann_id = ann['id']
            image_id = ann['image_id']

            # add mapping related to foil
            Foils[foil_id] = ann
            imgToFoils[image_id] = imgToFoils.get(image_id, []) + [ann]
            annToFoils[ann_id] = annToFoils.get(ann_id, []) + [ann] 

        # create class members
        self.Foils = Foils
        self.Anns = Anns
        self.Imgs = Imgs
        self.imgToFoils = imgToFoils
        self.imgToAnns = imgToAnns
        self.annToFoils = annToFoils
        print('index created.')

    def getFoilIds(self, image_ids=[], ann_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        ann_ids = ann_ids if type(ann_ids) == list else [ann_ids]

        if image_ids == [] and ann_ids == []:
            return list(self.foils.keys())
        
        if image_ids != []:
            img_foil_ids = []
            for img_id in image_ids:
                img_foil_ids.extend([foil['foil_id'] for foil in self.imgToFoils[img_id]])
        else:
            img_foil_ids = self.Foils.keys()

        if ann_ids != []:
            ann_foil_ids = []
            for ann_id in ann_ids:
                ann_foil_ids.extend([foil['foil_id'] for foil in self.annToFoils[ann_id]])
        else:
            ann_foil_ids = self.Foils.keys()

        img_foil_set = set(img_foil_ids)
        ann_foil_set = set(ann_foil_ids)
        foil_ids = ann_foil_set.intersection(img_foil_set)
        foil_ids = list(foil_ids)
        
        return foil_ids

    def getAnnIds(self, image_ids=[], foil_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        foil_ids = foil_ids if type(foil_ids) == list else [foil_ids]

        if image_ids == [] and foil_ids == []:
            return list(self.foils.keys())
        
        if image_ids != []:
            img_ann_ids = []
            for img_id in image_ids:
                img_ann_ids.extend([ann['id'] for ann in self.imgToAnns[img_id]])
        else:
            img_ann_ids = self.Anns.keys()

        if foil_id != []:
            foil_ann_ids = [self.Foils[foil_id]['id'] for foil_id in foil_ids]
        else:
            foil_ann_ids = self.Anns.keys()

        img_foil_set = set(img_foil_ids)
        foil_ann_ids = set(ann_foil_ids)
        foil_ids = foil_ann_ids.intersection(img_foil_set)
        foil_ids = list(foil_ids)
        
        return ann_ids

    def getImgIds(self, foil_ids=[], ann_ids=[]):
        foil_ids = foil_ids if type(foil_ids) == list else [foil_ids]
        ann_ids = ann_ids if type(ann_ids) is list else [ann_ids]

        if foil_ids != []:
            foil_img_ids = [self.Foils[foil_id]['image_id'] for foil_id in foil_ids]
        else:
            foil_img_ids = self.Imgs.keys()

        if ann_ids != []:
            ann_img_ids = []
            for ann_id in ann_ids:
                ann_img_ids.extend([ann['image_id'] for ann in self.Anns[ann_id]])
        else:
            image_ids = self.Imgs.keys()

        foil_img_ids = set(foil_img_ids)
        ann_img_ids = set(ann_img_ids)

        return image_ids

    def loadFoils(self, foil_ids=[]):
        if type(foil_ids) == list:
            return [self.Foils[foil_id] for foil_id in foil_ids]
        elif type(foil_ids) == int:
            return [self.foils[foil_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]


    # def showfoil(self, foil, seg_box='seg'):
    #     ax = plt.gca()
    #     # show image
    #     image = self.Imgs[foil['image_id']]
    #     I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
    #     ax.imshow(I)
    #     # show foiler expression
    #     for sid, sent in enumerate(foil['sentences']):
    #         print('%s. %s' % (sid + 1, sent['sent']))
    #     # show segmentations
    #     if seg_box == 'seg':
    #         ann_id = foil['ann_id']
    #         ann = self.Anns[ann_id]
    #         polygons = []
    #         color = []
    #         c = 'none'
    #         if type(ann['segmentation'][0]) == list:
    #             # polygon used for foilcoco*
    #             for seg in ann['segmentation']:
    #                 poly = np.array(seg).reshape((len(seg) / 2, 2))
    #                 polygons.append(Polygon(poly, True, alpha=0.4))
    #                 color.append(c)
    #             p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
    #             ax.add_collection(p)  # thick yellow polygon
    #             p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
    #             ax.add_collection(p)  # thin red polygon
    #         else:
    #             # mask used for foilclef
    #             rle = ann['segmentation']
    #             m = mask.decode(rle)
    #             img = np.ones((m.shape[0], m.shape[1], 3))
    #             color_mask = np.array([2.0, 166.0, 101.0]) / 255
    #             for i in range(3):
    #                 img[:, :, i] = color_mask[i]
    #             ax.imshow(np.dstack((img, m * 0.5)))
    #     # show bounding-box
    #     elif seg_box == 'box':
    #         ann_id = foil['ann_id']
    #         ann = self.Anns[ann_id]
    #         bbox = self.getfoilBox(foil['foil_id'])
    #         box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
    #         ax.add_patch(box_plot)
    #
    # def getMask(self, foil):
    #     # return mask, area and mask-center
    #     ann = self.foilToAnn[foil['foil_id']]
    #     image = self.Imgs[foil['image_id']]
    #     if type(ann['segmentation'][0]) == list:  # polygon
    #         rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
    #     else:
    #         rle = ann['segmentation']
    #     m = mask.decode(rle)
    #     m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
    #     m = m.astype(np.uint8)  # convert to np.uint8
    #     # compute area
    #     area = sum(mask.area(rle))  # should be close to ann['area']
    #     return {'mask': m, 'area': area}
    #
    # # # position
    # # position_x = np.mean(np.where(m==1)[1]) # [1] means columns (matlab style) -> x (c style)
    # # position_y = np.mean(np.where(m==1)[0]) # [0] means rows (matlab style)    -> y (c style)
    # # # mass position (if there were multiple regions, we use the largest one.)
    # # label_m = label(m, connectivity=m.ndim)
    # # regions = regionprops(label_m)
    # # if len(regions) > 0:
    # # 	largest_id = np.argmax(np.array([props.filled_area for props in regions]))
    # # 	largest_props = regions[largest_id]
    # # 	mass_y, mass_x = largest_props.centroid
    # # else:
    # # 	mass_x, mass_y = position_x, position_y
    # # # if centroid is not in mask, we find the closest point to it from mask
    # # if m[mass_y, mass_x] != 1:
    # # 	print('Finding closes mask point ...')
    # # 	kernel = np.ones((10, 10),np.uint8)
    # # 	me = cv2.erode(m, kernel, iterations = 1)
    # # 	points = zip(np.where(me == 1)[0].tolist(), np.where(me == 1)[1].tolist())  # row, col style
    # # 	points = np.array(points)
    # # 	dist   = np.sum((points - (mass_y, mass_x))**2, axis=1)
    # # 	id     = np.argsort(dist)[0]
    # # 	mass_y, mass_x = points[id]
    # # 	# return
    # # return {'mask': m, 'area': area, 'position_x': position_x, 'position_y': position_y, 'mass_x': mass_x, 'mass_y': mass_y}
    # # # show image and mask
    # # I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
    # # plt.figure()
    # # plt.imshow(I)
    # # ax = plt.gca()
    # # img = np.ones( (m.shape[0], m.shape[1], 3) )
    # # color_mask = np.array([2.0,166.0,101.0])/255
    # # for i in range(3):
    # #     img[:,:,i] = color_mask[i]
    # # ax.imshow(np.dstack( (img, m*0.5) ))
    # # plt.show()
    #
    # def showMask(self, foil):
    #     M = self.getMask(foil)
    #     msk = M['mask']
    #     ax = plt.gca()
    #     ax.imshow(msk)


# if __name__ == '__main__':
#     foiler = foilER(dataset='foilcocog', splitBy='google')
#     foil_ids = foiler.getfoilIds()
#     print(len(foil_ids))
#
#     print(len(foiler.Imgs))
#     print(len(foiler.imgTofoils))
#
#     foil_ids = foiler.getfoilIds(split='train')
#     print('There are %s training foilerred objects.' % len(foil_ids))
#
#     for foil_id in foil_ids:
#         foil = foiler.loadfoils(foil_id)[0]
#         if len(foil['sentences']) < 2:
#             continue
#
#         pprint(foil)
#         print('The label is %s.' % foiler.Cats[foil['category_id']])
#         plt.figure()
#         foiler.showfoil(foil, seg_box='box')
#         plt.show()
#
#     # plt.figure()
#     # foiler.showMask(foil)
#     # plt.show()
