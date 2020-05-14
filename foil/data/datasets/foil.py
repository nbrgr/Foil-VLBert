import os
import json
import _pickle as cPickle
from PIL import Image
import base64
import numpy as np
import time
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.bbox import bbox_iou_py_vectorized

from pycocotools.coco import COCO
from .foil_api.foil import FOIL


class Foil(Dataset):
    def __init__(self, root_path, data_path, boxes='gt', proposal_source='official',
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, **kwargs):
        """
        Foil Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to dataset
        :param boxes: boxes to use, 'gt' or 'proposal'
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(Foil, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        coco_annot_files = {
            "train2014": "annotations/instances_train2014.json",
            "val2014": "annotations/instances_val2014.json",
            "test2015": "annotations/image_info_test2015.json",
        }

        foil_annot_files = {
            "train": "foil/foilv1.0_train_2017.json",
            "test": "foil/foilv1.0_test_2017.json"
        }

        foil_vocab_file = "foil/vocab.txt"

        self.vg_proposal = ("vgbua_res101_precomputed", "trainval2014_resnet101_faster_rcnn_genome")

        self.test_mode = test_mode
        self.data_path = data_path
        self.root_path = root_path
        self.transform = transform

        vocab_file = open(os.path.join(data_path, foil_vocab_file), 'r')
        vocab_lines = vocab_file.readlines()
        vocab_lines = [v.strip() for v in vocab_lines]
        self.itos = vocab_lines 
        self.stoi = dict(list(zip(self.itos, range(len(vocab_lines)))))

        if self.test_mode:
            self.image_set = "val2014"
            coco_annot_file = coco_annot_files["val2014"]
        else:
            self.image_set = "train2014"
            coco_annot_file = coco_annot_files["train2014"]

        self.coco = COCO(annotation_file=os.path.join(data_path, coco_annot_file))
        self.foil = FOIL(data_path, 'train' if not test_mode else 'test')
        self.foil_ids = list(self.foil.Foils.keys())
        self.foils = self.foil.loadFoils(foil_ids=self.foil_ids)
        if 'proposal' in boxes:
            with open(os.path.join(data_path, proposal_dets), 'r') as f:
                proposal_list = json.load(f)
            self.proposals = {}
            for proposal in proposal_list:
                image_id = proposal['image_id']
                if image_id in self.proposals:
                    self.proposals[image_id].append(proposal['box'])
                else:
                    self.proposals[image_id] = [proposal['box']]
        self.boxes = boxes
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_annotations()
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

    @property
    def data_names(self):
        return ['image', 'boxes', 'im_info', 'expression', 'label', 'pos', 'target', 'mask']

    def __getitem__(self, index):
        idb = self.database[index]

        # image related
        img_id = idb['image_id']
        image = self._load_image(idb['image_fn'])
        im_info = torch.as_tensor([idb['width'], idb['height'], 1.0, 1.0])
        #if not self.test_mode:
        #    gt_box = torch.as_tensor(idb['gt_box'])
        flipped = False
        if self.boxes == 'gt':
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            boxes = []
            for ann in anns:
                x_, y_, w_, h_ = ann['bbox']
                boxes.append([x_, y_, x_ + w_, y_ + h_])
            boxes = torch.as_tensor(boxes)
        elif self.boxes == 'proposal':
            if self.proposal_source == 'official':
                boxes = torch.as_tensor(self.proposals[img_id])
                boxes[:, [2, 3]] += boxes[:, [0, 1]]
            elif self.proposal_source == 'vg':
                box_file = os.path.join(self.data_path, self.vg_proposal[0], '{0}.zip@/{0}'.format(self.vg_proposal[1]))
                boxes_fn = os.path.join(box_file, '{}.json'.format(idb['image_id']))
                boxes_data = self._load_json(boxes_fn)
                boxes = torch.as_tensor(
                    np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape(
                        (boxes_data['num_boxes'], -1))
                )
            else:
                raise NotImplemented
        elif self.boxes == 'proposal+gt' or self.boxes == 'gt+proposal':
            if self.proposal_source == 'official':
                boxes = torch.as_tensor(self.proposals[img_id])
                boxes[:, [2, 3]] += boxes[:, [0, 1]]
            elif self.proposal_source == 'vg':
                box_file = os.path.join(self.data_path, self.vg_proposal[0], '{0}.zip@/{0}'.format(self.vg_proposal[1]))
                boxes_fn = os.path.join(box_file, '{}.json'.format(idb['image_id']))
                boxes_data = self._load_json(boxes_fn)
                boxes = torch.as_tensor(
                    np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape(
                        (boxes_data['num_boxes'], -1))
                )
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            gt_boxes = []
            for ann in anns:
                x_, y_, w_, h_ = ann['bbox']
                gt_boxes.append([x_, y_, x_ + w_, y_ + h_])
            gt_boxes = torch.as_tensor(gt_boxes)
            boxes = torch.cat((boxes, gt_boxes), 0)
        else:
            raise NotImplemented

        if self.add_image_as_a_box:
            w0, h0 = im_info[0], im_info[1]
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)

        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)


        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)


        # assign label expression with the foil annotation
        label = idb['label']
        foil_pos = idb['pos']

        # expression
        exp = idb['caption_tokens']
        exp_ids = self.tokenizer.convert_tokens_to_ids(exp)

        target = self.stoi[idb['target_word']]
        mask = idb['mask']
        if self.test_mode:
            return image, boxes, im_info, exp_ids, label, foil_pos, target, mask
        else:
            return image, boxes, im_info, exp_ids, label, foil_pos, target, mask


    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_annotations(self):
        tic = time.time()
        database = []
        db_cache_name = 'foil_{}'.format(self.image_set)
        if self.zip_mode:
            db_cache_name = db_cache_name + '_zipmode'
        if self.test_mode:
            db_cache_name = db_cache_name + '_testmode'
        db_cache_root = os.path.join(self.root_path, 'cache')
        db_cache_path = os.path.join(db_cache_root, '{}.pkl'.format(db_cache_name))

        if os.path.exists(db_cache_path):
            if not self.ignore_db_cache:
                # reading cached database
                print('cached database found in {}.'.format(db_cache_path))
                with open(db_cache_path, 'rb') as f:
                    print('loading cached database from {}...'.format(db_cache_path))
                    tic = time.time()
                    database = cPickle.load(f)
                    print('Done (t={:.2f}s)'.format(time.time() - tic))
                    return database
            else:
                print('cached database ignored.')

        # ignore or not find cached database, reload it from annotation file
        print('loading database of split {}...'.format(self.image_set))
        tic = time.time()

        for foil_id, foil in zip(self.foil_ids, self.foils):
            iset = 'train2014'
            if self.zip_mode:
                image_fn = os.path.join(self.data_path, iset + '.zip@/' + iset,
                                        'COCO_{}_{:012d}.jpg'.format(iset, foil['image_id']))
            else:
                image_fn = os.path.join(self.root_path, self.data_path, iset, 'COCO_{}_{:012d}.jpg'.format(iset, foil['image_id']))

            expression_tokens = self.tokenizer.basic_tokenizer.tokenize(foil['caption'])
            expression_wps = []
            for token in expression_tokens:
                expression_wps.extend(self.tokenizer.wordpiece_tokenizer.tokenize(token))

            word_offsets = [0]

            for i, wp in enumerate(expression_wps):
                if wp[0] == '#':
                    #still inside single word
                    continue
                else:
                    #this is the beginning of a new word
                    word_offsets.append(i)

            word_offsets.append(len(expression_wps))

            target_word = foil['target_word']
            foil_word = foil['foil_word']
            target_wps = None
            target_pos = -1
            if foil['foil']:
                foil_wps = self.tokenizer.wordpiece_tokenizer.tokenize(foil_word)
                twps_len = len(foil_wps)
                for i in range(len(expression_wps) - twps_len):
                    if expression_wps[i:i+twps_len] == foil_wps:
                        target_pos = i
                        break
            else:
                twps_len = 1
            idb = {
                'ann_id': foil['id'],
                'foil_id': foil['foil_id'],
                'image_id': foil['image_id'],
                'image_fn': image_fn,
                'width': self.coco.imgs[foil['image_id']]['width'],
                'height': self.coco.imgs[foil['image_id']]['height'],
                'caption': foil['caption'].strip(),
                'caption_tokens': expression_wps,
                'target_word': foil['target_word'],
                'target': self.stoi.get(foil['target_word'], 0),
                'foil_word': foil['foil_word'],
                'label': foil['foil'],
                'pos': target_pos,
                'mask': twps_len
                }
            database.append(idb)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

