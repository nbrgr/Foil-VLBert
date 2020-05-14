import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertForPretraining, Task1Head, Task2Head, Task3Head

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        self.task1_head = Task1Head(config.NETWORK.VLBERT)
        self.task2_head = Task2Head(config.NETWORK.VLBERT)
        self.task3_head = Task3Head(config.NETWORK.VLBERT)

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        self.image_feature_extractor.init_weight()

        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        for m in self.task1_head.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.task2_head.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.task3_head.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        
        for param in self.image_feature_extractor.parameters():
            param.requires_grad = False

        for param in self.vlbert.parameters():
            param.requires_grad = False

        for param in self.object_linguistic_embeddings.parameters():
            param.requires_grad


    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      label,
                      pos,
                      target, 
                      mask
                      ):
        ###########################################

        if self.vlbert.training:
            self.vlbert.eval()

        if self.image_feature_extractor.training:
            self.image_feature_extractor.eval()

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        #our labels for foil are binary and 1 dimension
        #label = label[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        sequence_reps, object_reps, _ = self.vlbert(text_input_ids,
                                                      text_token_type_ids,
                                                      text_visual_embeddings,
                                                      text_mask,
                                                      object_vl_embeddings,
                                                      box_mask,
                                                      output_text_and_object_separately=True,
                                                      output_all_encoded_layers=False)

        cls_rep = sequence_reps[:, 0, :].squeeze(1)
        sequence_reps = sequence_reps[:, 1:, :]

        cls_log_probs = self.task1_head(cls_rep)
        
        pos_log_probs = self.task2_head(sequence_reps, text_mask[:, 1:])

        zeros = torch.zeros_like(text_input_ids)
        mask_len = mask.shape[1]
        error_cor_mask = zeros 
        error_cor_mask[:, 1: mask_len + 1] = mask
        error_cor_mask = error_cor_mask.bool()
        text_input_ids[error_cor_mask] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        sequence_reps, _, _ = self.vlbert(text_input_ids,
                                                text_token_type_ids,
                                                text_visual_embeddings,
                                                text_mask,
                                                object_vl_embeddings,
                                                box_mask,
                                                output_text_and_object_separately=True,
                                                output_all_encoded_layers=False)
    
        sequence_reps = sequence_reps[:, 1:, :]
        select_index = pos.view(-1,1).unsqueeze(2).repeat(1,1,768)
        select_index[select_index < 0] = 0
        masked_reps = torch.gather(sequence_reps, 1, select_index).squeeze(1)
        cor_log_probs = self.task3_head(masked_reps)

        loss_mask = label.view(-1).float()

        cls_loss = F.binary_cross_entropy(cls_log_probs.view(-1), label.view(-1).float(), reduction="none")
        pos_loss = F.nll_loss(pos_log_probs, pos, ignore_index=-1, reduction="none").view(-1) * loss_mask
        cor_loss = F.nll_loss(cor_log_probs.view(-1, cor_log_probs.shape[-1]), target, ignore_index=0, reduction="none").view(-1) * loss_mask

        loss = cls_loss.mean() + pos_loss.mean() + cor_loss.mean()

        outputs = {
            "cls_logits": cls_log_probs,
            "pos_logits": pos_log_probs,
            "cor_logits": cor_log_probs,
            "cls_label": label,
            "pos_label": pos,
            "cor_label": target
        }

        return outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression,
                          label,
                          pos,
                          target
    ):

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        origin_len = boxes.shape[1]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        #our labels for foil are binary and 1 dimension
        #label = label[:, :max_len]
        with torch.no_grad():
            obj_reps = self.image_feature_extractor(images=images,
                                                    boxes=boxes,
                                                    box_mask=box_mask,
                                                    im_info=im_info,
                                                    classes=None,
                                                    segms=None)

        ############################################
        # prepare text
            cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
            text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
            text_input_ids[:, 0] = cls_id
            text_input_ids[:, 1:-1] = expression
            _sep_pos = (text_input_ids > 0).sum(1)
            _batch_inds = torch.arange(expression.shape[0], device=expression.device)
            text_input_ids[_batch_inds, _sep_pos] = sep_id
            text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
            text_mask = text_input_ids > 0
            text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))



            object_linguistic_embeddings = self.object_linguistic_embeddings(
                boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
            )
            object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

            sequence_reps, object_reps, _ = self.vlbert(text_input_ids,
                                                        text_token_type_ids,
                                                        text_visual_embeddings,
                                                        text_mask,
                                                        object_vl_embeddings,
                                                        box_mask,
                                                        output_text_and_object_separately=True,
                                                        output_all_encoded_layers=False)

        cls_rep = sequence_reps[:, 0, :].squeeze(1)
        sequence_reps = sequence_reps[:, 1:, :]

        cls_log_probs = self.task1_head(cls_rep)
        
        pos_log_probs = self.task2_head(sequence_reps, text_mask[:, 1:])

        zeros = torch.zeros_like(text_input_ids)
        mask_len = mask.shape[1]
        error_cor_mask = zeros 
        error_cor_mask[:, 1: mask_len + 1] = mask
        error_cor_mask = error_cor_mask.bool()
        text_input_ids[error_cor_mask] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        with torch.no_grad():
            sequence_reps, _, _ = self.vlbert(text_input_ids,
                                                    text_token_type_ids,
                                                    text_visual_embeddings,
                                                    text_mask,
                                                    object_vl_embeddings,
                                                    box_mask,
                                                    output_text_and_object_separately=True,
                                                    output_all_encoded_layers=False)
    
        sequence_reps = sequence_reps[:, 1:, :]
        select_index = pos.view(-1,1).unsqueeze(2).repeat(1,1,768)
        select_index[select_index < 0] = 0
        masked_reps = torch.gather(sequence_reps, 1, select_index).squeeze(1)
        cor_log_probs = self.task3_head(masked_reps)

        loss_mask = label.view(-1).float()

        cls_loss = F.binary_cross_entropy(cls_log_probs.view(-1), label.view(-1).float(), reduction="none")
        pos_loss = F.nll_loss(pos_log_probs, pos, ignore_index=-1, reduction="none").view(-1) * loss_mask
        cor_loss = F.nll_loss(cor_log_probs.view(-1, cor_log_probs.shape[-1]), target, ignore_index=0, reduction="none").view(-1) * loss_mask

        loss = cls_loss.mean() + pos_loss.mean() + cor_loss.mean()

        outputs = {
            "cls_logits": cls_log_probs,
            "pos_logits": pos_log_probs,
            "cor_logits": cor_log_probs,
            "cls_label": label,
            "pos_label": pos,
            "cor_label": target
        }

        return outputs, loss
