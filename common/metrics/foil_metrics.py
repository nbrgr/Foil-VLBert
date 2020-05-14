import torch
from .eval_metric import EvalMetric

class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class ClsAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ClsAccuracy, self).__init__('ClsAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['cls_logits']
            cls_pred = (cls_logits > 0.5).long().view(-1)
            label = outputs['cls_label'].long()
            self.sum_metric += float((cls_pred == label).sum().item())
            self.num_inst += label.numel()


class PosAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(PosAccuracy, self).__init__('PosAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            pos_logits = outputs['pos_logits']
            pos_pred = pos_logits.argmax(dim=-1).view(-1)
            label = outputs['pos_label'].long().view(-1)
            keep = (label >= 0)
            self.sum_metric += float((pos_pred[keep] == label[keep]).sum().item())
            self.num_inst += keep.sum().item()


class CorAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(CorAccuracy, self).__init__('CorAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cor_pred = outputs['cor_logits'].argmax(dim=-1).view(-1).long()
            label = outputs['cor_label'].view(-1).long()
            keep = (label > 0)
            self.sum_metric += float((cor_pred[keep] == label[keep]).sum().item())
            self.num_inst += float(keep.sum().item())

