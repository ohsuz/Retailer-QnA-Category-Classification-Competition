import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_pretrained_bert import BertModel
import pytorch_pretrained_bert
import transformers


class IntentClassifier(nn.Module):

    def __init__(self, model_name):
        super(IntentClassifier, self).__init__()
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 118)
    
    def forward(self, x, mask, segs, target=None):
        if target is not None:
            output = self.model(input_ids = x.long(), attention_mask = mask.float(), token_type_ids = segs.long(), labels=target.unsqueeze(1))
        else:
            output = self.model(input_ids = x.long(), attention_mask = mask.float(), token_type_ids = segs.long())
        return output