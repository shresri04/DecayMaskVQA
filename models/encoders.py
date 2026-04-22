import torch
import torchnn as nn
from transformers import AutoModel

class textEncoder(nn.Module):
    def __init__(self, model_name):
        super(textEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class imageEncoder(nn.Module):
    def __init__(self, model_name):
        super(imageEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state

