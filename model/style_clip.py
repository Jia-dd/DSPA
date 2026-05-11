import torch.nn as nn
from model.prompt_learner import PromptLearner
from model.text_encoder import TextEncoder
from model.classification_head import ClassificationHead


class StyleCLIP(nn.Module):
    def __init__(self, cfg, classnames, pretrained_clip):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, pretrained_clip)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(pretrained_clip)
        self.image_encoder = pretrained_clip.visual
        self.classification_head = ClassificationHead(cfg.EMBEDDING_DIM, cfg.N_CLS)

        self.dtype = pretrained_clip.dtype

    def image_forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        output = image_features

        return output

    def text_forward(self, input, tokenized_input):
        text_features = self.text_encoder(input, tokenized_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        output = text_features

        return output
