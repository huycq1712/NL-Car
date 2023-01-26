import torch
from torch import nn as nn
from torch import functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPConfig, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPProcessor

class SiameseLocalGlobal(nn.Module):
    
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.device = self.model_cfg['device']
        self.embed_dim = self.model_cfg['embed_dim'] #256
        self.in_embed_dim = self.model_cfg['in_embed_dim'] #512
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        #self.processor_images = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.local_visual_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.global_visual_extractor = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.local_visual_projector = nn.Linear(self.in_embed_dim, self.embed_dim)
        self.global_visual_projector = nn.Linear(self.in_embed_dim, self.embed_dim)
        
        self.local_text_extractor = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.global_text_extractor = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.local_text_projector = nn.Linear(self.in_embed_dim, self.embed_dim)
        self.global_text_projector = nn.Linear(self.in_embed_dim, self.embed_dim)
        
        self.cat_visual_projector = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.cat_text_projector = nn.Linear(self.embed_dim*2, self.embed_dim)
        
        self.local_text_instance_head = nn.Linear(self.embed_dim, model_cfg['num_classes'])
        self.local_image_instance_head = nn.Linear(self.embed_dim, model_cfg['num_classes'])
        
        self.global_text_instance_head = nn.Linear(self.embed_dim, model_cfg['num_classes'])
        self.global_image_instance_head = nn.Linear(self.embed_dim, model_cfg['num_classes'])
        
        self.cat_text_instance_head = nn.Linear(self.embed_dim, model_cfg['num_classes'])
        self.cat_image_instance_head = nn.Linear(self.embed_dim, model_cfg['num_classes'])
        
    
    def encode_images(self, images, global_images):
        local_image_features = self.local_visual_extractor(images).image_embeds 
        global_image_features = self.global_visual_extractor(global_images).image_embeds 
        
        local_image_features = self.local_visual_projector(local_image_features)
        global_image_features = self.global_visual_projector(global_image_features)
        
        cat_image_features = torch.cat([local_image_features, global_image_features], dim=1)
        cat_image_features = self.cat_visual_projector(cat_image_features)
        
        return cat_image_features, local_image_features, global_image_features
    
    
    def encode_texts(self, texts, global_texts):
        local_texts = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        global_texts = self.tokenizer(global_texts, padding=True, return_tensors="pt").to(self.device)
        
        local_text_features = self.local_text_extractor(**local_texts).text_embeds 
        global_text_features = self.global_text_extractor(**global_texts).text_embeds 
        
        local_text_features = self.local_text_projector(local_text_features)
        global_text_features = self.global_text_projector(global_text_features)
        
        cat_text_features = torch.cat([local_text_features, global_text_features], dim=1)
        cat_text_features = self.cat_text_projector(cat_text_features)
        
        return cat_text_features, local_text_features, global_text_features
        
    
    def forward(self, local_images, global_images, local_texts, global_texts):
        
        #local_images = self.processor_images(images=local_images, return_tensors="pt").to(self.device)
        local_texts = self.tokenizer(local_texts, padding=True, return_tensors="pt").to(self.device)
        
        #global_images = self.processor_images(images=global_images, return_tensors="pt").to(self.device)
        global_texts = self.tokenizer(global_texts, padding=True, return_tensors="pt").to(self.device)
        
        local_image_features = self.local_visual_extractor(local_images).image_embeds 
        global_image_features = self.global_visual_extractor(global_images).image_embeds 
        local_text_features = self.local_text_extractor(**local_texts).text_embeds 
        global_text_features = self.global_text_extractor(**global_texts).text_embeds 
        
        local_image_features = self.local_visual_projector(local_image_features)
        global_image_features = self.global_visual_projector(global_image_features)
        local_text_features = self.local_text_projector(local_text_features)
        global_text_features = self.global_text_projector(global_text_features)
        
        cat_image_features = torch.cat([local_image_features, global_image_features], dim=1)
        cat_text_features = torch.cat([local_text_features, global_text_features], dim=1)
        
        cat_image_features = self.cat_visual_projector(cat_image_features)
        cat_text_features = self.cat_text_projector(cat_text_features)
        
        instance_local_image_logits = self.local_image_instance_head(local_image_features)
        instace_local_text_logits = self.local_text_instance_head(local_text_features)
        
        instance_global_image_logits = self.global_image_instance_head(global_image_features)
        instance_global_text_logits = self.global_text_instance_head(global_text_features)
        
        instance_cat_image_logits = self.cat_image_instance_head(cat_image_features)
        instance_cat_text_logits = self.cat_text_instance_head(cat_text_features)
        
        cat_image_features, cat_text_features, local_image_features, local_text_features, global_image_features, global_text_features = map(lambda t: F.normalize(t, p = 2, dim = -1), (cat_image_features, cat_text_features, local_image_features, local_text_features, global_image_features, global_text_features))
        
        return {
            "pairs": [(cat_image_features, cat_text_features), (local_image_features, local_text_features), (global_image_features, global_text_features)], 
            "instance_logits": [instance_cat_image_logits, instance_cat_text_logits, instance_local_image_logits, instace_local_text_logits, instance_global_image_logits, instance_global_text_logits]
        }


def build_model(model_cfg):
    model = SiameseLocalGlobal(model_cfg)
    return model.to(model_cfg['device'])   
        
if __name__ == '__main__':
    
    model_cfg = {
        'device': 'cuda',
        'embed_dim': 256,
        'in_embed_dim': 512,
        
    }
    input_image = torch.rand(4 ,3, 224, 224).to('cuda')
    
    input_text = "A cute cat"
    
    input = {
        'crop_image': input_image,
        'global_image': input_image,
        'motion_image': input_image,
        'nl': [input_text]*4,
        'nl_other': [input_text]*4,
        'car_noun': [input_text]*4,
    }
    
    model = SiameseLocalGlobal(model_cfg).to('cuda')
    
    output = model(input['crop_image'], input['global_image'], input['nl'], input['nl_other'])
