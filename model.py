# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torchvision.models as models

class MedVQAModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_answers=3, cnn_trainable=False, dropout=0.3):
        super().__init__()

        self.cnn_backbone, self.image_feat_dim = self.build_cnn_backbone()
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        for p in self.cnn_backbone.parameters():
            p.requires_grad = cnn_trainable

        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.text_feat_dim = self.bert.config.hidden_size

        self.roi_gate = nn.Sequential(
            nn.Conv2d(self.image_feat_dim, 1, 1),
            nn.Sigmoid()
        )

        fusion_dim = self.image_feat_dim + self.text_feat_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(512, num_answers)

    def build_cnn_backbone(self):
        raise NotImplementedError

    def forward(self, images, input_ids, attention_mask, roi_masks=None):
        features = self.cnn_backbone(images)
        roi_attention = self.roi_gate(features)

        if roi_masks is not None:
            roi_scaled = F.interpolate(
                roi_masks.float().unsqueeze(1),
                size=features.shape[2:], mode="bilinear", align_corners=False
            )
            features = features * (1.0 + roi_scaled * roi_attention)

        pooled = self.cnn_pool(features).squeeze(-1).squeeze(-1)

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_out.last_hidden_state[:, 0, :]

        fused = torch.cat([pooled, text_emb], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)

        return logits, features, roi_attention


class ResNet50MedVQA(MedVQAModel):
    def build_cnn_backbone(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        return backbone, 2048
    
    