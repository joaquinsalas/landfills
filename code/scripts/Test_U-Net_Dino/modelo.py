from transformers import AutoModel  # Para cargar el modelo de hugging face
import torch  # Motor de deep learning
import torch.nn as nn  # Modulo de redes neuronales

class UNetDINO(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: DINOv2-large preentrenado
        self.encoder = AutoModel.from_pretrained("facebook/dinov2-large")
        for param in self.encoder.parameters():
            param.requires_grad = False  # Congela el encoder al inicio

        # Decoder: sube de 1024 canales a 1 mascara (dinov2-large usa 1024 no 768)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]

        # Extrae features del encoder sin guardar gradientes
        with torch.no_grad():
            outputs = self.encoder(pixel_values=x, output_hidden_states=True)

        # Tomar los patch tokens ignorando el CLS token
        features = outputs.last_hidden_state[:, 1:, :]

        # Reshape a mapa espacial 14x14
        h = w = int(features.shape[1] ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, 1024, h, w)  # 1024 para large

        # Decoder sube de 14x14 a 224x224
        mask = self.decoder(features)
        mask = nn.functional.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)

        return mask