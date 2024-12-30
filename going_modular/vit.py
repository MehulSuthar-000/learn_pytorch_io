
class ViT(nn.Module):
    def __init__(self, num_classes=3, dim=768, nheads=12, num_encoder_layers=12):
        super(ViT, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=nheads), num_layers=num_encoder_layers)
        self.fc = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1) # mean pool the encoder outputs
        x = self.fc(x)
        return x
