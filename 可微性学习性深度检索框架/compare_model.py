import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 模型定义

class BaselineCNN(nn.Module):
    """简单 CNN baseline"""
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet18(nn.Module):
    """标准 ResNet18"""
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    """标准 VGG16"""
    def __init__(self, num_classes=10, pretrained=False):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VisionTransformer(nn.Module):
    """简化版 Vision Transformer"""
    def __init__(self, num_classes=10, img_size=32, patch_size=4, dim=128, depth=6, heads=8, mlp_dim=256):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        self.to_cls_token = nn.Identity()
        self.fc = nn.Linear(dim, num_classes)

        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, img):
        p = self.patch_size
        B, C, H, W = img.shape
        patches = img.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * p * p)

        x = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(x.size(1))]
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.fc(x)


# 工厂函数（返回模型并放到 GPU）
def get_model(name="baseline", num_classes=10, pretrained=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    name = name.lower()
    if name == "baseline":
        model = BaselineCNN(num_classes)
    elif name == "resnet18":
        model = ResNet18(num_classes, pretrained)
    elif name == "vgg16":
        model = VGG16(num_classes, pretrained)
    elif name == "vit":
        model = VisionTransformer(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model.to(device)
