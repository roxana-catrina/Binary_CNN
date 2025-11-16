"""
Model Hibrid: Combină TumorClassifier custom cu ResNet18
Trei strategii de combinare implementate
"""

import torch
import torch.nn as nn
import torchvision.models as models


class HybridTumorClassifier(nn.Module):
    """
    Model hibrid care combină features din:
    1. TumorClassifier custom (modelul tău)
    2. ResNet18 pre-trained (transfer learning)

    Apoi concatenează features și face predicția finală
    """

    def __init__(self, num_classes=3, input_size=224, fusion_type='concat'):
        """
        Args:
            num_classes: Numărul de clase (3 pentru tine)
            input_size: Dimensiunea input (224)
            fusion_type: Tipul de combinare
                - 'concat': Concatenează features (RECOMANDAT)
                - 'add': Adună features
                - 'attention': Folosește attention mechanism
        """
        super(HybridTumorClassifier, self).__init__()

        self.fusion_type = fusion_type
        print(f"[HYBRID] Creating hybrid model with fusion type: {fusion_type}")

        # ==================== BRANCH 1: MODELUL TĂU CUSTOM ====================
        print("[HYBRID] Initializing Custom CNN branch...")
        self.custom_features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )

        # Calculează dimensiunea features custom
        self.custom_feature_size = self._get_conv_output(self.custom_features, input_size)
        print(f"[HYBRID] Custom CNN feature size: {self.custom_feature_size}")

        # ==================== BRANCH 2: RESNET18 ====================
        print("[HYBRID] Loading ResNet18 pre-trained...")
        self.resnet = models.resnet18(pretrained=True)

        # Extrage numărul de features din ResNet
        self.resnet_feature_size = self.resnet.fc.in_features  # 512 pentru ResNet18

        # Îndepărtează ultimul layer (fc) pentru a extrage doar features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        print(f"[HYBRID] ResNet18 feature size: {self.resnet_feature_size}")

        # ==================== FEATURE FUSION ====================
        if fusion_type == 'concat':
            # Concatenează features din ambele modele
            combined_features = self.custom_feature_size + self.resnet_feature_size
            print(f"[HYBRID] Concatenated features: {combined_features}")

            self.fusion_classifier = nn.Sequential(
                nn.Linear(combined_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        elif fusion_type == 'add':
            # Proiectează ambele features la aceeași dimensiune și le adună
            target_dim = 512

            self.custom_projection = nn.Linear(self.custom_feature_size, target_dim)
            self.resnet_projection = nn.Linear(self.resnet_feature_size, target_dim)

            self.fusion_classifier = nn.Sequential(
                nn.BatchNorm1d(target_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(target_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        elif fusion_type == 'attention':
            # Folosește attention pentru a pondera features
            combined_features = self.custom_feature_size + self.resnet_feature_size

            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(combined_features, 256),
                nn.Tanh(),
                nn.Linear(256, 2),  # 2 weights: pentru custom și resnet
                nn.Softmax(dim=1)
            )

            # Project to same dimension
            target_dim = 512
            self.custom_projection = nn.Linear(self.custom_feature_size, target_dim)
            self.resnet_projection = nn.Linear(self.resnet_feature_size, target_dim)

            self.fusion_classifier = nn.Sequential(
                nn.BatchNorm1d(target_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(target_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        print(f"[HYBRID] Hybrid model initialized successfully! ✅")

    def _get_conv_output(self, conv_layers, input_size):
        """Calculează dimensiunea output după layerele convolutional"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            output = conv_layers(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        # ==================== EXTRACT FEATURES ====================
        # Features din modelul custom
        custom_features = self.custom_features(x)
        custom_features = custom_features.view(custom_features.size(0), -1)

        # Features din ResNet
        resnet_features = self.resnet(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)

        # ==================== FEATURE FUSION ====================
        if self.fusion_type == 'concat':
            # Concatenare simplă
            combined = torch.cat([custom_features, resnet_features], dim=1)
            output = self.fusion_classifier(combined)

        elif self.fusion_type == 'add':
            # Proiectare și adunare
            custom_proj = self.custom_projection(custom_features)
            resnet_proj = self.resnet_projection(resnet_features)
            combined = custom_proj + resnet_proj
            output = self.fusion_classifier(combined)

        elif self.fusion_type == 'attention':
            # Attention-based fusion
            # Concatenare pentru attention weights
            concat_for_attention = torch.cat([custom_features, resnet_features], dim=1)
            attention_weights = self.attention(concat_for_attention)

            # Proiectare
            custom_proj = self.custom_projection(custom_features)
            resnet_proj = self.resnet_projection(resnet_features)

            # Weighted sum
            combined = (attention_weights[:, 0:1] * custom_proj +
                       attention_weights[:, 1:2] * resnet_proj)

            output = self.fusion_classifier(combined)

        return output


class EnsembleTumorClassifier(nn.Module):
    """
    Model Ensemble: Predicții separate din fiecare model, apoi average
    Mai simplu dar tot foarte eficient!
    """

    def __init__(self, num_classes=3, input_size=224):
        super(EnsembleTumorClassifier, self).__init__()

        print("[ENSEMBLE] Creating ensemble model...")

        # ==================== MODEL 1: CUSTOM CNN ====================
        self.custom_features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )

        custom_feature_size = self._get_conv_output(self.custom_features, input_size)

        self.custom_classifier = nn.Sequential(
            nn.Linear(custom_feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # ==================== MODEL 2: RESNET18 ====================
        self.resnet = models.resnet18(pretrained=True)
        resnet_feature_size = self.resnet.fc.in_features

        # Înlocuiește ultimul layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet_feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        print("[ENSEMBLE] Ensemble model initialized! ✅")

    def _get_conv_output(self, conv_layers, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            output = conv_layers(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        # Predicții din modelul custom
        custom_out = self.custom_features(x)
        custom_out = custom_out.view(custom_out.size(0), -1)
        custom_pred = self.custom_classifier(custom_out)

        # Predicții din ResNet
        resnet_pred = self.resnet(x)

        # Average predictions (ensemble)
        output = (custom_pred + resnet_pred) / 2.0

        return output


# Pentru compatibilitate cu codul existent
def create_hybrid_model(num_classes=3, input_size=224, model_type='hybrid_concat'):
    """
    Factory function pentru a crea modelul dorit

    Args:
        model_type:
            - 'hybrid_concat': Feature concatenation (RECOMANDAT) ⭐⭐⭐⭐⭐
            - 'hybrid_add': Feature addition ⭐⭐⭐⭐
            - 'hybrid_attention': Attention fusion ⭐⭐⭐⭐⭐
            - 'ensemble': Ensemble averaging ⭐⭐⭐⭐
    """

    if model_type == 'hybrid_concat':
        return HybridTumorClassifier(num_classes, input_size, fusion_type='concat')
    elif model_type == 'hybrid_add':
        return HybridTumorClassifier(num_classes, input_size, fusion_type='add')
    elif model_type == 'hybrid_attention':
        return HybridTumorClassifier(num_classes, input_size, fusion_type='attention')
    elif model_type == 'ensemble':
        return EnsembleTumorClassifier(num_classes, input_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test modelul
    import torch

    print("\n" + "="*70)
    print("TESTARE MODELE HIBRIDE")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test fiecare tip
    for model_type in ['hybrid_concat', 'hybrid_add', 'hybrid_attention', 'ensemble']:
        print(f"\n{'='*70}")
        print(f"Testing: {model_type}")
        print(f"{'='*70}")

        model = create_hybrid_model(num_classes=3, input_size=224, model_type=model_type)
        model = model.to(device)

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)

        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✅ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

