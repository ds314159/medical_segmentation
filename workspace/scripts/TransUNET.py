########################################################################################################################
###### Script Complet, modifiable, implémentant TRANS-UNET son entrainement sur nos notre dataset la sauvegarde #######
############################## et le nécessaire d'une évaluation complète ##############################################

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
from tqdm import tqdm
import pandas as pd
import time
import json


# Configuration des chemins d'accès
DATASET_DIR = "../datasets/"
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "images_augmented")
TRAIN_MASKS_DIR = os.path.join(DATASET_DIR, "masks_gray_augmented")
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, "val_images")
VAL_MASKS_DIR = os.path.join(DATASET_DIR, "val_masks_gray")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "test_images")
TEST_MASKS_DIR = os.path.join(DATASET_DIR, "test_masks_gray")
OUTPUT_DIR = "../results/TransUNET"  # Nouveau dossier pour cette implémentation

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paramètres généraux
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 3  # Poumon gauche, poumon droit, cœur
IMG_SIZE = 512
BATCH_SIZE = 16  # maximum possible avec nos ressources et cette architecture
LEARNING_RATE = 1e-4
EPOCHS = 50
MASK_VALUES = {
    "background": 0,
    "poumon_droit": 100,
    "poumon_gauche": 150,
    "coeur": 255
}

# Classe pour le dataset
class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Lister tous les fichiers d'images
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = os.path.join(self.masks_dir, os.path.basename(img_path))

        # Charger l'image et le masque
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Charger le masque grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convertir les valeurs de pixels en indices de classe (0, 1, 2, 3)
        # 0: background, 1: poumon droit, 2: poumon gauche, 3: cœur
        mask_encoded = np.zeros_like(mask)
        mask_encoded[mask == MASK_VALUES["poumon_droit"]] = 1
        mask_encoded[mask == MASK_VALUES["poumon_gauche"]] = 2
        mask_encoded[mask == MASK_VALUES["coeur"]] = 3

        # Appliquer transformations (redimensionnement et normalisation)
        if self.transform:
            transformed = self.transform(image=image, mask=mask_encoded)
            image = transformed["image"]
            mask_encoded = transformed["mask"]

        return image, mask_encoded

# Fonction pour les transformations - identique à celle utilisée précédemment
def get_transforms():
    """Appliquer redimensionnement et normalisation"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ======= IMPLÉMENTATION DES COMPOSANTS DU TRANSFORMER ========

class MultiHeadAttention(nn.Module):
    """Implémentation de l'attention multi-têtes pour le Transformer"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # nécessaire pour TorchScript (ne peut pas traiter directement un tenseur comme un tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """Implémentation du MLP (Multi-Layer Perceptron) pour le Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Block du Transformer avec attention multi-têtes et MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DoubleConv(nn.Module):
    """
    Double bloc de convolution comme dans U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Ajout de BatchNorm pour stabilité
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ======= IMPLÉMENTATION DE TransUNET ==========================================
# ====== Architecture U-NET améliorée avec des têtes d'attention ===============
# ==============================================================================

class TransUNet(nn.Module):
    """
    Implémentation de TransUNet - combinaison de l'architecture U-Net avec des Transformers
    Version allégée pour un modèle de petite taille .
    """
    def __init__(self, in_channels=3, num_classes=4, img_size=224, patch_size=16,
                 embed_dim=256, depth=4, num_heads=8, mlp_ratio=4., qkv_bias=True):
        super(TransUNet, self).__init__()

        # Dimensions des filtres pour le CNN
        self.filters = [32, 64, 128, 256, 512]  # Réduits par rapport à la version originale

        # Partie 1: CNN Encoder (partie gauche du U)
        self.down1 = DoubleConv(in_channels, self.filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(self.filters[0], self.filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConv(self.filters[1], self.filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = DoubleConv(self.filters[2], self.filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Partie 2: Transformer Encoder
        # Projection des features CNN vers l'espace d'embedding du transformer
        self.patch_embed = nn.Conv2d(self.filters[3], embed_dim, kernel_size=1)

        # Position embeddings
        # Initialisé à 4096 mais utilisé de manière adaptative
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Blocs du Transformer (réduits)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=0.1, attn_drop=0.1)
            for _ in range(depth)
        ])

        # Projection pour revenir à une feature map 2D
        self.transformer_head = nn.Linear(embed_dim, self.filters[3])

        # Partie 3: Fond du U
        self.bottom = DoubleConv(self.filters[3], self.filters[4])

        # Partie 4: CNN Decoder (partie droite du U)
        self.up4 = nn.ConvTranspose2d(self.filters[4], self.filters[3], kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(self.filters[4], self.filters[3])  # Après concaténation, la dimension de canal double

        self.up3 = nn.ConvTranspose2d(self.filters[3], self.filters[2], kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(self.filters[3], self.filters[2])

        self.up2 = nn.ConvTranspose2d(self.filters[2], self.filters[1], kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(self.filters[2], self.filters[1])

        self.up1 = nn.ConvTranspose2d(self.filters[1], self.filters[0], kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(self.filters[1], self.filters[0])

        # Couche finale - convolution 1x1 pour produire la carte de segmentation
        self.final_conv = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder CNN
        # Encoder CNN
        x1 = self.down1(x)                    # shape: B, 32, 512, 512
        x2 = self.down2(self.pool1(x1))       # shape: B, 64, 256, 256
        x3 = self.down3(self.pool2(x2))       # shape: B, 128, 128, 128
        x4 = self.down4(self.pool3(x3))       # shape: B, 256, 64, 64

        # Après pooling (avant Transformer)
        x4_pooled = self.pool4(x4)            # shape: B, 256, 32, 32

        # Transformer Encoder
        B, C, H, W = x4_pooled.shape          # shape: B, 256, 32, 32

        # Projeter les features CNN vers l'espace d'embedding
        x_proj = self.patch_embed(x4_pooled)  # shape: B, embed_dim, 32, 32

        # Transformer en séquence
        x_tokens = x_proj.flatten(2).transpose(1, 2)  # shape: B, 1024, embed_dim

        # Adapter positional embeddings à la taille réelle des tokens
        num_tokens = x_tokens.shape[1]
        pos_embed_resized = self.pos_embed[:, :num_tokens, :]  # shape: 1, 1024, embed_dim

        # Ajouter les positional embeddings
        x_tokens = x_tokens + pos_embed_resized
        x_tokens = self.pos_drop(x_tokens)

        # Traitement par les blocs Transformer
        for block in self.transformer_blocks:
            x_tokens = block(x_tokens)

        # Retransformer les tokens en feature map
        x5 = self.transformer_head(x_tokens)  # shape: B, 1024, 256
        x5 = x5.transpose(1, 2).reshape(B, 256, H, W)  # shape: B, 256, 32, 32

        # Traitement du fond du U
        x_bottom = self.bottom(x4_pooled)     # shape: B, 512, 32, 32

        # Décodeur CNN avec skip connections
        x = self.up4(x_bottom)                # shape: B, 256, 64, 64
        x = torch.cat([x4, x], dim=1)         # shape: B, 512, 64, 64
        x = self.upconv4(x)                   # shape: B, 256, 64, 64

        x = self.up3(x)                       # shape: B, 128, 128, 128
        x = torch.cat([x3, x], dim=1)         # shape: B, 256, 128, 128
        x = self.upconv3(x)                   # shape: B, 128, 128, 128

        x = self.up2(x)                       # shape: B, 64, 256, 256
        x = torch.cat([x2, x], dim=1)         # shape: B, 128, 256, 256
        x = self.upconv2(x)                   # shape: B, 64, 256, 256

        x = self.up1(x)                       # shape: B, 32, 512, 512
        x = torch.cat([x1, x], dim=1)         # shape: B, 64, 512, 512
        x = self.upconv1(x)                   # shape: B, 32, 512, 512

        # Convolution finale pour obtenir la carte de segmentation
        return self.final_conv(x)             # shape: B, num_classes, 512, 512


# Fonction de perte combinant Dice Loss et Cross-Entropy - identique à celle précédente
class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_weight = 0.5

    def forward(self, inputs, targets):
        # Cross-Entropy Loss
        ce_loss = self.ce_loss(inputs, targets)

        # Dice Loss
        inputs_softmax = F.softmax(inputs, dim=1)
        inputs_softmax = inputs_softmax.float()
        targets_one_hot = F.one_hot(targets, inputs.size(1)).permute(0, 3, 1, 2).float()

        # Calculer Dice pour chaque classe et faire la moyenne
        dice_numerator = 2.0 * (inputs_softmax * targets_one_hot).sum(dim=[2, 3])
        dice_denominator = inputs_softmax.sum(dim=[2, 3]) + targets_one_hot.sum(dim=[2, 3]) + 1e-6
        dice_score = (dice_numerator / dice_denominator).mean(dim=0)
        dice_loss = 1.0 - dice_score.mean()

        # Combiner les pertes
        return ce_loss + self.dice_weight * dice_loss

# Fonction pour compter les paramètres du modèle
def count_parameters(model):
    """Compter les paramètres entraînables et non-entraînables du modèle"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params

    return {
        'trainable': trainable_params,
        'frozen': frozen_params,
        'total': total_params
    }

# Fonction d'entraînement - réutiliser celle du script original
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    best_val_iou = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'lr': [],
        'epoch_time': []
    }

    # Scheduler pour réduire le learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
            for images, masks in t:
                images = images.to(device)
                masks = masks.long().to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)

                # Backward pass et optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                t.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_iou, class_ious = evaluate_model(model, val_loader, criterion, device)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # Enregistrer l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Time: {epoch_time:.1f}s")
        print(f"Class IoUs: BG: {class_ious[0]:.4f}, Poumon D: {class_ious[1]:.4f}, Poumon G: {class_ious[2]:.4f}, Coeur: {class_ious[3]:.4f}")

        # Mise à jour du scheduler
        scheduler.step(val_loss)

        # Sauvegarder le meilleur modèle selon la perte de validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
            print(f"Modèle sauvegardé avec val_loss: {val_loss:.4f}")

        # Sauvegarder le meilleur modèle selon l'IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_iou.pth'))
            print(f"Modèle sauvegardé avec val_iou: {val_iou:.4f}")

        # Sauvegarder le modèle à chaque 10 époques
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'history': history
            }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))

    return model, history

# Fonction d'évaluation - réutiliser celle du script original
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.long().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)

            # Convertir les prédictions en classe
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            all_preds.extend(preds)
            all_masks.extend(masks)

    val_loss /= len(dataloader.dataset)

    # Convertir en arrays pour calculer IoU
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    # Calculer IoU par classe
    class_ious = []
    for cls in range(CLASSES + 1):  # +1 pour le fond
        if np.sum(all_masks == cls) == 0 and np.sum(all_preds == cls) == 0:
            # Si la classe n'est pas présente dans les prédictions ni dans les masques, IoU = 1
            class_ious.append(1.0)
        elif np.sum(all_masks == cls) == 0 or np.sum(all_preds == cls) == 0:
            # Si la classe est présente dans les masques mais pas dans les prédictions (ou vice versa), IoU = 0
            class_ious.append(0.0)
        else:
            # Calculer l'IoU pour cette classe
            intersection = np.sum((all_masks == cls) & (all_preds == cls))
            union = np.sum((all_masks == cls) | (all_preds == cls))
            class_ious.append(intersection / union)

    # IoU moyen sur toutes les classes
    mean_iou = np.mean(class_ious)

    return val_loss, mean_iou, class_ious

# Fonction pour visualiser les prédictions - réutiliser celle du script original
def visualize_predictions(model, test_loader, device, num_samples=5):
    model.eval()

    # Classe à couleur pour la visualisation
    class_colors = [
        [0, 0, 0],        # Fond - noir
        [0, 128, 0],      # Poumon droit - vert
        [128, 0, 0],      # Poumon gauche - rouge
        [0, 0, 255]       # Cœur - bleu
    ]
    class_colors = np.array(class_colors)

    # Récupérer quelques échantillons
    batch_idx = 0
    samples_count = 0

    # Créer un subplot
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.cpu().numpy()

            # Prédire les masques
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # Convertir les images en numpy pour visualisation
            images = images.cpu().numpy()

            # Pour chaque image dans le batch
            for i in range(min(len(images), num_samples - samples_count)):
                # Récupérer image, masque réel et masque prédit
                img = images[i].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                mask = masks[i]
                pred = preds[i]

                # Dénormaliser l'image
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                # Créer les masques colorés
                mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for cls_idx, color in enumerate(class_colors):
                    mask_color[mask == cls_idx] = color

                pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for cls_idx, color in enumerate(class_colors):
                    pred_color[pred == cls_idx] = color

                # Afficher image, masque réel et masque prédit
                axes[samples_count, 0].imshow(img)
                axes[samples_count, 0].set_title("Image")
                axes[samples_count, 0].axis('off')

                axes[samples_count, 1].imshow(mask_color)
                axes[samples_count, 1].set_title("Masque Réel")
                axes[samples_count, 1].axis('off')

                axes[samples_count, 2].imshow(pred_color)
                axes[samples_count, 2].set_title("Masque Prédit")
                axes[samples_count, 2].axis('off')

                samples_count += 1
                if samples_count >= num_samples:
                    break

            if samples_count >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_samples.png'))
    plt.close()

# Fonction pour tester le modèle
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Évaluation sur le jeu de test"):
            images = images.to(device)
            masks = masks.cpu().numpy()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_masks.extend(masks)

    # Convertir en arrays pour calculer les métriques
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    # Calculer IoU par classe
    class_ious = []
    class_names = ["Background", "Poumon Droit", "Poumon Gauche", "Cœur"]

    for cls in range(CLASSES + 1):  # +1 pour le fond
        if np.sum(all_masks == cls) == 0 and np.sum(all_preds == cls) == 0:
            # Si la classe n'est pas présente dans les prédictions ni dans les masques, IoU = 1
            iou = 1.0
        elif np.sum(all_masks == cls) == 0 or np.sum(all_preds == cls) == 0:
            # Si la classe est présente dans les masques mais pas dans les prédictions (ou vice versa), IoU = 0
            iou = 0.0
        else:
            # Calculer l'IoU pour cette classe
            intersection = np.sum((all_masks == cls) & (all_preds == cls))
            union = np.sum((all_masks == cls) | (all_preds == cls))
            iou = intersection / union

        class_ious.append(iou)
        print(f"IoU pour {class_names[cls]}: {iou:.4f}")

    mean_iou = np.mean(class_ious)
    print(f"IoU moyen: {mean_iou:.4f}")

    # Calculer la précision pixel-wise
    pixel_accuracy = np.mean(all_preds == all_masks)
    print(f"Précision pixel-wise: {pixel_accuracy:.4f}")

    # Enregistrer les résultats
    results = {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "class_ious": dict(zip(class_names, class_ious))
    }

    # Sauvegarder les résultats au format JSON
    with open(os.path.join(OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results

# Fonction principale pour l'entraînement de TransUNet
def main():
    print(f"=== Entraînement TransUNet pour segmentation thoracique ===")
    print(f"Device utilisé: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Créer les transformations (normalisation et redimensionnement)
    transform = get_transforms()

    # Créer les datasets
    train_dataset = ChestXrayDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform=transform)
    val_dataset = ChestXrayDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=transform)
    test_dataset = ChestXrayDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, transform=transform)

    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Nombre d'images d'entraînement: {len(train_dataset)}")
    print(f"Nombre d'images de validation: {len(val_dataset)}")
    print(f"Nombre d'images de test: {len(test_dataset)}")

    # Créer le modèle TransUNet (version allégée)
    model = TransUNet(in_channels=3, num_classes=CLASSES+1, img_size=IMG_SIZE, patch_size=16,
                    embed_dim=256, depth=4, num_heads=8, mlp_ratio=4., qkv_bias=True)
    model = model.to(DEVICE)

    # Afficher le résumé du modèle et les paramètres
    param_stats = count_parameters(model)
    print(f"\nModèle: TransUNet (lightweight)")
    print(f"Paramètres totaux: {param_stats['total']:,}")
    print(f"Paramètres entraînables: {param_stats['trainable']:,}")

    # Définir la fonction de perte et l'optimiseur
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Entraîner le modèle
    print("\nDébut de l'entraînement...")
    start_time = time.time()
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE)
    total_time = time.time() - start_time

    print(f"\nEntraînement terminé en {total_time/60:.1f} minutes!")

    # Sauvegarder l'historique d'entraînement
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Courbes de Perte')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.title('IoU de Validation')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Taux d\'Apprentissage')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'))

    # Charger le meilleur modèle selon IoU
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model_iou.pth')
    model.load_state_dict(torch.load(best_model_path))

    # Visualiser les prédictions
    print("\nGénération des visualisations...")
    visualize_predictions(model, test_loader, DEVICE, num_samples=5)

    # Évaluer sur le jeu de test
    print("\nÉvaluation sur le jeu de test...")
    test_results = test_model(model, test_loader, DEVICE)

    print("\n=== Rapport Final ===")
    print(f"Meilleur IoU de validation: {max(history['val_iou']):.4f}")
    print(f"IoU moyen sur le jeu de test: {test_results['mean_iou']:.4f}")
    print(f"Précision pixel-wise sur le jeu de test: {test_results['pixel_accuracy']:.4f}")
    print(f"IoU par classe:")
    for cls_name, iou in test_results['class_ious'].items():
        print(f"  - {cls_name}: {iou:.4f}")

    print("\nRésultats et modèles sauvegardés dans:", OUTPUT_DIR)
    print("=== Terminé ===")

if __name__ == "__main__":
    main()