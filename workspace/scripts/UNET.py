########################################################################################################################
####### Script Complet, modifiable, implémentant U-NET, son entrainement sur nos notre dataset, la sauvegarde et #######
################################# le nécéssaire d'une évaluation complète ##############################################


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
OUTPUT_DIR = "../results/UNET"  # sortie hist, poids et aperçu de résultats

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paramètres généraux
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 3  # Poumon gauche, droit, cœur (implicitement 4 avec le background)
IMG_SIZE = 512  # Toutes les images seront redimensionnées à 512×512
BATCH_SIZE = 16  # max que l'on peut se permettre ici avec nos ressources mémoire
LEARNING_RATE = 1e-4
EPOCHS = 50

# Valeurs des classes après conversion en gray scale
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
    """Appliquer normalisation et redimensionnement pour garantir des dimensions cohérentes"""
    return A.Compose([
        # Ici, nous faisons le choix du 512*512 par manque de ressources
        A.Resize(IMG_SIZE, IMG_SIZE),
        # Normalisation avec des valeurs standards admises pour ce type
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])



# ======= IMPLÉMENTATION DU U-NET ORIGINAL =====================================
# Implémentation de l'architecture U-Net originale telle que
# décrite dans l'article "U-Net: Convolutional Networks for
# Biomedical Image Segmentation" par Ronneberger et al. (2015)
# ==============================================================================

class DoubleConv(nn.Module):
    """
    Double bloc de convolution du U-Net original.
    Chaque étape du U-Net utilise deux convolutions 3x3 suivies d'une ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Dans le papier original, pas de padding, ce qui réduit la taille de l'image
        # Utilisation de padding=1 pour maintenir la taille constante, ce qui simplifie l'architecture
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OriginalUNet(nn.Module):
    """
    Implémentation du U-Net original avec ses caractéristiques distinctives:
    - Architecture en forme de "U" symétrique
    - Chemins de contraction (encodeur) et d'expansion (décodeur)
    - Skip connections entre niveaux correspondants
    """
    def __init__(self, in_channels=3, num_classes=4):
        super(OriginalUNet, self).__init__()

        # Nombre de filtres initial (64 dans le papier original)
        # Progression des filtres: 64, 128, 256, 512, 1024
        filters = [64, 128, 256, 512, 1024]

        # Chemin de contraction (partie gauche du U) - Encodeur
        # À chaque étape: double convolution + max pooling
        self.down1 = DoubleConv(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = DoubleConv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fond du U (partie la plus profonde)
        self.bottom = DoubleConv(filters[3], filters[4])

        # Chemin d'expansion (partie droite du U) - Décodeur
        # À chaque étape: convolution transposée + concaténation avec skip + double convolution
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(filters[4], filters[3])  # Après concaténation, la dimension de canal double

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(filters[1], filters[0])

        # Couche finale - convolution 1x1 pour produire la carte de segmentation
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Chemin de contraction et stockage des cartes de caractéristiques pour les skip connections
        x1 = self.down1(x)       # Premier niveau d'encodage
        x2 = self.down2(self.pool1(x1))  # Deuxième niveau
        x3 = self.down3(self.pool2(x2))  # Troisième niveau
        x4 = self.down4(self.pool3(x3))  # Quatrième niveau

        # Fond du U
        x5 = self.bottom(self.pool4(x4))

        # Chemin d'expansion avec skip connections
        # Pour chaque niveau: upsampling, concaténation avec la carte correspondante, double convolution
        x = self.up4(x5)
        # Gérer les différences potentielles de dimensions à cause des arrondis lors du pooling
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x], dim=1)  # Concaténation selon l'axe des canaux
        x = self.upconv4(x)

        x = self.up3(x)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.upconv3(x)

        x = self.up2(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.upconv2(x)

        x = self.up1(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.upconv1(x)

        # Convolution finale pour obtenir la carte de segmentation
        return self.final_conv(x)

# Fonction de perte combinant Dice Loss et Cross-Entropy
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

# Fonction pour compter les paramètres du modèle ( ce script est voué à être
# réutilisé avec des modifications, du coup on rajoute tous les outils
# nécéssaires pour généraliser à des modèles inspirés du U-net)
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

# Fonction d'entraînement
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

# Fonction d'évaluation, par souci  de reproductibilité du script
# et d'exécution incluse de certaines évaluations de base ( l'idée est de lancer
# le script et de directement avoir des résultats interprétables)
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

# Fonction pour créer une visualisation des prédictions
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

# Fonction principale
def main():
    print(f"=== Entraînement U-Net Original pour segmentation thoracique ===")
    print(f"Device utilisé: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Créer les transformations (normalisation et redimensionnement)
    transform = get_transforms()

    # Créer les datasets pour le modèle
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

    # Instancier un modèle U-Net original
    model = OriginalUNet(in_channels=3, num_classes=CLASSES+1)  # +1 pour le fond
    model = model.to(DEVICE)

    # Afficher le résumé du modèle et les paramètres
    param_stats = count_parameters(model)
    print(f"\nModèle: U-Net Original")
    print(f"Paramètres totaux: {param_stats['total']:,}")
    print(f"Paramètres entraînables: {param_stats['trainable']:,}")

    # Définir la fonction de perte et l'optimiseur
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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