import os
import numpy as np
import cv2
import glob
import pandas as pd
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import random


# Configuration des chemins d'accès à adapter en fonction de l'arborescence

INPUT_DIR = "../../datasets/"
INPUT_IMAGES_DIR = os.path.join(INPUT_DIR, "images")
INPUT_MASKS_GRAY_DIR = os.path.join(INPUT_DIR, "masks_labels")
INPUT_MASKS_RGB_DIR = os.path.join(INPUT_DIR, "masks_rgb")
OUTPUT_DIR = "../../datasets/"
SPLIT_CSV = os.path.join(INPUT_DIR, "dataset_split.csv")

# Créer les répertoires de sortie pour les données augmentées
os.makedirs(os.path.join(OUTPUT_DIR, "images_augmented"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks_gray_augmented"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks_rgb_augmented"), exist_ok=True)

# Classes à considérer avec valeurs spécifiques
# Format pour les masques grayscale: {classe: valeur_pixel}
CLASSES_GRAY = {
    "poumon_droit": 100,
    "poumon_gauche": 150,
    "coeur": 255
}

# Format pour les masques RGB: {classe: [R, G, B]}
CLASSES_RGB = {
    "poumon_gauche": [255, 0, 0],    # Rouge
    "poumon_droit": [0, 255, 0],     # Vert
    "coeur": [0, 0, 255]             # Bleu
}

def load_dataset_split():
    """Charge le fichier CSV de split et retourne les listes d'images par partition"""
    if not os.path.exists(SPLIT_CSV):
        print(f"ERREUR: Fichier de split {SPLIT_CSV} non trouvé!")
        return None

    df = pd.read_csv(SPLIT_CSV)
    print(f"Fichier de split chargé: {len(df)} entrées")

    # Grouper par split
    train_files = df[df['split'] == 'train']['image_filename'].tolist()
    val_files = df[df['split'] == 'val']['image_filename'].tolist()
    test_files = df[df['split'] == 'test']['image_filename'].tolist()

    print(f"Distribution des images:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")

    return train_files, val_files, test_files

def verify_dataset(train_files):
    """Vérifie la présence de toutes les images et masques du train set"""
    missing_files = []

    for img_name in train_files:
        img_path = os.path.join(INPUT_IMAGES_DIR, img_name)
        gray_mask_path = os.path.join(INPUT_MASKS_GRAY_DIR, img_name)
        rgb_mask_path = os.path.join(INPUT_MASKS_RGB_DIR, img_name)

        if not os.path.exists(img_path):
            missing_files.append(f"Image: {img_path}")
        if not os.path.exists(gray_mask_path):
            missing_files.append(f"Masque grayscale: {gray_mask_path}")
        if not os.path.exists(rgb_mask_path):
            missing_files.append(f"Masque RGB: {rgb_mask_path}")

    if missing_files:
        print("ATTENTION: Fichiers manquants:")
        for file in missing_files[:10]:  # Afficher les 10 premiers manquants
            print(f"  {file}")
        if len(missing_files) > 10:
            print(f"  ... et {len(missing_files)-10} autres")
        return False

    # Vérifier un échantillon de masques
    sample_img_name = train_files[0]
    sample_gray = cv2.imread(os.path.join(INPUT_MASKS_GRAY_DIR, sample_img_name), cv2.IMREAD_GRAYSCALE)
    unique_gray = np.unique(sample_gray)
    print(f"Valeurs uniques dans un masque grayscale échantillon: {unique_gray}")

    return True

def visualize_sample(img_name):
    """Visualise un exemple d'image avec ses masques"""
    img_path = os.path.join(INPUT_IMAGES_DIR, img_name)
    gray_mask_path = os.path.join(INPUT_MASKS_GRAY_DIR, img_name)
    rgb_mask_path = os.path.join(INPUT_MASKS_RGB_DIR, img_name)

    if not all(os.path.exists(path) for path in [img_path, gray_mask_path, rgb_mask_path]):
        print(f"ERREUR: Fichiers manquants pour l'échantillon {img_name}")
        return

    # Charger les images
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray_mask = cv2.imread(gray_mask_path, cv2.IMREAD_GRAYSCALE)
    rgb_mask = cv2.imread(rgb_mask_path)
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)

    # Afficher les images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gray_mask, cmap='gray')
    plt.title("Masque grayscale")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(rgb_mask)
    plt.title("Masque RGB")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_visualization.png"))
    plt.close()
    print(f"Visualisation sauvegardée dans {os.path.join(OUTPUT_DIR, 'sample_visualization.png')}")

def create_augmentation_pipeline():
    """Crée un pipeline d'augmentation adapté selon vos spécifications"""
    return A.Compose([
        # Translations légères (horizontales et verticales)
        A.Affine(
            translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
            p=0.7
        ),

        # Zooms in et out légers
        A.RandomScale(scale_limit=(-0.1, 0.1), p=0.7),

        # Transformations spatiales légères
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),

        # Rotations légères
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),

        # Ajustements d'intensité et de contraste
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),

        # Élastique - très léger pour ne pas déformer l'anatomie
        A.ElasticTransform(alpha=30, sigma=5, alpha_affine=5, p=0.2, border_mode=cv2.BORDER_CONSTANT),

        # Grille de déformation - légère
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2, border_mode=cv2.BORDER_CONSTANT),

        # Simulation de variations de qualité d'image
        A.ImageCompression(quality_lower=85, quality_upper=95, p=0.3),

        # Clahe pour améliorer le contraste local
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),

        # Ajout d'ombres légères - simule variations d'acquisition
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.1)
    ],
    # Important: tous les masques doivent être transformés de la même manière
    additional_targets={'mask_gray': 'mask', 'mask_rgb': 'image'})

def augment_dataset(train_files, augmentation_factor=5):
    """Augmente uniquement les images du train set et les sauvegarde avec la nomenclature demandée"""
    total_original = len(train_files)

    print(f"Début de l'augmentation de données pour {total_original} images du train set...")

    # Pipeline d'augmentation
    transform = create_augmentation_pipeline()

    # Traiter les images originales et créer les augmentations
    for img_name in tqdm(train_files):
        # Construire les chemins des fichiers
        img_path = os.path.join(INPUT_IMAGES_DIR, img_name)
        gray_mask_path = os.path.join(INPUT_MASKS_GRAY_DIR, img_name)
        rgb_mask_path = os.path.join(INPUT_MASKS_RGB_DIR, img_name)

        # Obtenir le nom de base pour la nouvelle convention de nommage
        base_name = img_name.split('.')[0]  # "xray_001"

        # Charger les images
        image = cv2.imread(img_path)
        gray_mask = cv2.imread(gray_mask_path, cv2.IMREAD_GRAYSCALE)
        rgb_mask = cv2.imread(rgb_mask_path)

        # Sauvegarder l'original avec le suffixe _augmented_0
        orig_name = f"{base_name}_augmented_0.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images_augmented", orig_name), image)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "masks_gray_augmented", orig_name), gray_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "masks_rgb_augmented", orig_name), rgb_mask)

        # Générer les augmentations
        for j in range(augmentation_factor):
            # Préparer l'entrée pour le pipeline d'augmentation
            augmented = transform(
                image=image,
                mask_gray=gray_mask,
                mask_rgb=rgb_mask
            )

            aug_image = augmented['image']
            aug_gray_mask = augmented['mask_gray']
            aug_rgb_mask = augmented['mask_rgb']

            # Générer les noms des fichiers augmentés avec le suffixe _augmented_X
            aug_name = f"{base_name}_augmented_{j+1}.png"

            # Sauvegarder les images augmentées
            cv2.imwrite(os.path.join(OUTPUT_DIR, "images_augmented", aug_name), aug_image)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "masks_gray_augmented", aug_name), aug_gray_mask)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "masks_rgb_augmented", aug_name), aug_rgb_mask)

    # Compter les fichiers générés
    total_augmented = len(glob.glob(os.path.join(OUTPUT_DIR, "images_augmented", "*.png")))
    print(f"Augmentation terminée! {total_original} images originales + {total_augmented - total_original} augmentées = {total_augmented} images au total")

def analyze_class_distribution(sample_files):
    """Analyse la distribution des classes dans les masques originaux et augmentés"""
    # Sélectionner un échantillon pour l'analyse
    sample_size = min(50, len(sample_files))

    # Analyse des masques originaux (grayscale)
    print("Analyse de la distribution des classes dans les masques originaux...")
    gray_stats_orig = {cls: 0 for cls in CLASSES_GRAY.keys()}
    total_pixels_orig = 0

    for img_name in random.sample(sample_files, sample_size):
        mask_path = os.path.join(INPUT_MASKS_GRAY_DIR, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for cls_name, value in CLASSES_GRAY.items():
            # Compter les pixels de cette classe
            gray_stats_orig[cls_name] += np.sum(mask == value)
        total_pixels_orig += mask.size

    print("Distribution originale des classes (grayscale):")
    for cls_name, count in gray_stats_orig.items():
        percentage = (count / total_pixels_orig) * 100
        print(f"  {cls_name}: {percentage:.2f}%")

    # Analyse des masques augmentés (grayscale)
    base_names = [img_name.split('.')[0] for img_name in sample_files]
    aug_files = []
    for base in random.sample(base_names, min(10, len(base_names))):
        aug_files.extend([f"{base}_augmented_{i}.png" for i in range(5)])  # Échantillon des augmentations

    gray_stats_aug = {cls: 0 for cls in CLASSES_GRAY.keys()}
    total_pixels_aug = 0

    for aug_name in aug_files:
        mask_path = os.path.join(OUTPUT_DIR, "masks_gray_augmented", aug_name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            for cls_name, value in CLASSES_GRAY.items():
                # Compter les pixels de cette classe
                gray_stats_aug[cls_name] += np.sum(mask == value)
            total_pixels_aug += mask.size

    if total_pixels_aug > 0:
        print("\nDistribution des classes après augmentation (grayscale):")
        for cls_name, count in gray_stats_aug.items():
            percentage = (count / total_pixels_aug) * 100
            print(f"  {cls_name}: {percentage:.2f}%")

def visualize_augmentations(sample_name):
    """Visualise un exemple d'image et ses augmentations"""
    base_name = sample_name.split('.')[0]

    # Trouver toutes les versions augmentées
    aug_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "images_augmented", f"{base_name}_augmented_*.png")))

    if not aug_files:
        print(f"ERREUR: Aucune augmentation trouvée pour {sample_name}")
        return

    # Charger les images
    images = []
    titles = []

    for aug_path in aug_files[:9]:  # Limiter à 9 pour la visualisation
        aug_name = os.path.basename(aug_path)
        aug_num = aug_name.split('_')[-1].split('.')[0]

        img = cv2.imread(aug_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        titles.append(f"Aug {aug_num}")

    # Afficher les images
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "augmentation_examples.png"))
    plt.close()
    print(f"Visualisation des augmentations sauvegardée dans {os.path.join(OUTPUT_DIR, 'augmentation_examples.png')}")

if __name__ == "__main__":
    print("=== Data Augmentation pour Images Médicales ===")

    # Charger la répartition train/val/test depuis le CSV
    split_result = load_dataset_split()

    if split_result:
        train_files, val_files, test_files = split_result

        # Vérifier la présence des fichiers
        print("\nVérification des fichiers du train set...")
        if verify_dataset(train_files):
            # Visualiser un échantillon
            print("\nVisualisation d'un échantillon d'image...")
            visualize_sample(train_files[0])

            # Lancer l'augmentation uniquement sur le train set
            print("\nDébut de l'augmentation des données d'entraînement...")
            augment_dataset(train_files, augmentation_factor=8)  # 8 versions augmentées par image

            # Analyser la distribution des classes avant/après augmentation
            print("\nAnalyse de la distribution des classes...")
            analyze_class_distribution(train_files)

            # Visualiser les augmentations d'une image
            print("\nVisualisation des augmentations d'une image...")
            visualize_augmentations(train_files[0])

            print("\n=== Augmentation terminée! ===")
            print(f"Images augmentées sauvegardées dans:")
            print(f"  {os.path.join(OUTPUT_DIR, 'images_augmented')}")
            print(f"  {os.path.join(OUTPUT_DIR, 'masks_gray_augmented')}")
            print(f"  {os.path.join(OUTPUT_DIR, 'masks_rgb_augmented')}")
        else:
            print("\nProblème détecté dans les fichiers. Veuillez vérifier les chemins et formats.")
    else:
        print("\nImpossible de charger la répartition train/val/test depuis le CSV.")