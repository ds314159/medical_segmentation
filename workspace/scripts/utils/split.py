import os
import pandas as pd
import shutil
from tqdm import tqdm

# Configuration
INPUT_DIR = "../../datasets/"
INPUT_IMAGES_DIR = os.path.join(INPUT_DIR, "images")
INPUT_MASKS_GRAY_DIR = os.path.join(INPUT_DIR, "masks_labels")
INPUT_MASKS_RGB_DIR = os.path.join(INPUT_DIR, "masks_rgb")
SPLIT_CSV = os.path.join(INPUT_DIR, "dataset_split.csv")

# Création des dossiers de sortie
for split in ['val', 'test']:
    os.makedirs(os.path.join(INPUT_DIR, f"{split}_images"), exist_ok=True)
    os.makedirs(os.path.join(INPUT_DIR, f"{split}_masks_labels"), exist_ok=True)
    os.makedirs(os.path.join(INPUT_DIR, f"{split}_masks_rgb"), exist_ok=True)

def organize_val_test_data():
    """Organise les images de validation et de test dans leurs dossiers respectifs"""

    if not os.path.exists(SPLIT_CSV):
        print(f"ERREUR: Fichier de split {SPLIT_CSV} non trouvé!")
        return False

    # Charger le fichier CSV
    try:
        df = pd.read_csv(SPLIT_CSV)
        print(f"Fichier de split chargé: {len(df)} entrées")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV: {e}")
        return False

    # Filtrer pour obtenir uniquement les images de validation et de test
    val_files = df[df['split'] == 'val']['image_filename'].tolist()
    test_files = df[df['split'] == 'test']['image_filename'].tolist()

    print(f"Images de validation trouvées: {len(val_files)}")
    print(f"Images de test trouvées: {len(test_files)}")

    # Fonction pour copier les fichiers
    def copy_files(file_list, split):
        success_count = 0
        error_files = []

        for img_name in tqdm(file_list, desc=f"Copie des fichiers {split}"):
            # Chemins des fichiers source
            img_src = os.path.join(INPUT_IMAGES_DIR, img_name)
            mask_gray_src = os.path.join(INPUT_MASKS_GRAY_DIR, img_name)
            mask_rgb_src = os.path.join(INPUT_MASKS_RGB_DIR, img_name)

            # Chemins des fichiers destination
            img_dst = os.path.join(INPUT_DIR, f"{split}_images", img_name)
            mask_gray_dst = os.path.join(INPUT_DIR, f"{split}_masks_labels", img_name)
            mask_rgb_dst = os.path.join(INPUT_DIR, f"{split}_masks_rgb", img_name)

            # Vérifier que tous les fichiers source existent
            if not all([os.path.exists(p) for p in [img_src, mask_gray_src, mask_rgb_src]]):
                missing = []
                if not os.path.exists(img_src): missing.append("image")
                if not os.path.exists(mask_gray_src): missing.append("masque grayscale")
                if not os.path.exists(mask_rgb_src): missing.append("masque RGB")
                error_files.append(f"{img_name} (manquant: {', '.join(missing)})")
                continue

            try:
                # Copier les fichiers
                shutil.copy2(img_src, img_dst)
                shutil.copy2(mask_gray_src, mask_gray_dst)
                shutil.copy2(mask_rgb_src, mask_rgb_dst)
                success_count += 1
            except Exception as e:
                error_files.append(f"{img_name} (erreur: {str(e)})")

        return success_count, error_files

    # Copier les fichiers de validation
    print("\nTraitement des fichiers de validation...")
    val_success, val_errors = copy_files(val_files, "val")

    # Copier les fichiers de test
    print("\nTraitement des fichiers de test...")
    test_success, test_errors = copy_files(test_files, "test")

    # Rapport final
    print("\n=== Rapport de création des dossiers val/test ===")
    print(f"Validation: {val_success}/{len(val_files)} fichiers copiés avec succès")
    print(f"Test: {test_success}/{len(test_files)} fichiers copiés avec succès")

    if val_errors or test_errors:
        print("\nProblèmes rencontrés:")
        for err in val_errors + test_errors:
            print(f"  - {err}")

    # Vérifier le nombre de fichiers dans chaque dossier
    val_images = len(os.listdir(os.path.join(INPUT_DIR, "val_images")))
    val_masks_gray = len(os.listdir(os.path.join(INPUT_DIR, "val_masks_gray")))
    val_masks_rgb = len(os.listdir(os.path.join(INPUT_DIR, "val_masks_rgb")))

    test_images = len(os.listdir(os.path.join(INPUT_DIR, "test_images")))
    test_masks_gray = len(os.listdir(os.path.join(INPUT_DIR, "test_masks_gray")))
    test_masks_rgb = len(os.listdir(os.path.join(INPUT_DIR, "test_masks_rgb")))

    print("\nNombre de fichiers dans les dossiers créés:")
    print(f"  Validation: {val_images} images, {val_masks_gray} masques gray, {val_masks_rgb} masques RGB")
    print(f"  Test: {test_images} images, {test_masks_gray} masques gray, {test_masks_rgb} masques RGB")

    return val_success > 0 and test_success > 0

if __name__ == "__main__":
    print("=== Création des dossiers de validation et de test ===")

    success = organize_val_test_data()

    if success:
        print("\n Organisation des données terminée avec succès!")
        print("Les fichiers ont été organisés dans les dossiers suivants:")
        print(f"  - {os.path.join(INPUT_DIR, 'val_images')}")
        print(f"  - {os.path.join(INPUT_DIR, 'val_masks_gray')}")
        print(f"  - {os.path.join(INPUT_DIR, 'val_masks_rgb')}")
        print(f"  - {os.path.join(INPUT_DIR, 'test_images')}")
        print(f"  - {os.path.join(INPUT_DIR, 'test_masks_gray')}")
        print(f"  - {os.path.join(INPUT_DIR, 'test_masks_rgb')}")
    else:
        print("\n Problèmes lors de l'organisation des données. Veuillez vérifier les erreurs ci-dessus.")