import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_model(model_class, checkpoint_path, device, in_channels=3, num_classes=4):
    """
    Charge un modèle à partir d'un checkpoint sauvegardé.

    Args:
        model_class: La classe du modèle (ex: OriginalUNet)
        checkpoint_path: Chemin vers le fichier de checkpoint (.pth)
        device: Device sur lequel charger le modèle (CPU ou CUDA)
        in_channels: Nombre de canaux d'entrée (3 pour RGB)
        num_classes: Nombre de classes de segmentation (inclut le fond)

    Returns:
        Le modèle chargé, prêt pour l'inférence
    """
    model = model_class(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)

    # Vérifier si le fichier existe
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le fichier de checkpoint {checkpoint_path} n'existe pas")

    # Charger les poids du modèle
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Passer en mode évaluation

    return model


def predict_image(model, image_path, transform, device):
    """
    Applique le modèle de segmentation sur une image donnée.

    Args:
        model: Le modèle de segmentation chargé
        image_path: Chemin vers l'image à segmenter
        transform: Fonction de transformation à appliquer à l'image
        device: Device sur lequel exécuter l'inférence

    Returns:
        pred: Le masque prédit (indices de classe)
        image: L'image originale
    """
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas")

    # Charger et prétraiter l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Appliquer la transformation
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Faire la prédiction
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

    return pred, image


def save_test_predictions(model, test_loader, output_dir, device, class_colors=None):
    """
    Génère et sauvegarde les prédictions pour toutes les images du jeu de test.

    Args:
        model: Le modèle de segmentation chargé
        test_loader: DataLoader pour le jeu de test
        output_dir: Dossier où sauvegarder les prédictions
        device: Device sur lequel exécuter l'inférence
        class_colors: Liste des couleurs pour chaque classe (optionnel)
    """
    model.eval()
    predictions_dir = os.path.join(output_dir, 'test_predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Définir les couleurs par défaut si non spécifiées
    if class_colors is None:
        class_colors = [
            [0, 0, 0],  # Fond - noir
            [0, 128, 0],  # Poumon droit - vert
            [128, 0, 0],  # Poumon gauche - rouge
            [0, 0, 255]  # Cœur - bleu
        ]
    class_colors = np.array(class_colors)

    # Itérer sur toutes les images du jeu de test
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Sauvegarde des prédictions")):
            images = images.to(device)
            masks = masks.cpu().numpy()

            # Prédire les masques
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # Pour chaque image dans le batch
            for j, (pred, mask) in enumerate(zip(preds, masks)):
                idx = i * test_loader.batch_size + j

                # Créer le masque prédit coloré
                pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for cls_idx, color in enumerate(class_colors):
                    pred_color[pred == cls_idx] = color

                # Créer le masque de vérité terrain coloré
                gt_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for cls_idx, color in enumerate(class_colors):
                    gt_color[mask == cls_idx] = color

                # Sauvegarder l'image prédite
                cv2.imwrite(
                    os.path.join(predictions_dir, f'pred_{idx}.png'),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                )

                # Sauvegarder la vérité terrain
                cv2.imwrite(
                    os.path.join(predictions_dir, f'gt_{idx}.png'),
                    cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
                )

                # Sauvegarder la prédiction brute (indices de classe)
                np.save(os.path.join(predictions_dir, f'pred_{idx}.npy'), pred)

    print(f"Prédictions sauvegardées dans {predictions_dir}")


def visualize_prediction(pred_mask, original_image, output_path=None, class_colors=None, alpha=0.5):
    """
    Affiche et/ou sauvegarde une visualisation d'une prédiction de segmentation superposée à l'image originale.

    Args:
        pred_mask: Le masque prédit (indices de classe)
        original_image: L'image originale
        output_path: Chemin pour sauvegarder l'image (optionnel)
        class_colors: Liste des couleurs pour chaque classe (optionnel)
        alpha: Transparence de la superposition (entre 0 et 1)
    """
    # Définir les couleurs par défaut si non spécifiées
    if class_colors is None:
        class_colors = [
            [0, 0, 0],  # Fond - noir
            [0, 128, 0],  # Poumon droit - vert
            [128, 0, 0],  # Poumon gauche - rouge
            [0, 0, 255]  # Cœur - bleu
        ]
    class_colors = np.array(class_colors)

    # Créer un masque coloré
    pred_color = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for cls_idx, color in enumerate(class_colors):
        pred_color[pred_mask == cls_idx] = color

    # Créer un masque d'opacité (fond transparent, segmentations visibles)
    opacity = np.zeros_like(pred_mask, dtype=float)
    opacity[pred_mask > 0] = alpha  # Rendre le fond transparent

    # Superposer le masque prédit sur l'image originale
    overlay = original_image.copy().astype(float)
    for c in range(3):  # Pour chaque canal RGB
        overlay[:, :, c] = (1 - opacity) * original_image[:, :, c] + opacity * pred_color[:, :, c]

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Afficher les images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Segmentation superposée")
    plt.axis('off')

    plt.tight_layout()

    # Sauvegarder si un chemin est fourni
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisation sauvegardée dans {output_path}")

    plt.show()

    return overlay  # Retourner l'image superposée


def batch_inference(model, dataloader, device, output_dir=None, save_visualizations=False, class_colors=None,
                    alpha=0.5):
    """
    Exécute l'inférence sur un jeu de données complet et calcule les métriques.

    Args:
        model: Le modèle de segmentation chargé
        dataloader: DataLoader pour le jeu de données
        device: Device sur lequel exécuter l'inférence
        output_dir: Dossier où sauvegarder les résultats (optionnel)
        save_visualizations: Si True, sauvegarde les visualisations pour chaque image
        class_colors: Liste des couleurs pour chaque classe (optionnel)
        alpha: Transparence de la superposition (entre 0 et 1)

    Returns:
        Un dictionnaire contenant les métriques calculées sur l'ensemble du jeu de données
    """
    model.eval()

    # Initialiser un DataFrame pour stocker les métriques par image
    import pandas as pd
    metrics_data = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if save_visualizations:
            os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    num_classes = 4  # Fond + 3 classes
    class_names = ["Fond", "Poumon droit", "Poumon gauche", "Coeur"]

    all_metrics = {
        'iou_per_class': [[] for _ in range(num_classes)],
        'dice_per_class': [[] for _ in range(num_classes)],
        'pixel_accuracy': []
    }

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(dataloader, desc="Évaluation")):
            # Obtenir les noms de fichiers à partir du dataloader
            # Si votre dataloader ne fournit pas les noms, vous pouvez utiliser l'index
            batch_indices = list(range(i * dataloader.batch_size, (i + 1) * dataloader.batch_size))
            batch_indices = batch_indices[:len(images)]  # Ajuster pour le dernier batch

            images = images.to(device)
            masks = masks.cpu().numpy()

            # Prédire les masques
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # Pour chaque image dans le batch
            for j, (pred, mask) in enumerate(zip(preds, masks)):
                idx = batch_indices[j]
                filename = f"image_{idx}"  # À remplacer par le vrai nom si disponible

                # Calculer les métriques
                metrics = calculate_metrics(pred, mask, num_classes)

                # Accumuler les métriques pour les moyennes globales
                for cls in range(num_classes):
                    all_metrics['iou_per_class'][cls].append(metrics['iou_per_class'][cls])
                    all_metrics['dice_per_class'][cls].append(metrics['dice_per_class'][cls])
                all_metrics['pixel_accuracy'].append(metrics['pixel_accuracy'])

                # Ajouter les métriques de cette image au DataFrame
                row = {
                    'filename': filename,
                    'pixel_accuracy': metrics['pixel_accuracy'],
                    'mean_iou': np.mean(metrics['iou_per_class']),
                    'mean_dice': np.mean(metrics['dice_per_class'])
                }

                # Ajouter les métriques par classe
                for cls in range(num_classes):
                    row[f'iou_{class_names[cls]}'] = metrics['iou_per_class'][cls]
                    row[f'dice_{class_names[cls]}'] = metrics['dice_per_class'][cls]

                metrics_data.append(row)

                # Sauvegarder les visualisations si demandé
                if output_dir and save_visualizations:
                    # Récupérer l'image originale
                    img = images[j].cpu().numpy().transpose(1, 2, 0)
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    img = (img * 255).astype(np.uint8)

                    # Créer le masque prédit coloré
                    pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for cls_idx, color in enumerate(class_colors or [[0, 0, 0], [0, 128, 0], [128, 0, 0], [0, 0, 255]]):
                        pred_color[pred == cls_idx] = color

                    # Créer un masque d'opacité
                    opacity = np.zeros_like(pred, dtype=float)
                    opacity[pred > 0] = alpha

                    # Superposer
                    overlay = img.copy()
                    for c in range(3):
                        overlay[:, :, c] = (1 - opacity) * img[:, :, c] + opacity * pred_color[:, :, c]

                    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                    # Sauvegarder l'image superposée
                    plt.figure(figsize=(10, 8))
                    plt.imshow(overlay)
                    plt.title(f"IoU: {row['mean_iou']:.3f}, Dice: {row['mean_dice']:.3f}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'visualizations', f'{filename}_overlay.png'))
                    plt.close()

    # Créer le DataFrame final et le sauvegarder
    metrics_df = pd.DataFrame(metrics_data)
    if output_dir:
        metrics_df.to_csv(os.path.join(output_dir, 'metrics_per_image.csv'), index=False)

    # Calculer les moyennes finales
    result = {
        'mean_iou': np.mean([np.mean(cls_ious) for cls_ious in all_metrics['iou_per_class']]),
        'mean_dice': np.mean([np.mean(cls_dices) for cls_dices in all_metrics['dice_per_class']]),
        'pixel_accuracy': np.mean(all_metrics['pixel_accuracy']),
        'iou_per_class': [np.mean(cls_ious) for cls_ious in all_metrics['iou_per_class']],
        'dice_per_class': [np.mean(cls_dices) for cls_dices in all_metrics['dice_per_class']],
    }

    # Sauvegarder les résultats au format JSON
    if output_dir:
        import json
        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(result, f, indent=4)

    return result, metrics_df


def compare_models(models, image_path, transforms, device, output_path=None, class_colors=None):
    """
    Compare les prédictions de plusieurs modèles sur une même image.

    Args:
        models: Dictionnaire {nom_modèle: modèle}
        image_path: Chemin vers l'image à segmenter
        transforms: Dictionnaire {nom_modèle: transform} ou transform unique
        device: Device sur lequel exécuter l'inférence
        output_path: Chemin pour sauvegarder l'image (optionnel)
        class_colors: Liste des couleurs pour chaque classe (optionnel)
    """
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas")

    # Définir les couleurs par défaut si non spécifiées
    if class_colors is None:
        class_colors = [
            [0, 0, 0],  # Fond - noir
            [0, 128, 0],  # Poumon droit - vert
            [128, 0, 0],  # Poumon gauche - rouge
            [0, 0, 255]  # Cœur - bleu
        ]
    class_colors = np.array(class_colors)

    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Nombre de modèles + image originale
    n_plots = len(models) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))

    # Afficher l'image originale
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    # Pour chaque modèle
    for i, (model_name, model) in enumerate(models.items(), 1):
        # Obtenir la transformation appropriée
        if isinstance(transforms, dict):
            transform = transforms[model_name]
        else:
            transform = transforms

        # Appliquer la transformation
        transformed = transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        # Faire la prédiction
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]

        # Créer un masque coloré
        pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cls_idx, color in enumerate(class_colors):
            pred_color[pred == cls_idx] = color

        # Afficher la prédiction
        axes[i].imshow(pred_color)
        axes[i].set_title(f"Modèle: {model_name}")
        axes[i].axis('off')

    plt.tight_layout()

    # Sauvegarder si un chemin est fourni
    if output_path:
        plt.savefig(output_path)
        print(f"Comparaison sauvegardée dans {output_path}")

    plt.show()


def calculate_metrics(pred_mask, gt_mask, num_classes):
    """
    Calcule diverses métriques de segmentation entre une prédiction et une vérité terrain.

    Args:
        pred_mask: Le masque prédit (indices de classe)
        gt_mask: Le masque de vérité terrain (indices de classe)
        num_classes: Nombre de classes (inclut le fond)

    Returns:
        Un dictionnaire contenant les métriques calculées
    """
    metrics = {
        'iou_per_class': [],
        'dice_per_class': [],
        'pixel_accuracy': np.mean(pred_mask == gt_mask)
    }

    # Calculer IoU et Dice pour chaque classe
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            iou = 1.0  # Si la classe n'est pas présente dans la prédiction ni dans la GT
        else:
            iou = intersection / union

        # Dice coefficient (F1 score)
        if pred_cls.sum() + gt_cls.sum() == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / (pred_cls.sum() + gt_cls.sum())

        metrics['iou_per_class'].append(iou)
        metrics['dice_per_class'].append(dice)

    # Calculer les moyennes
    metrics['mean_iou'] = np.mean(metrics['iou_per_class'])
    metrics['mean_dice'] = np.mean(metrics['dice_per_class'])

    return metrics


