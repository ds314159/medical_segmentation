# Projet de Segmentation d'Imageries Médicales

## Description
Ce projet se concentre sur la segmentation anatomique multiclasse de radiographies thoraciques en utilisant différentes variantes d'architecture U-Net. L'implémentation compare trois modèles (U-Net, ResNet U-Net et TransU-Net) pour la segmentation des poumons et du cœur dans les radiographies thoraciques.

## Structure du Projet
```
medical_segmentation/
│
├── workspace/
│   ├── datasets/                      # Dépôt de données
│   │   ├── images/                    # Radiographies originales
│   │   ├── test/                      # Données de test
│   │   │   ├── test_images/           # Images de test
│   │   │   ├── test_masks_gray/       # Masques de segmentation en niveaux de gris
│   │   │   └── test_masks_rgb/        # Masques de segmentation en RGB
│   │   ├── train/                     # Données d'entraînement (celles sur lesquelles il y a eu augmentation)
│   │   │   ├── images_augmented/      # Images augmentées pour l'entraînement
│   │   │   ├── masks_gray_augmented/  # Masques en niveaux de gris augmentés
│   │   │   └── masks_rgb_augmented/   # Masques RGB augmentés
│   │   ├── val/                       # Données de validation
│   │   │   ├── val_images/            # Images de validation
│   │   │   ├── val_masks_gray/        # Masques en niveaux de gris
│   │   │   └── val_masks_rgb/         # Masques RGB
│   │   └── dataset_split.csv          # Fichier définissant la répartition des données
│   │
│   ├── results/                       # Résultats des modèles
│   │   ├── ResnetUNET/                # Résultats du modèle ResnetUNET
│   │   ├── TransUNET/                 # Résultats du modèle TransUNET
│   │   └── UNET/                      # Résultats du modèle UNET
│   │
│   └── scripts/                       # Scripts
│       ├── utils/                     # Utilitaires communs
│       ├── ResnetUNET.py              # Implémentation et évaluation de ResnetUNET
│       ├── TransUNET.py               # Implémentation et évaluation de TransUNET
│       ├── UNET.py                    # Implémentation et évaluation de UNET
│       └── demo.ipynb                 # Notebook de démonstration
│
├── rapport.pdf                        # Rapport du projet au format article
├── README.md                          # Vue d'ensemble et instructions du projet
└── requirements.txt                   # Dépendances du projet
```

## Installation
Pour configurer l'environnement du projet :

```bash
git clone https://github.com/username/medical_segmentation.git
cd medical_segmentation
pip install -r requirements.txt
```

## Données
Le projet utilise des radiographies thoraciques avec des masques de segmentation annotés manuellement pour 3 classes :
- Poumon droit 
- Poumon gauche 
- Cœur 

Les données sont organisées en ensembles d'entraînement, de validation et de test avec des formats de masque en niveaux de gris et en RGB.

## Modèles
Le projet implémente et compare trois architectures de segmentation :
1. **U-Net** : Architecture U-Net originale
2. **ResnetUNET** : U-Net avec backbone ResNet50 pré-entraîné
3. **TransUNET** : U-Net avec mécanismes d'attention basés sur les transformers

## Utilisation
Pour une démonstration des modèles :
```bash
jupyter notebook workspace/scripts/demo.ipynb
```

Pour l'entraînement et l'évaluation des modèles, utiliser les scripts correspondants :
```bash
python workspace/scripts/UNET.py
python workspace/scripts/ResnetUNET.py 
python workspace/scripts/TransUNET.py
```

## Résultats
Les modèles atteignent des performances élevées pour la segmentation multiclasse avec :
- IoU moyen > 0,88
- Précision pixel par pixel > 0,90 (classe background majoritaire à l'origine de cette exceptionnelle performance)

ResNet+U-Net obtient les meilleurs résultats (mIoU 0,8980), tandis que TransUNet montre une efficacité remarquable avec des performances similaires malgré trois fois moins de paramètres.

## Rapport
Pour une méthodologie détaillée, la configuration expérimentale et une analyse complète des résultats, consultez `rapport.pdf`.

## Dépendances
Les dépendances principales incluent :
- PyTorch
- OpenCV
- Albumentations (pour l'augmentation des données)
- NumPy
- Matplotlib

## Licence
Ce projet est uniquement à des fins éducatives.