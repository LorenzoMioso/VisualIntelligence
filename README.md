# VisualIntelligence

## Folder Structure

```bash
src/
├── data/
│   ├── __init__.py
│   ├── dataset.py       # LungCancerDataset class e data loading
│   ├── preprocessing.py # Conversione grayscale e preprocessing
│   └── utils.py         # Funzioni di utility per estrazione e split dataset
├── models/
│   ├── __init__.py
│   └── cnn.py           # Definizione ImageClassifier CNN
├── training/
│   ├── __init__.py
│   ├── trainer.py       # Funzioni di training
│   └── metrics.py       # Calcolo metriche (accuracy, F1)
├── visualization/
│   ├── __init__.py
│   ├── dataset_vis.py   # Funzioni per visualizzare il dataset
│   ├── filters_vis.py   # Visualizzazione filtri CNN
│   └── xai.py           # Metodi XAI (occlusione, saliency)
├── config.py            # Configurazioni e iperparametri
└── main.py              # Script principale
```
