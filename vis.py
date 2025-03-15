from src.config import MODEL_CONFIG, class_0, class_1
from src.dataset import DataManager
from src.visualization.dataset_vis import DatasetVisualizer


def main():
    data_manager = DataManager()
    df = data_manager.prepare_dataset(MODEL_CONFIG.target_image_size)
    dv = DatasetVisualizer(df)
    benign_color, _ = dv.compute_class_color(f"{DATASET_PATH}/{class_0}")
    cancer_color, _ = dv.compute_class_color(f"{DATASET_PATH}/{class_1}")
    dv.visualize_class_colors(benign_color, cancer_color)


if __name__ == "__main__":
    main()
