from src.config import DATASET_PATH, TARGET_IMAGE_SIZE, class_0, class_1
from src.dataset import DatasetCreator
from src.visualization.dataset_vis import DatasetVisualizer


def main():
    dataset = DatasetCreator()
    df = dataset.init(TARGET_IMAGE_SIZE)
    dv = DatasetVisualizer(df)
    benign_color, _ = dv.compute_class_color(f"{DATASET_PATH}/{class_0}")
    cancer_color, _ = dv.compute_class_color(f"{DATASET_PATH}/{class_1}")
    dv.visualize_class_colors(benign_color, cancer_color)


if __name__ == "__main__":
    main()
