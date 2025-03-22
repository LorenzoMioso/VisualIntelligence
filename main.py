from torchvision import transforms

from src.config import MODEL_CONFIG, PATH_CONFIG, device
from src.dataset import DataManager
from src.models.cnn import CNNImageClassifier
from src.models.scatnet import ScatNetImageClassifier
from src.models.utils import ModelAnalyzer
from src.training.metrics import TrainingMetrics
from src.training.training import CrossValidationTrainer
from src.visualization.dataset_vis import DatasetVisualizer
from src.visualization.xai import XAI
from src.visualization.xai_captum import XAI_CAPTUM


class Runner:
    """Main class to handle all experiment workflows"""

    def __init__(self):
        """Initialize experiment with a model class"""
        self.model_class = CNNImageClassifier
        self.data_manager = None
        self.df = None
        self.train_splits = None
        self.val_splits = None
        self.stats = None

    def set_model_class(self, model_class):
        """Set the model class for the experiment"""
        if model_class not in [ScatNetImageClassifier, CNNImageClassifier]:
            raise ValueError(
                "Invalid model class. Choose either ScatNetImageClassifier or ScatNetOptimizer."
            )
        self.model_class = model_class

    def prepare_dataset(self):
        """Prepare dataset and return necessary components"""
        self.data_manager = DataManager()
        self.df = self.data_manager.prepare_dataset(MODEL_CONFIG.target_image_size)
        self.train_splits, self.val_splits = self.data_manager.create_splits()
        self.stats = self.data_manager.get_stats_from_file()
        if not self.stats:
            self.data_manager.compute_statistics()
            self.stats = self.data_manager.get_stats_from_file()
        return self

    def get_model(self, checkpoint_id=None):
        print(f"Loading model from checkpoint {checkpoint_id}")
        """Load or create a model"""
        model_analyzer = ModelAnalyzer()

        if checkpoint_id is not None:
            # Load from checkpoint
            checkpoint_path = f"{PATH_CONFIG.model_checkpoint_path}{checkpoint_id}_{self.model_class.__name__}.pth"
            model = model_analyzer.load_checkpoint(checkpoint_path)
        else:
            print(f"Creating model instance of {self.model_class.__name__}")
            # Create new model instance
            model = self.model_class().to(device)

        return model

    def get_loaders(self, fold_id=0, batch_size=None):
        """Create dataloaders for a specific fold"""
        self.prepare_dataset()

        train_idx = self.train_splits[f"train_{fold_id}"].values
        val_idx = self.val_splits[f"val_{fold_id}"].values
        mean = self.stats[str(fold_id)]["mean"]
        std = self.stats[str(fold_id)]["std"]

        batch_size = batch_size or MODEL_CONFIG.batch_size
        return self.data_manager.create_dataloaders(
            train_idx, val_idx, mean, std, batch_size
        )

    def train_model(self):
        """Train model from scratch using all folds"""
        print(f"Training {self.model_class.__name__} model...")
        self.prepare_dataset()
        print(f"Dataset prepared with {len(self.df)} samples")

        # Create model
        print(f"Creating model instance of {self.model_class.__name__}")
        model = self.get_model()
        print(f"Model created: {model}")

        # Get loaders for first fold (for inspection)
        print("Creating dataloaders for first fold")
        train_loader, val_loader = self.get_loaders()
        print(f"Train loader length: {len(train_loader)}")

        # Analyze model architecture
        model_analyzer = ModelAnalyzer(model, train_loader, val_loader)
        model_analyzer.inspect_model_architecture()

        # Train using cross-validation
        cv_trainer = CrossValidationTrainer(model)
        print("Training all folds...")
        cv_trainer.train_all_folds(self.df, num_folds=10)

        return model

    def show_training_images(self, model_class):
        """Show training images for a specific fold"""
        TrainingMetrics().show_training_results(fold_id=0, model_class=model_class)

    def run_xai_analysis(self, checkpoint_id=0, idx=4981, show_original=True):
        """Run XAI methods on a sample image"""
        self.prepare_dataset()

        # Load model
        model = self.get_model(checkpoint_id=checkpoint_id)

        # Initialize XAI
        xai = XAI(model)

        # Get test image
        image, label, _ = DatasetVisualizer(self.df).get_dataset_image(
            tensor=True, dataset_path=PATH_CONFIG.dataset_path, idx=idx
        )

        # Normalize image
        fold_id = 0  # Using first fold's statistics
        mean = self.stats[str(fold_id)]["mean"]
        std = self.stats[str(fold_id)]["std"]
        val_transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        image = val_transform(image)

        # Run XAI methods
        xai.backpropagation(image, show_original=show_original)
        xai.guided_backpropagation(image, show_original=show_original)

        return image, label

    def show_filters(self, checkpoint_id=9):
        model = self.get_model(checkpoint_id=checkpoint_id)
        xai = XAI(model)
        if self.model_class == ScatNetImageClassifier:
            xai.show_wavelet_filters()
            xai.show_low_pass_filter()
        else:
            xai.show_conv_filters()

    def run_captum_analysis(self, checkpoint_id=0, idx=0, show_original=True):
        """Run model attributions using Captum"""
        self.prepare_dataset()

        # Load model
        model = self.get_model(checkpoint_id=checkpoint_id)

        # Get test image
        image, label, _ = DatasetVisualizer(self.df).get_dataset_image(
            tensor=True, dataset_path=PATH_CONFIG.dataset_path, idx=idx
        )

        # Visualize attributions
        attributions = XAI_CAPTUM(model).visualize_model_attributions(
            image, methods=["Guided Backpropagation"], show_original=show_original
        )

        return image, label, attributions


def main():
    """Main function to execute experiments"""
    # Choose which model to use
    model_class = ScatNetImageClassifier
    # model_class = CNNImageClassifier

    # Create experiment runner
    runner = Runner()
    runner.set_model_class(model_class)

    # Choose which experiment to run
    # runner.train_model()

    # Show filters
    runner.show_filters(checkpoint_id=9)

    tm = TrainingMetrics()
    metrics = tm.compute_metrics_all_folds(runner.data_manager, model_class)
    print(metrics)

    image, label = runner.run_xai_analysis()
    print(f"Analyzed image with label: {label}")

    image, label, _ = runner.run_captum_analysis()
    print(f"Generated Captum attributions for image with label: {label}")


if __name__ == "__main__":
    main()
