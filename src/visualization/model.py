from torchview import draw_graph
from torchvision import transforms

from src.config import MODEL_CONFIG, PATH_CONFIG, device
from src.dataset import DataManager
from src.models.cnn import CNNImageClassifier
from src.models.scatnet import ScatNetImageClassifier
from src.models.scatnet_optimizer import ScatNetOptimizer
from src.models.utils import ModelAnalyzer
from src.training.metrics import TrainingMetrics
from src.training.training import CrossValidationTrainer
from src.visualization.dataset_vis import DatasetVisualizer
from src.visualization.xai import XAI
from src.visualization.xai_captum import XAI_CAPTUM

model = CNNImageClassifier()
batch_size = 2
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(
    model, input_size=(batch_size, MODEL_CONFIG.target_image_size), device="meta"
)
model_graph.visual_graph
