from pydantic import BaseModel
from typing import Literal, List


class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001


class ExperimentConfig(BaseModel):
    model: Literal["simple_cnn", "lenet5", "vgg11", "resnet18"]
    dataset: Literal["mnist", "cifar10"]
    training: TrainingConfig

class CompareConfig(BaseModel):
    dataset: Literal["mnist", "cifar10"]
    training: TrainingConfig

class ClassConfidence(BaseModel):
    label: str
    confidence: float

class PredictResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidences: List[ClassConfidence]

class GradCamResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidences: List[ClassConfidence]
    gradcam_image: str

