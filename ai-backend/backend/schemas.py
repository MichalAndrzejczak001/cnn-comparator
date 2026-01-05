from pydantic import BaseModel
from typing import Literal

class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001

class ExperimentConfig(BaseModel):
    model: Literal["simple_cnn"]
    dataset: Literal["mnist"]
    training: TrainingConfig

