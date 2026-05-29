export interface TrainingConfig {
  epochs: number
  batch_size: number
  learning_rate: number
}

export interface ExperimentRequest {
  model: string
  dataset: string
  training: TrainingConfig
}

export interface ExperimentResponse {
  id: number
  model: string
  dataset: string
  epochs: number
  batch_size: number
  learning_rate: number
  train_loss_per_epoch: number[]
  test_loss_per_epoch: number[] | null
  test_loss: number
  test_accuracy: number
  training_time_seconds: number
  confusion_matrix: number[][] | null
  note: string | null
  model_id: string | null
  created_at: string
}

export interface ClassConfidence {
  label: string
  confidence: number
}

export interface ClassifyResponse {
  predicted_class: string
  predicted_index: number
  confidences: ClassConfidence[]
}

export interface GradCamResponse {
  predicted_class: string
  predicted_index: number
  confidences: ClassConfidence[]
  gradcam_image: string
}

export interface ModelResult {
  model: string
  train_loss_per_epoch: number[]
  test_loss_per_epoch: number[] | null
  test_loss: number
  test_accuracy: number
  training_time_seconds: number
  confusion_matrix: number[][] | null
}

export interface CompareResult {
  dataset: string
  epochs: number
  results: ModelResult[]
}

export const MODELS = ['simple_cnn', 'lenet5', 'vgg11', 'resnet18'] as const
export const DATASETS = ['mnist', 'cifar10'] as const

export type ModelName = (typeof MODELS)[number]
export type DatasetName = (typeof DATASETS)[number]

export const MODEL_LABELS: Record<string, string> = {
  simple_cnn: 'Simple CNN',
  lenet5: 'LeNet-5',
  vgg11: 'VGG-11',
  resnet18: 'ResNet-18',
}

export const MODEL_COLORS: Record<string, string> = {
  simple_cnn: '#4f86f7',
  lenet5: '#f59e0b',
  vgg11: '#10b981',
  resnet18: '#f43f5e',
}
