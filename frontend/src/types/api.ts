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
  sample_gradcams: SampleGradCam[] | null
}

export interface SampleGradCam {
  true_label: string
  predicted_label: string
  confidence: number
  gradcam_image: string
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

export const MODELS = ['simple_cnn', 'lenet5', 'alexnet', 'vgg11', 'resnet18', 'mobilenet'] as const
export const DATASETS = ['mnist', 'fashion_mnist', 'cifar10'] as const

export type ModelName = (typeof MODELS)[number]
export type DatasetName = (typeof DATASETS)[number]

export const MODEL_LABELS: Record<string, string> = {
  simple_cnn: 'Simple CNN',
  lenet5: 'LeNet-5',
  alexnet: 'AlexNet',
  vgg11: 'VGG-11',
  resnet18: 'ResNet-18',
  mobilenet: 'MobileNet V1',
}

export const DATASET_LABELS: Record<string, string> = {
  mnist: 'MNIST',
  fashion_mnist: 'Fashion-MNIST',
  cifar10: 'CIFAR-10',
}

export const MODEL_COLORS: Record<string, string> = {
  simple_cnn: '#4f86f7',
  lenet5: '#f59e0b',
  alexnet: '#8b5cf6',
  vgg11: '#10b981',
  resnet18: '#f43f5e',
  mobilenet: '#06b6d4',
}
