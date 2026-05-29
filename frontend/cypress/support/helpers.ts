// JWT with payload {"sub":"testuser"} — decoded by decodeJwtSub in App.tsx
export const TEST_JWT =
  'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0dXNlciJ9.test-signature'

export const MOCK_EXPERIMENT = {
  id: 1,
  model: 'lenet5',
  dataset: 'mnist',
  epochs: 5,
  batch_size: 32,
  learning_rate: 0.001,
  train_loss_per_epoch: [0.8, 0.5, 0.35, 0.25, 0.2],
  test_loss_per_epoch: [0.7, 0.48, 0.38, 0.28, 0.22],
  test_loss: 0.22,
  test_accuracy: 0.95,
  training_time_seconds: 12.5,
  confusion_matrix: null,
  note: null,
  created_at: '2024-01-15T10:00:00',
}

export const MOCK_EXPERIMENT_2 = {
  id: 2,
  model: 'resnet18',
  dataset: 'cifar10',
  epochs: 3,
  batch_size: 64,
  learning_rate: 0.01,
  train_loss_per_epoch: [1.2, 0.9, 0.7],
  test_loss_per_epoch: [1.1, 0.85, 0.65],
  test_loss: 0.65,
  test_accuracy: 0.82,
  training_time_seconds: 25.0,
  confusion_matrix: null,
  note: 'baseline',
  created_at: '2024-01-16T12:00:00',
}
