import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from backend.training.trainer import train, evaluate


class FixedPredictor(nn.Module):
    """Returns a fixed logit of 100 for one class regardless of input."""

    def __init__(self, num_classes: int, predicted_class: int):
        super().__init__()
        self.num_classes = num_classes
        self.predicted_class = predicted_class

    def forward(self, x):
        batch = x.size(0)
        out = torch.zeros(batch, self.num_classes)
        out[:, self.predicted_class] = 100.0
        return out


class TinyMLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(32 * 32, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def make_loader(num_classes: int = 10, n_samples: int = 20, batch_size: int = 4, in_channels: int = 1):
    X = torch.randn(n_samples, in_channels, 32, 32)
    y = torch.randint(0, num_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


class TestEvaluate:

    def test_returns_required_keys(self):
        model = TinyMLP()
        loader = make_loader()
        result = evaluate(model, loader, device="cpu")
        assert "loss" in result
        assert "accuracy" in result
        assert "confusion_matrix" in result

    def test_perfect_accuracy_when_all_predictions_match_labels(self):
        model = FixedPredictor(num_classes=3, predicted_class=0)
        X = torch.randn(12, 1, 32, 32)
        y = torch.zeros(12, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=4)

        result = evaluate(model, loader, device="cpu")

        assert result["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy_when_no_predictions_match_labels(self):
        model = FixedPredictor(num_classes=3, predicted_class=0)
        X = torch.randn(12, 1, 32, 32)
        y = torch.ones(12, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=4)

        result = evaluate(model, loader, device="cpu")

        assert result["accuracy"] == pytest.approx(0.0)

    def test_accuracy_between_0_and_1(self):
        model = TinyMLP()
        loader = make_loader(n_samples=20)
        result = evaluate(model, loader, device="cpu")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_loss_is_non_negative(self):
        model = TinyMLP()
        loader = make_loader()
        result = evaluate(model, loader, device="cpu")
        assert result["loss"] >= 0.0

    def test_confusion_matrix_is_2d_list(self):
        model = TinyMLP()
        loader = make_loader()
        result = evaluate(model, loader, device="cpu")
        matrix = result["confusion_matrix"]
        assert isinstance(matrix, list)
        assert all(isinstance(row, list) for row in matrix)

    def test_confusion_matrix_size_covers_all_label_classes(self):
        # Labels span 4 classes (0–3), model always predicts class 0.
        # evaluate() sets num_classes = max(max_labels=3, max_preds=0) + 1 = 4.
        model = FixedPredictor(num_classes=4, predicted_class=0)
        y = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
        X = torch.randn(16, 1, 32, 32)
        loader = DataLoader(TensorDataset(X, y), batch_size=4)

        result = evaluate(model, loader, device="cpu")
        matrix = result["confusion_matrix"]

        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)

    def test_confusion_matrix_counts_sum_to_total_samples(self):
        model = TinyMLP()
        n_samples = 20
        loader = make_loader(n_samples=n_samples)
        result = evaluate(model, loader, device="cpu")
        matrix = result["confusion_matrix"]
        total = sum(cell for row in matrix for cell in row)
        assert total == n_samples

    def test_confusion_matrix_diagonal_equals_correct_predictions(self):
        # Model always predicts class 0, all labels are 0 → perfect for class 0.
        # matrix is 1×1 with value 12.
        model = FixedPredictor(num_classes=1, predicted_class=0)
        X = torch.randn(12, 1, 32, 32)
        y = torch.zeros(12, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=4)

        result = evaluate(model, loader, device="cpu")
        matrix = result["confusion_matrix"]

        assert matrix[0][0] == 12


class TestTrain:

    def test_train_history_length_equals_epochs(self):
        model = TinyMLP()
        loader = make_loader(n_samples=16, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_history, _, _ = train(model, loader, loader, epochs=3, optimizer=optimizer, device="cpu")

        assert len(train_history) == 3

    def test_test_history_length_equals_epochs(self):
        model = TinyMLP()
        loader = make_loader(n_samples=16, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        _, test_history, _ = train(model, loader, loader, epochs=3, optimizer=optimizer, device="cpu")

        assert len(test_history) == 3

    def test_training_time_is_positive(self):
        model = TinyMLP()
        loader = make_loader(n_samples=16, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        _, _, training_time = train(model, loader, loader, epochs=2, optimizer=optimizer, device="cpu")

        assert training_time >= 0

    def test_train_loss_values_are_positive(self):
        model = TinyMLP()
        loader = make_loader(n_samples=16, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_history, _, _ = train(model, loader, loader, epochs=2, optimizer=optimizer, device="cpu")

        assert all(loss > 0 for loss in train_history)

    def test_test_loss_values_are_positive(self):
        model = TinyMLP()
        loader = make_loader(n_samples=16, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        _, test_history, _ = train(model, loader, loader, epochs=2, optimizer=optimizer, device="cpu")

        assert all(loss > 0 for loss in test_history)

    def test_loss_values_are_floats(self):
        model = TinyMLP()
        loader = make_loader(n_samples=16, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_history, test_history, _ = train(model, loader, loader, epochs=2, optimizer=optimizer, device="cpu")

        assert all(isinstance(v, float) for v in train_history)
        assert all(isinstance(v, float) for v in test_history)
