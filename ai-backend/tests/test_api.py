from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

MOCK_LOADERS = (MagicMock(), MagicMock(), 10, 1, (32, 32))
MOCK_TRAIN_RESULT = ([0.5, 0.3], [0.45, 0.28], 1.5)
MOCK_EVAL_RESULT = {
    "loss": 0.28,
    "accuracy": 0.92,
    "confusion_matrix": [[9, 1], [0, 10]],
}

EXPERIMENT_PAYLOAD = {
    "model": "lenet5",
    "dataset": "mnist",
    "training": {"epochs": 2, "batch_size": 32, "learning_rate": 0.001},
}

COMPARE_PAYLOAD = {
    "dataset": "mnist",
    "training": {"epochs": 2, "batch_size": 32, "learning_rate": 0.001},
}


@pytest.fixture
def mock_deps():
    """Patches load_dataset, train and evaluate so no I/O or real training happens."""
    with (
        patch("backend.main.load_dataset") as mock_load,
        patch("backend.main.train") as mock_train,
        patch("backend.main.evaluate") as mock_eval,
    ):
        mock_load.return_value = MOCK_LOADERS
        mock_train.return_value = MOCK_TRAIN_RESULT
        mock_eval.return_value = MOCK_EVAL_RESULT
        yield mock_load, mock_train, mock_eval


# --- GET /health ---

class TestHealthEndpoint:

    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# --- POST /experiments ---

class TestExperimentsEndpoint:

    def test_valid_request_returns_200(self, mock_deps):
        response = client.post("/experiments", json=EXPERIMENT_PAYLOAD)
        assert response.status_code == 200

    def test_response_contains_required_fields(self, mock_deps):
        response = client.post("/experiments", json=EXPERIMENT_PAYLOAD)
        body = response.json()
        assert "status" in body
        assert "train_loss_per_epoch" in body
        assert "test_loss_per_epoch" in body
        assert "test_loss" in body
        assert "test_accuracy" in body
        assert "confusion_matrix" in body
        assert "training_time_seconds" in body

    def test_train_loss_per_epoch_length_matches_epochs(self, mock_deps):
        mock_load, mock_train, mock_eval = mock_deps
        epochs = 2
        mock_train.return_value = ([0.5, 0.3], [0.45, 0.28], 1.5)

        response = client.post("/experiments", json={**EXPERIMENT_PAYLOAD, "training": {"epochs": epochs, "batch_size": 32, "learning_rate": 0.001}})
        body = response.json()

        assert len(body["train_loss_per_epoch"]) == epochs

    def test_test_loss_per_epoch_length_matches_epochs(self, mock_deps):
        mock_load, mock_train, mock_eval = mock_deps
        epochs = 2
        mock_train.return_value = ([0.5, 0.3], [0.45, 0.28], 1.5)

        response = client.post("/experiments", json={**EXPERIMENT_PAYLOAD, "training": {"epochs": epochs, "batch_size": 32, "learning_rate": 0.001}})
        body = response.json()

        assert len(body["test_loss_per_epoch"]) == epochs

    def test_load_dataset_called_with_dataset_from_request(self, mock_deps):
        mock_load, _, _ = mock_deps
        client.post("/experiments", json=EXPERIMENT_PAYLOAD)
        mock_load.assert_called_once_with("mnist", 32)

    def test_train_called_with_epochs_from_request(self, mock_deps):
        _, mock_train, _ = mock_deps
        client.post("/experiments", json=EXPERIMENT_PAYLOAD)
        call_args = mock_train.call_args
        assert call_args.args[3] == 2  # epochs is 4th positional argument

    def test_invalid_model_name_returns_422(self):
        payload = {**EXPERIMENT_PAYLOAD, "model": "nonexistent_cnn"}
        response = client.post("/experiments", json=payload)
        assert response.status_code == 422

    def test_invalid_dataset_name_returns_422(self):
        payload = {**EXPERIMENT_PAYLOAD, "dataset": "imagenet"}
        response = client.post("/experiments", json=payload)
        assert response.status_code == 422

    def test_missing_model_field_returns_422(self):
        payload = {k: v for k, v in EXPERIMENT_PAYLOAD.items() if k != "model"}
        response = client.post("/experiments", json=payload)
        assert response.status_code == 422

    def test_missing_dataset_field_returns_422(self):
        payload = {k: v for k, v in EXPERIMENT_PAYLOAD.items() if k != "dataset"}
        response = client.post("/experiments", json=payload)
        assert response.status_code == 422

    def test_all_four_models_accepted(self, mock_deps):
        for model_name in ["simple_cnn", "lenet5", "vgg11", "resnet18"]:
            payload = {**EXPERIMENT_PAYLOAD, "model": model_name}
            response = client.post("/experiments", json=payload)
            assert response.status_code == 200, f"Failed for model: {model_name}"

    def test_both_datasets_accepted(self, mock_deps):
        for dataset in ["mnist", "cifar10"]:
            payload = {**EXPERIMENT_PAYLOAD, "dataset": dataset}
            response = client.post("/experiments", json=payload)
            assert response.status_code == 200, f"Failed for dataset: {dataset}"


# --- POST /compare ---

class TestCompareEndpoint:

    def test_valid_request_returns_200(self, mock_deps):
        response = client.post("/compare", json=COMPARE_PAYLOAD)
        assert response.status_code == 200

    def test_response_contains_4_results(self, mock_deps):
        response = client.post("/compare", json=COMPARE_PAYLOAD)
        assert len(response.json()["results"]) == 4

    def test_response_includes_dataset_field(self, mock_deps):
        response = client.post("/compare", json=COMPARE_PAYLOAD)
        assert response.json()["dataset"] == "mnist"

    def test_response_includes_epochs_field(self, mock_deps):
        response = client.post("/compare", json=COMPARE_PAYLOAD)
        assert response.json()["epochs"] == 2

    def test_all_four_model_names_present_in_results(self, mock_deps):
        response = client.post("/compare", json=COMPARE_PAYLOAD)
        returned_models = {r["model"] for r in response.json()["results"]}
        assert returned_models == {"simple_cnn", "lenet5", "vgg11", "resnet18"}

    def test_each_result_has_required_fields(self, mock_deps):
        response = client.post("/compare", json=COMPARE_PAYLOAD)
        required = {"model", "train_loss_per_epoch", "test_loss_per_epoch",
                    "test_loss", "test_accuracy", "training_time_seconds", "confusion_matrix"}
        for result in response.json()["results"]:
            assert required.issubset(result.keys())

    def test_invalid_dataset_returns_422(self):
        payload = {**COMPARE_PAYLOAD, "dataset": "imagenet"}
        response = client.post("/compare", json=payload)
        assert response.status_code == 422

    def test_missing_dataset_field_returns_422(self):
        payload = {k: v for k, v in COMPARE_PAYLOAD.items() if k != "dataset"}
        response = client.post("/compare", json=payload)
        assert response.status_code == 422

    def test_cifar10_dataset_accepted(self, mock_deps):
        payload = {**COMPARE_PAYLOAD, "dataset": "cifar10"}
        response = client.post("/compare", json=payload)
        assert response.status_code == 200
