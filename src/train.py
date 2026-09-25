import copy
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from model import MultiDiseaseNN


DATA_PATH = "data/processed/ehr_unscaled_dataset.csv"
OUTPUT_DIR = "data/processed"

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "baseline_model_v2.pth",
)

PREPROCESSOR_PATH = os.path.join(
    OUTPUT_DIR,
    "preprocessing.joblib",
)

METADATA_PATH = os.path.join(
    OUTPUT_DIR,
    "model_metadata.json",
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.csv",
)

SPLIT_PATH = os.path.join(
    OUTPUT_DIR,
    "patient_splits.csv",
)

BINARY_COLS = [
    "gender_encoded",
]

CONTINUOUS_COLS = [
    "age",
    "Potassium",
    "Sodium",
    "Heart_Rate",
    "Respiratory_Rate",
]

FEATURE_COLS = BINARY_COLS + CONTINUOUS_COLS

TARGET_COLS = [
    "Hypertension",
    "Hyperlipidemia",
    "Diabetes",
    "Atrial_Fibrillation",
]

RANDOM_SEED = 42
DECISION_THRESHOLD = 0.5


def set_seed(seed=42):
    """
    Lock random-number generators to make experiments
    as reproducible as reasonably possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


class EHRDataset(Dataset):
    """
    PyTorch dataset containing already transformed
    features and multi-label disease targets.
    """

    def __init__(self, features, targets):
        self.features = torch.tensor(
            features,
            dtype=torch.float32,
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.features[index],
            self.targets[index],
        )


def create_preprocessor():
    """
    Build a preprocessing system that will be fitted
    using training patients only.
    """
    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    continuous_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "binary",
                binary_pipeline,
                BINARY_COLS,
            ),
            (
                "continuous",
                continuous_pipeline,
                CONTINUOUS_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def print_split_summary(name, split_data):
    """
    Display the number of admissions, patients and
    positive disease labels in a data split.
    """
    patient_targets = (
        split_data
        .groupby("subject_id")[TARGET_COLS]
        .max()
    )

    print(f"\n{name} split:")
    print(f"  Admissions: {len(split_data)}")
    print(
        f"  Unique patients: "
        f"{split_data['subject_id'].nunique()}"
    )

    print("  Positive admissions:")
    for disease in TARGET_COLS:
        print(
            f"    {disease}: "
            f"{int(split_data[disease].sum())}"
        )

    print("  Patients with each disease:")
    for disease in TARGET_COLS:
        print(
            f"    {disease}: "
            f"{int(patient_targets[disease].sum())}"
        )


def evaluate_model(
    model,
    dataloader,
    criterion,
    target_cols,
    threshold=0.5,
):
    """
    Evaluate a trained model without updating its weights.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                batch_y.cpu().numpy()
            )

    probabilities = np.vstack(all_probabilities)
    targets = np.vstack(all_targets)

    binary_predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {}

    for index, disease in enumerate(target_cols):
        disease_targets = targets[:, index]
        disease_probabilities = probabilities[:, index]
        disease_predictions = binary_predictions[:, index]

        unique_classes = np.unique(disease_targets)

        if len(unique_classes) > 1:
            roc_auc = roc_auc_score(
                disease_targets,
                disease_probabilities,
            )
        else:
            roc_auc = float("nan")

        if disease_targets.sum() > 0:
            pr_auc = average_precision_score(
                disease_targets,
                disease_probabilities,
            )
        else:
            pr_auc = float("nan")

        metrics[disease] = {
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Precision": precision_score(
                disease_targets,
                disease_predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                disease_targets,
                disease_predictions,
                zero_division=0,
            ),
            "F1-Score": f1_score(
                disease_targets,
                disease_predictions,
                zero_division=0,
            ),
            "Positive Cases": int(
                disease_targets.sum()
            ),
            "Total Cases": int(
                len(disease_targets)
            ),
        }

    average_loss = total_loss / total_samples

    return average_loss, metrics


def train_model():
    set_seed(RANDOM_SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("--- Step 1: Loading Unscaled Dataset ---")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} was not found. "
            "Run python src\\data_pipeline.py first."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = (
        ["subject_id", "hadm_id"]
        + FEATURE_COLS
        + TARGET_COLS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_columns}"
        )

    print(f"Admissions loaded: {len(data)}")
    print(
        f"Unique patients loaded: "
        f"{data['subject_id'].nunique()}"
    )

    print(
        "\n--- Step 2: Patient-Level "
        "Train/Validation/Test Split ---"
    )

    unique_subjects = (
        data["subject_id"]
        .drop_duplicates()
        .to_numpy()
    )

    train_subjects, temporary_subjects = train_test_split(
        unique_subjects,
        test_size=0.30,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    validation_subjects, test_subjects = train_test_split(
        temporary_subjects,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_data = data[
        data["subject_id"].isin(train_subjects)
    ].copy()

    validation_data = data[
        data["subject_id"].isin(validation_subjects)
    ].copy()

    test_data = data[
        data["subject_id"].isin(test_subjects)
    ].copy()

    assert set(train_subjects).isdisjoint(
        set(validation_subjects)
    )

    assert set(train_subjects).isdisjoint(
        set(test_subjects)
    )

    assert set(validation_subjects).isdisjoint(
        set(test_subjects)
    )

    print_split_summary(
        "Training",
        train_data,
    )

    print_split_summary(
        "Validation",
        validation_data,
    )

    print_split_summary(
        "Test",
        test_data,
    )

    split_records = []

    for subject_id in train_subjects:
        split_records.append(
            {
                "subject_id": subject_id,
                "split": "train",
            }
        )

    for subject_id in validation_subjects:
        split_records.append(
            {
                "subject_id": subject_id,
                "split": "validation",
            }
        )

    for subject_id in test_subjects:
        split_records.append(
            {
                "subject_id": subject_id,
                "split": "test",
            }
        )

    pd.DataFrame(split_records).to_csv(
        SPLIT_PATH,
        index=False,
    )

    print(
        "\n--- Step 3: Leakage-Safe "
        "Imputation and Scaling ---"
    )

    preprocessor = create_preprocessor()

    # Fit using training patients only.
    x_train = preprocessor.fit_transform(
        train_data[FEATURE_COLS]
    )

    # Validation and test patients cannot influence
    # the fitted imputation or scaling values.
    x_validation = preprocessor.transform(
        validation_data[FEATURE_COLS]
    )

    x_test = preprocessor.transform(
        test_data[FEATURE_COLS]
    )

    y_train = train_data[
        TARGET_COLS
    ].to_numpy(dtype=np.float32)

    y_validation = validation_data[
        TARGET_COLS
    ].to_numpy(dtype=np.float32)

    y_test = test_data[
        TARGET_COLS
    ].to_numpy(dtype=np.float32)

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    print(
        "Saved fitted preprocessing system to: "
        f"{PREPROCESSOR_PATH}"
    )

    print(
        f"Transformed feature count: "
        f"{x_train.shape[1]}"
    )

    train_dataset = EHRDataset(
        x_train,
        y_train,
    )

    validation_dataset = EHRDataset(
        x_validation,
        y_validation,
    )

    test_dataset = EHRDataset(
        x_test,
        y_test,
    )

    data_generator = torch.Generator()
    data_generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        generator=data_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    print(
        "\n--- Step 4: Model and "
        "Class-Imbalance Setup ---"
    )

    input_dimension = x_train.shape[1]

    model = MultiDiseaseNN(
        input_dim=input_dimension,
        output_dim=len(TARGET_COLS),
    )

    positive_counts = y_train.sum(axis=0)
    negative_counts = len(y_train) - positive_counts

    if np.any(positive_counts == 0):
        diseases_without_positives = [
            TARGET_COLS[index]
            for index, count in enumerate(
                positive_counts
            )
            if count == 0
        ]

        raise ValueError(
            "Training split has no positive cases for: "
            f"{diseases_without_positives}"
        )

    positive_weights = torch.tensor(
        negative_counts / positive_counts,
        dtype=torch.float32,
    )

    print(
        "Positive class weights:",
        positive_weights.numpy(),
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("\n--- Step 5: Training with Early Stopping ---")

    maximum_epochs = 100
    patience = 10
    minimum_improvement = 0.0001

    best_validation_loss = float("inf")
    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    best_epoch = 0
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(maximum_epochs):
        model.train()

        total_training_loss = 0.0
        total_training_samples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)

            total_training_loss += (
                loss.item() * batch_size
            )

            total_training_samples += batch_size

        average_training_loss = (
            total_training_loss
            / total_training_samples
        )

        validation_loss, _ = evaluate_model(
            model,
            validation_loader,
            criterion,
            TARGET_COLS,
            DECISION_THRESHOLD,
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "training_loss": average_training_loss,
                "validation_loss": validation_loss,
            }
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: "
            f"{average_training_loss:.4f} | "
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
            - minimum_improvement
        ):
            best_validation_loss = validation_loss
            best_epoch = epoch + 1

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                "\nEarly stopping activated at "
                f"epoch {epoch + 1}."
            )
            break

    pd.DataFrame(training_history).to_csv(
        HISTORY_PATH,
        index=False,
    )

    model.load_state_dict(
        best_model_weights
    )

    checkpoint = {
        "model_state_dict": best_model_weights,
        "input_dim": input_dimension,
        "output_dim": len(TARGET_COLS),
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "decision_threshold": DECISION_THRESHOLD,
        "best_epoch": best_epoch,
        "random_seed": RANDOM_SEED,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Saved Version 2 model to: {MODEL_PATH}")

    print(
        "\n--- Step 6: Final Evaluation "
        "on Unseen Test Patients ---"
    )

    test_loss, test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        TARGET_COLS,
        DECISION_THRESHOLD,
    )

    print(f"Test loss: {test_loss:.4f}")

    for disease, metrics in test_metrics.items():
        print(f"\n--- {disease} ---")

        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(
                    f"  {metric_name}: "
                    f"{value:.4f}"
                )
            else:
                print(
                    f"  {metric_name}: "
                    f"{value}"
                )

    metadata = {
        "dataset": DATA_PATH,
        "total_admissions": int(len(data)),
        "total_unique_patients": int(
            data["subject_id"].nunique()
        ),
        "training_patients": int(
            train_data["subject_id"].nunique()
        ),
        "validation_patients": int(
            validation_data["subject_id"].nunique()
        ),
        "test_patients": int(
            test_data["subject_id"].nunique()
        ),
        "raw_feature_columns": FEATURE_COLS,
        "transformed_feature_order": (
            BINARY_COLS + CONTINUOUS_COLS
        ),
        "target_columns": TARGET_COLS,
        "decision_threshold": DECISION_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "research_warning": (
            "Research prototype only. "
            "Not validated for clinical use."
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    print(f"\nSaved metadata to: {METADATA_PATH}")
    print(f"Saved training history to: {HISTORY_PATH}")
    print(f"Saved patient splits to: {SPLIT_PATH}")


if __name__ == "__main__":
    train_model()