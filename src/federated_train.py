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

from torch.utils.data import DataLoader, Dataset

from model import MultiDiseaseNN


DATA_PATH = "data/processed/ehr_unscaled_dataset.csv"
SPLIT_PATH = "data/processed/patient_splits.csv"
PREPROCESSOR_PATH = "data/processed/preprocessing.joblib"

FEDERATED_OUTPUT_DIR = "data/processed/federated"
LOCAL_MODEL_DIR = os.path.join(
    FEDERATED_OUTPUT_DIR,
    "local_models",
)

CLIENT_ASSIGNMENT_PATH = os.path.join(
    FEDERATED_OUTPUT_DIR,
    "client_assignments.csv",
)

CLIENT_SUMMARY_PATH = os.path.join(
    FEDERATED_OUTPUT_DIR,
    "local_training_summary.json",
)

INITIAL_MODEL_PATH = os.path.join(
    FEDERATED_OUTPUT_DIR,
    "initial_global_model.pth",
)

FEATURE_COLS = [
    "gender_encoded",
    "age",
    "Potassium",
    "Sodium",
    "Heart_Rate",
    "Respiratory_Rate",
]

TARGET_COLS = [
    "Hypertension",
    "Hyperlipidemia",
    "Diabetes",
    "Atrial_Fibrillation",
]

NUMBER_OF_CLIENTS = 3
LOCAL_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def set_seed(seed=42):
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


class LocalEHRDataset(Dataset):
    """
    Dataset held by one simulated healthcare institution.
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


def create_client_assignments(
    training_subject_ids,
    number_of_clients,
    seed,
):
    """
    Divide unique training patients between simulated
    hospitals. One patient can belong to only one hospital.
    """
    shuffled_subjects = np.array(
        training_subject_ids,
        copy=True,
    )

    random_generator = np.random.default_rng(seed)
    random_generator.shuffle(shuffled_subjects)

    subject_groups = np.array_split(
        shuffled_subjects,
        number_of_clients,
    )

    client_assignments = {}

    for client_index, subject_group in enumerate(
        subject_groups,
        start=1,
    ):
        client_name = f"hospital_{client_index}"

        client_assignments[client_name] = (
            subject_group.tolist()
        )

    return client_assignments


def verify_client_separation(client_assignments):
    """
    Confirm that no patient was assigned to two hospitals.
    """
    all_assigned_subjects = []

    for subject_ids in client_assignments.values():
        all_assigned_subjects.extend(subject_ids)

    if len(all_assigned_subjects) != len(
        set(all_assigned_subjects)
    ):
        raise ValueError(
            "Patient overlap detected between hospitals."
        )

    print(
        "Verified: every patient belongs to "
        "exactly one simulated hospital."
    )


def calculate_local_positive_weights(targets):
    """
    Calculate class weights using only the client's
    local disease distribution.
    """
    positive_counts = targets.sum(axis=0)
    negative_counts = len(targets) - positive_counts

    weights = np.ones(
        len(TARGET_COLS),
        dtype=np.float32,
    )

    for index in range(len(TARGET_COLS)):
        positives = positive_counts[index]
        negatives = negative_counts[index]

        if positives > 0 and negatives > 0:
            weights[index] = negatives / positives

        else:
            # A client cannot learn a disease class if it
            # contains only positive or only negative cases.
            # Use a neutral weight and report the limitation.
            weights[index] = 1.0

            print(
                f"  Warning: {TARGET_COLS[index]} has "
                f"positives={int(positives)} and "
                f"negatives={int(negatives)}."
            )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def train_local_model(
    client_name,
    client_data,
    initial_model_weights,
    preprocessor,
    client_seed,
):
    """
    Train one hospital's model using only that
    hospital's patient records.
    """
    print(f"\n--- Training {client_name} ---")

    patient_count = (
        client_data["subject_id"].nunique()
    )

    admission_count = len(client_data)

    print(f"Local patients: {patient_count}")
    print(f"Local admissions: {admission_count}")

    print("Local disease counts:")

    for disease in TARGET_COLS:
        print(
            f"  {disease}: "
            f"{int(client_data[disease].sum())}"
        )

    # The shared preprocessor transforms the local records.
    # The raw local rows remain inside this function.
    local_features = preprocessor.transform(
        client_data[FEATURE_COLS]
    )

    local_targets = client_data[
        TARGET_COLS
    ].to_numpy(dtype=np.float32)

    local_dataset = LocalEHRDataset(
        local_features,
        local_targets,
    )

    local_generator = torch.Generator()
    local_generator.manual_seed(client_seed)

    local_loader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=local_generator,
    )

    local_model = MultiDiseaseNN(
        input_dim=local_features.shape[1],
        output_dim=len(TARGET_COLS),
    )

    # Every hospital begins with identical model weights.
    local_model.load_state_dict(
        copy.deepcopy(initial_model_weights)
    )

    local_positive_weights = (
        calculate_local_positive_weights(
            local_targets
        )
    )

    print(
        "Local positive weights:",
        local_positive_weights.numpy(),
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=local_positive_weights
    )

    optimizer = optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE,
    )

    epoch_losses = []

    for epoch in range(LOCAL_EPOCHS):
        local_model.train()

        total_loss = 0.0
        total_samples = 0

        for batch_features, batch_targets in local_loader:
            optimizer.zero_grad()

            logits = local_model(
                batch_features
            )

            loss = criterion(
                logits,
                batch_targets,
            )

            loss.backward()
            optimizer.step()

            current_batch_size = (
                batch_features.size(0)
            )

            total_loss += (
                loss.item()
                * current_batch_size
            )

            total_samples += current_batch_size

        average_loss = total_loss / total_samples
        epoch_losses.append(float(average_loss))

        print(
            f"  Epoch {epoch + 1:02d}/"
            f"{LOCAL_EPOCHS} | "
            f"Local Loss: {average_loss:.4f}"
        )

    local_model_path = os.path.join(
        LOCAL_MODEL_DIR,
        f"{client_name}_model.pth",
    )

    local_checkpoint = {
        "client_name": client_name,
        "model_state_dict": (
            local_model.state_dict()
        ),
        "patient_count": patient_count,
        "admission_count": admission_count,
        "feature_columns": FEATURE_COLS,
        "target_columns": TARGET_COLS,
        "local_epochs": LOCAL_EPOCHS,
        "final_local_loss": epoch_losses[-1],
    }

    torch.save(
        local_checkpoint,
        local_model_path,
    )

    print(
        f"Saved local model weights to: "
        f"{local_model_path}"
    )

    return {
        "client_name": client_name,
        "patient_count": int(patient_count),
        "admission_count": int(admission_count),
        "disease_counts": {
            disease: int(
                client_data[disease].sum()
            )
            for disease in TARGET_COLS
        },
        "epoch_losses": epoch_losses,
        "final_local_loss": epoch_losses[-1],
        "model_path": local_model_path,
    }


def run_local_training():
    set_seed(RANDOM_SEED)

    print(
        "--- Objective 3: Simulating "
        "Healthcare Institutions ---"
    )

    required_files = [
        DATA_PATH,
        SPLIT_PATH,
        PREPROCESSOR_PATH,
    ]

    for required_file in required_files:
        if not os.path.exists(required_file):
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_file}"
            )

    data = pd.read_csv(DATA_PATH)
    patient_splits = pd.read_csv(SPLIT_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    training_subject_ids = (
        patient_splits[
            patient_splits["split"] == "train"
        ]["subject_id"]
        .tolist()
    )

    training_data = data[
        data["subject_id"].isin(
            training_subject_ids
        )
    ].copy()

    validation_subjects = set(
        patient_splits[
            patient_splits["split"] == "validation"
        ]["subject_id"]
    )

    test_subjects = set(
        patient_splits[
            patient_splits["split"] == "test"
        ]["subject_id"]
    )

    local_training_subjects = set(
        training_data["subject_id"]
    )

    if not local_training_subjects.isdisjoint(
        validation_subjects
    ):
        raise ValueError(
            "Validation patient found in local training."
        )

    if not local_training_subjects.isdisjoint(
        test_subjects
    ):
        raise ValueError(
            "Test patient found in local training."
        )

    print(
        f"Available training patients: "
        f"{training_data['subject_id'].nunique()}"
    )

    print(
        f"Available training admissions: "
        f"{len(training_data)}"
    )

    client_assignments = create_client_assignments(
        training_subject_ids,
        NUMBER_OF_CLIENTS,
        RANDOM_SEED,
    )

    verify_client_separation(
        client_assignments
    )

    os.makedirs(
        LOCAL_MODEL_DIR,
        exist_ok=True,
    )

    assignment_records = []

    for client_name, subject_ids in (
        client_assignments.items()
    ):
        print(
            f"{client_name}: "
            f"{len(subject_ids)} patients"
        )

        for subject_id in subject_ids:
            assignment_records.append(
                {
                    "subject_id": subject_id,
                    "client": client_name,
                }
            )

    pd.DataFrame(assignment_records).to_csv(
        CLIENT_ASSIGNMENT_PATH,
        index=False,
    )

    print(
        f"Saved client assignments to: "
        f"{CLIENT_ASSIGNMENT_PATH}"
    )

    print(
        "\n--- Objective 4: "
        "Local Model Training ---"
    )

    # This is the initial global model.
    # Every client receives an identical copy.
    initial_global_model = MultiDiseaseNN(
        input_dim=len(FEATURE_COLS),
        output_dim=len(TARGET_COLS),
    )

    initial_model_weights = copy.deepcopy(
        initial_global_model.state_dict()
    )

    torch.save(
        {
            "model_state_dict": (
                initial_model_weights
            ),
            "input_dim": len(FEATURE_COLS),
            "output_dim": len(TARGET_COLS),
            "random_seed": RANDOM_SEED,
        },
        INITIAL_MODEL_PATH,
    )

    print(
        f"Saved identical initial model to: "
        f"{INITIAL_MODEL_PATH}"
    )

    training_summaries = []

    for client_number, (
        client_name,
        subject_ids,
    ) in enumerate(
        client_assignments.items(),
        start=1,
    ):
        client_data = training_data[
            training_data["subject_id"].isin(
                subject_ids
            )
        ].copy()

        client_summary = train_local_model(
            client_name=client_name,
            client_data=client_data,
            initial_model_weights=(
                initial_model_weights
            ),
            preprocessor=preprocessor,
            client_seed=(
                RANDOM_SEED + client_number
            ),
        )

        training_summaries.append(
            client_summary
        )

    output_summary = {
        "number_of_clients": NUMBER_OF_CLIENTS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "raw_patient_records_shared": False,
        "validation_patients_used": False,
        "test_patients_used": False,
        "clients": training_summaries,
        "note": (
            "This is a simulated federated setup. "
            "Local model weights have not yet been "
            "aggregated."
        ),
    }

    with open(
        CLIENT_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            output_summary,
            summary_file,
            indent=4,
        )

    print(
        f"\nSaved local training summary to: "
        f"{CLIENT_SUMMARY_PATH}"
    )

    print(
        "\nObjectives 3 and 4 completed:"
    )

    print(
        "  1. Training patients were divided "
        "between simulated hospitals."
    )

    print(
        "  2. No validation or test patients "
        "were used for local training."
    )

    print(
        "  3. Each hospital trained an "
        "independent local model."
    )

    print(
        "  4. Only model checkpoints were saved."
    )

    print(
        "  5. Model aggregation has not yet "
        "been performed."
    )


if __name__ == "__main__":
    run_local_training()