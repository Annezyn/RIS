"""RIS codebook-selection MLP training and inference.

The model maps a 1024-dimensional sampled CSI feature vector to 1024
normalized achievable-rate scores. Inference selects the codebook with
``argmax`` over the predicted scores. This is a regression task.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT_PATH = SCRIPT_DIR / "ris_mlp_weight.pth"

DATA_FILENAMES = (
    "X.npy",
    "Y.npy",
    "train_indices.npy",
    "validation_indices.npy",
)

INPUT_DIM = 1024
OUTPUT_DIM = 1024

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.1
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42


def resolve_data_dir(explicit_dir: str | None = None) -> Path:
    """Find the directory containing all four dataset files.

    The explicit path has highest priority. Otherwise, check the current
    working directory, this script's directory, and its parent directory.
    This makes the script work both inside the repository and from Jupyter.
    """
    if explicit_dir is not None:
        candidates = [Path(explicit_dir).expanduser().resolve()]
    else:
        candidates = [
            Path.cwd(),
            SCRIPT_DIR,
            SCRIPT_DIR.parent,
        ]

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if all((candidate / name).is_file() for name in DATA_FILENAMES):
            return candidate

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find X.npy, Y.npy, train_indices.npy, and "
        f"validation_indices.npy. Checked:\n{checked_text}\n"
        "Pass the correct directory with --data-dir."
    )


def set_random_seed(seed: int) -> None:
    """Seed libraries used by training and DataLoader shuffling only."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Use CUDA when available; otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RISRegressionDataset(Dataset):
    """Map provided split indices to on-demand samples from memory maps."""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        if x_data.ndim != 2 or y_data.ndim != 2:
            raise ValueError("X.npy and Y.npy must both be two-dimensional.")
        if x_data.shape != y_data.shape:
            raise ValueError(
                f"X and Y shapes differ: {x_data.shape} vs {y_data.shape}."
            )

        self.x_data = x_data
        self.y_data = y_data
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)

        if self.indices.size == 0:
            raise ValueError("The provided index array is empty.")
        if self.indices.min() < 0 or self.indices.max() >= x_data.shape[0]:
            raise IndexError("A split index is outside the X/Y sample range.")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_index = int(self.indices[item])

        # Copy only the current row, not the complete dataset.
        x_sample = np.array(
            self.x_data[sample_index],
            dtype=np.float32,
            copy=True,
        )
        y_sample = np.array(
            self.y_data[sample_index],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(x_sample), torch.from_numpy(y_sample)


class RISCodebookMLP(nn.Module):
    """Lightweight fully connected regressor: 1024-512-256-512-1024."""

    def __init__(self, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # No activation: the output is a continuous 1024-score vector.
            nn.Linear(512, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Build loaders using exactly the provided train/validation indices."""
    x_path = data_dir / "X.npy"
    y_path = data_dir / "Y.npy"

    # Read-only memory maps keep the full arrays off the Python heap and GPU.
    x_data = np.load(x_path, mmap_mode="r")
    y_data = np.load(y_path, mmap_mode="r")
    train_indices = np.load(data_dir / "train_indices.npy")
    validation_indices = np.load(data_dir / "validation_indices.npy")

    if x_data.shape[1] != INPUT_DIM or y_data.shape[1] != OUTPUT_DIM:
        raise ValueError(
            "Expected X/Y feature dimensions of "
            f"{INPUT_DIM}/{OUTPUT_DIM}, got "
            f"{x_data.shape[1]}/{y_data.shape[1]}."
        )
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Training and validation indices overlap.")

    train_dataset = RISRegressionDataset(x_data, y_data, train_indices)
    validation_dataset = RISRegressionDataset(
        x_data,
        y_data,
        validation_indices,
    )

    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2

    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle batches only; never perform a new split.
        generator=shuffle_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )
    return train_loader, validation_loader, train_indices, validation_indices


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return weighted validation MSE and Top-1 codebook accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)

        # Derive the true best codebook directly from each Y row.
        predicted_indices = predictions.argmax(dim=1)
        true_indices = y_batch.argmax(dim=1)

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predicted_indices == true_indices).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def load_trained_model(
    weight_path: str | Path = DEFAULT_WEIGHT_PATH,
    device: torch.device | None = None,
    dropout: float = DEFAULT_DROPOUT,
) -> tuple[RISCodebookMLP, torch.device]:
    """Reconstruct the model and load saved weights for inference."""
    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    if device is None:
        device = select_device()

    model = RISCodebookMLP(dropout=dropout).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@torch.no_grad()
def predict_best_codebook(
    model: RISCodebookMLP,
    feature: np.ndarray | torch.Tensor,
    device: torch.device,
) -> int:
    """Return the best RIS codebook index for one 1024-dimensional feature."""
    if isinstance(feature, np.ndarray):
        feature_tensor = torch.from_numpy(
            np.array(feature, dtype=np.float32, copy=True)
        )
    elif isinstance(feature, torch.Tensor):
        feature_tensor = feature.detach().to(dtype=torch.float32, device="cpu")
    else:
        raise TypeError("feature must be a NumPy array or a PyTorch tensor.")

    feature_tensor = feature_tensor.reshape(-1)
    if feature_tensor.numel() != INPUT_DIM:
        raise ValueError(
            f"Expected {INPUT_DIM} input features, "
            f"got {feature_tensor.numel()}."
        )

    model.eval()
    scores = model(feature_tensor.unsqueeze(0).to(device))
    return int(scores.argmax(dim=1).item())


def train_mlp(args: argparse.Namespace) -> Path:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1.")

    set_random_seed(args.seed)
    device = select_device()
    data_dir = resolve_data_dir(args.data_dir)
    weight_path = Path(args.weights).expanduser().resolve()
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_loader, validation_loader, train_indices, validation_indices = (
        build_dataloaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
        )
    )
    print(f"Training samples: {len(train_indices):,}")
    print(f"Validation samples: {len(validation_indices):,}")

    model = RISCodebookMLP(dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_loss = float("inf")
    best_top1_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
        )
        validation_loss, top1_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_loss:.6f} | "
            f"Top1_acc={top1_accuracy:.4%}"
        )

        if (
            top1_accuracy > best_top1_accuracy
            or (
                top1_accuracy == best_top1_accuracy
                and validation_loss < best_validation_loss
            )
        ):
            best_top1_accuracy = top1_accuracy
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), weight_path)

    # Reload the best validation checkpoint so subsequent inference is consistent.
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Best validation Top1 accuracy: {best_top1_accuracy:.4%}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Weights saved to: {weight_path}")
    return weight_path


def inference_example(
    data_dir: str | Path | None = None,
    weight_path: str | Path = DEFAULT_WEIGHT_PATH,
    sample_index: int = 0,
    feature: np.ndarray | torch.Tensor | None = None,
) -> int:
    """Predict one codebook index from a supplied or file-loaded feature."""
    if feature is None:
        resolved_data_dir = resolve_data_dir(
            None if data_dir is None else str(data_dir)
        )
        features = np.load(
            resolved_data_dir / "X.npy",
            mmap_mode="r",
        )
        feature = np.array(features[sample_index], dtype=np.float32, copy=True)

    model, device = load_trained_model(weight_path, device=select_device())
    best_index = predict_best_codebook(model, feature, device)
    print(f"Sample {sample_index}: best RIS codebook index = {best_index}")
    return best_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight MLP for RIS codebook selection."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing X.npy, Y.npy, and the split indices.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHT_PATH),
        help="Output path for ris_mlp_weight.pth.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Keep 0 on Windows/Jupyter unless worker processes are needed.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Skip training and run the single-sample inference example.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="X.npy row used by the inference example.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.inference:
        inference_example(
            data_dir=cli_args.data_dir,
            weight_path=cli_args.weights,
            sample_index=cli_args.sample_index,
        )
    else:
        train_mlp(cli_args)
