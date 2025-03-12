# Copyright (c) 2025
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import Optional, Union, Any

#######################
# Misc Settings
#######################

@dataclass
class Embedding:
    # Model class
    model: Any
    # Embedding size
    embedding_size: int
    # Name
    name: str
    # Preprocessor
    preprocessor: str

@dataclass
class SynthConfig:
    # Config name
    name: str
    # Config object
    config: Any

@dataclass
class Synth:
    # Synth name
    name: str
    # Synth object
    synth: Any

@dataclass
class Transform:
    # Mode of the transformation
    mode: str
    # List of transformations
    transform: Any

@dataclass
class Trainer:
    # Devices to use
    devices: Union[list[int], str, int]
    # Max number of epochs
    max_epochs: int
    # Root directory to save logs
    default_root_dir: str
    # Logger functions
    logger: list[Any]
    # Callback functions
    callbacks: list[Any]
    # Update norm from distributed training
    sync_batchnorm: bool
    # Detect anomalies mode
    detect_anomaly: bool
    # Frequency for logging
    log_every_n_steps: int
    # Frequency for checking validation
    check_val_every_n_epoch: int
    # CUDA acceleration
    benchmark: bool

#######################
# Scheduler Settings
#######################

@dataclass
class Scheduler:
    # Milestones proportional to the number of epochs
    milestones: list[float]

#######################
# General Settings
#######################

@dataclass
class General:
    # Synthetic or load from disk
    synthetic: bool
    # Directory to log results
    log_dir: str
    # Number of training epochs
    epochs: int
    # Training constrative hyperparameters
    unif_t: int
    align_alpha: int
    align_w: int
    unif_w: int
    # Whether to use tensorboard
    use_tensorboard: bool
    # Checkpoint to load, if any
    ckpt: Optional[str]
    # Run name if specified, otherwise will use a formatted datetime
    run_name: Optional[str]
    # Max number of audio files logged
    max_audio_log_n: int
    # Frequency of validation
    val_interval: int
    # Optimizer to use (adam or sgd)
    optimizer: str
    # Learning rate scheduler
    scheduler: Scheduler

#######################
# Synthetic Settings
#######################

@dataclass
class Synthetic:
    # Perturbation applied to the parameters to transform audio
    delta: float

#######################
# Pregeneration Settings
#######################

@dataclass
class Pregeneration:
    # Number of pairs (positive, negative) to generate for data analysis
    n_pairs: int
    # Perturbation applied to the parameters to transform audio
    delta: float
    # Number of seconds to layer for real world sounds
    n_layers: int
    # Batch size
    batch_size: int
    # Data directory to load the generated data
    data_dir: str
    # Checkpoint path to load the embedding model
    ckpt_path: str
    # Device for the embedding model
    device: str

#######################
# Data Settings
#######################

@dataclass
class Data:
    # Path to the dataset, if any
    path: str
    # Batch size
    batch_size: int
    # Number of workers
    n_workers: int
    # Train size in percentage
    train_size: float
    # Whether to apply transformations or not
    apply_transform: bool
    # Maximum number of samples
    max_n: int
    # Duration of the audio samples
    duration: int
    # Sample rate of the audio samples
    sample_rate: int
    # Paths for impulse responses (used in Audiomentations augmentations)
    ir_paths: str
    # Whether to apply temporal jitter augmentation
    temporal_jitter: bool
    # Number of layers to stack for real-world sounds
    n_layers: int
    # Settings for synthetic data
    synthetic: Synthetic
    # Settings for data pregeneration for analysis
    pregeneration: Pregeneration

######################
# System Settings
######################


@dataclass
class System:
    # Random seed for reproducibility
    seed: int
    # The devices to use. Positive number (int or str), a sequence of device indices (list or str),
    # the value -1 for all available devices, or "auto" for automatic selection.
    device: Union[list[int], str, int]


######################
# The Config
######################

@dataclass
class Config:
    # Embedding model
    embedding: Embedding
    # Synth config
    synthconfig: SynthConfig
    # Synth architecture
    synth: Synth
    # Transformations to apply
    transform: Transform
    # Trainer settings
    trainer: Trainer
    # General settings
    general: General
    # Data settings
    data: Data
    # System settings
    system: System
