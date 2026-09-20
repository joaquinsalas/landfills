from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

INPUT_SIZE = 512
SPLITS = ["train", "val"]
SEED = 42
BATCH_SIZE = 2
NUM_WORKERS = 4
EPOCHS = 5
LEARNING_RATE = 1e-4

BACKBONE_TYPE = "dinov3"
BACKBONE_NAME = "vitl16_sat"
BACKBONE_PATH = PROJECT_ROOT / "weights" / "vitl16_sat.pth"
BACKBONE_FREEZE = True
EMBED_DIM = 1024
PATCH_SIZE = 16

MODEL_OUTPUT_PATH = CHECKPOINT_DIR / "best_unet_landfills.pth"
LAST_MODEL_OUTPUT_PATH = CHECKPOINT_DIR / "last_unet_landfills.pth"
TRAINING_LOG_PATH = OUTPUT_DIR / "training_logs.csv"
METRICS_PLOT_PATH = OUTPUT_DIR / "metrics_summary.png"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"

INFERENCE_INPUT_DIR = DATA_DIR / "val" / "landfills"
INFERENCE_WEIGHTS_PATH = MODEL_OUTPUT_PATH
INFERENCE_THRESHOLD = 0.3
