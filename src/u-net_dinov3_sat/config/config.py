from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "squared"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

INPUT_SIZE = 512
SPLITS = ["train", "val"]
BATCH_SIZE = 2
NUM_WORKERS = 4
EPOCHS = 60
LEARNING_RATE = 1e-4

BACKBONE_TYPE = "dinov3"
BACKBONE_NAME = "dinov3_vitl16"
BACKBONE_PATH = None
BACKBONE_FREEZE = False
EMBED_DIM = 1024
PATCH_SIZE = 16

MODEL_OUTPUT_PATH = CHECKPOINT_DIR / "unet_landfills.pth"
TRAINING_LOG_PATH = OUTPUT_DIR / "training_logs.csv"
METRICS_PLOT_PATH = OUTPUT_DIR / "metrics_summary.png"

INFERENCE_INPUT_DIR = DATA_DIR / "val" / "landfills"
INFERENCE_WEIGHTS_PATH = MODEL_OUTPUT_PATH
INFERENCE_THRESHOLD = 0.5
