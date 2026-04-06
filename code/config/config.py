# dataset options
DATA_DIR = '/home/emilio/Work/landfills/dataset/squared/'
OUTPUT_DIR = 'out'
INPUT_SIZE = 926
SPLITS = ['train', 'val']

# Training options
# Available models:
# googlenet, mobilenet_v3_large, resnet50, swin_v2_b, vit_b_16, vit_b_32
# MODEL = ["googlenet", "mobilenet_v3_large", "resnet50", "swin_v2_b", "vit_b_16", "vit_b_32"]

MODEL = ["dinov3", "alphaearth", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 25
