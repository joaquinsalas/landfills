DATA_DIR    = '/home/emilio/Work/landfills/dataset/squared'
OUTPUT_DIR  = '/home/emilio/Work/landfills/out'
INPUT_SIZE  = 512
SPLITS      = ['train', 'val']
BATCH_SIZE  = 2
NUM_WORKERS = 4
EPOCHS      = 60

BACKBONE_TYPE   = 'dinov2'
BACKBONE_NAME   = 'dinov3_vitl16_sat493m'   # nombre timm
BACKBONE_PATH   = '/home/emilio/Work/landfills/code/dinov3/models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
BACKBONE_FREEZE = False
EMBED_DIM       = 1024
PATCH_SIZE      = 16
