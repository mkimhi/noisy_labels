from pathlib import Path
import matplotlib.pyplot as plt

CODE_PATH = Path(__file__).parent.parent
PROJECT_PATH = CODE_PATH.parent

SEED = 1489973
DEFAULT_PLT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

RESOURCES_PATH = PROJECT_PATH / 'resources'
CODE_RESOURCES_PATH = CODE_PATH / 'resources'

RAW_DATA_DIR_PATH =         RESOURCES_PATH / 'data'
PROCESSED_DATA_DIR_PATH =   RESOURCES_PATH / 'processed-resources'
CHECKPOINTS_DIR_PATH =      RESOURCES_PATH / 'checkpoints'
RESULTS_DIR_PATH =          RESOURCES_PATH / 'results'

SEED = 46878459

NOT_ALL_DATA_IS_AVAILABLE = True

# packages
MMDETECTION = 'mmdetection'
DETECTRON2 = 'detectron2'

CPU = 'cpu'
GROUND_TRUTH = 'ground-truth'

WNB_PROJECT_NAME = 'noisy-segmentation'
