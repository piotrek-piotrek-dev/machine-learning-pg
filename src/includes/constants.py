from enum import Enum
from pathlib import Path

DESCRIPTION_REPORT_NAME_TEMPLATE = "describe_dataframe_XXX.html"
PROJECT_NAME = "machine-learning-final-project"
REPORTS_DIR = Path("reports")
ATTACHMENTS_DIR = Path("attachments")
TOKEN_FILE = Path(Path.home(), ".kaggle", "kaggle.json")
DATASET_DST_DIR = Path("..", "data_sets")
RANDOM_STATE_MAGIC_NUMBER = 2898

class Stages(Enum):
    INIT = 0
    DATA_GATHERING = 1,
    DATA_DESCRIPTION = 2,
    DATA_CLEANUP = 3,
    DATA_WRANGLING = 4,
    DATA_EXPLORATION = 5,
    FEATURE_SELECTION = 6,
    TRAINING = 7,
    MODEL_ADJUSTING = 8,
    RESULTS_INTERPRETATION = 9,

class AttachmentTypes(Enum):
    PROFILEREPORT = 0,
    MATPLOTLIB_CHART = 1,
    PLAINTEXT = 2,
    HTML = 3,
    JSON = 4,
    CSV = 5,
    PLOTLY_CHART =6,

class Metrics(Enum):
    ACCURACY = 0,
    PRECISION = 1,
    MAE = 2,
    CLASSIFICATION_REPORT_DICT = 4,
    CLASSIFICATION_REPORT = 5,
    CONFUSION_MATRIX_ARRAY = 6,
    FIT_TIME = 7,
    XVAL = 8,
