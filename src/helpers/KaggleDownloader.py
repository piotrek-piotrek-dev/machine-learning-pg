import logging
from pathlib import Path
from typing import Optional
from kaggle.api.kaggle_api_extended import KaggleApi
from src.includes.constants import DATASET_DST_DIR, TOKEN_FILE

log = logging.getLogger(__name__)


class KaggleDownloader:
    kaggleApi: KaggleApi

    def __init__(self):
        log.debug("check if token is present")
        if not Path.exists(TOKEN_FILE):
            log.critical("token not present in %s", TOKEN_FILE)
            raise Exception("""
            You need to provide a kaggle user token, please visit: https://www.kaggle.com/docs/api
            place the .json file in %s folder and rerun the app
            """, TOKEN_FILE)
        self.kaggleApi = KaggleApi()
        log.info("Authenticating with Kaggle")
        self.kaggleApi.authenticate()

    def downloadDataSet(self, dataSetName: str, dataSetFile: str = None) -> Optional[list:Path]:
        dsFiles = self.kaggleApi.dataset_list_files(dataSetName)
        if len(dsFiles.files) == 0:
            raise Exception("The requested dataset '%s' does not contain any files", dataSetName)

        if not self.isDataSetPresentOnSystem(dsFiles.files):
            if dataSetFile is None:
                log.debug("downloading whole dataset %s to %s", dataSetName, DATASET_DST_DIR)
                self.kaggleApi.dataset_download_files(dataSetName,
                                                 DATASET_DST_DIR,
                                                 unzip=True)
                if self.isDataSetPresentOnSystem(dsFiles.files):
                    return [ Path(DATASET_DST_DIR, f) for f in dsFiles.files ]
                else:
                    return None
            else:
                log.debug("downloading file '%s' from dataset '%s'", dataSetFile, dataSetName)
                self.kaggleApi.dataset_download_file(dataSetName,
                                                dataSetFile,
                                                DATASET_DST_DIR)
                if self.isDataSetPresentOnSystem([dataSetFile]):
                    return [Path(DATASET_DST_DIR, dataSetFile)]
                else:
                    return None
        else:
            return [ Path(DATASET_DST_DIR, f.name) for f in dsFiles.files ]


    def isDataSetPresentOnSystem(self, dsFiles: list) -> bool:
        if not DATASET_DST_DIR.exists():
            raise Exception("""The dataset dir '%s' not created""", DATASET_DST_DIR)

        if len(list(DATASET_DST_DIR.iterdir())) == 0:
            return False

        if any(fn in [file.name for file in dsFiles] for fn in [file.name for file in DATASET_DST_DIR.iterdir()]):
            return True
        else:
            return False

    def isDataSetDirEmpty(self) -> bool:
        return len(list(DATASET_DST_DIR.iterdir())) == 0

    def setDataSetDirectory(self) -> None:
        if not DATASET_DST_DIR.exists():
            DATASET_DST_DIR.mkdir()
