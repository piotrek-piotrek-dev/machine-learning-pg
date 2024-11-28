import logging
from pathlib import Path
from typing import override
import pandas
from typing import Optional

from src.helpers.KaggleDownloader import KaggleDownloader
from src.includes.constants import Phases
from src.objects.AbstractMachineLearning import AbstractMachineLearning

log = logging.getLogger(__name__)


class WineTaseML(AbstractMachineLearning):

    def __init__(self):
        #"zynicide/wine-reviews"
        super().__init__()
        self.dataSetName = "wine-quality-dataset"
        self.dataSetKaggleURI = "adarshde/wine-quality-dataset"

    def describeDataSet(self):
        print(f"the shape of the dataset is: {self.mainDataFrame.shape}\n"
              f"columns are: {self.mainDataFrame.columns}\n"
              f"a few sample data to visualize:\n{self.mainDataFrame.head()}\n"
              f"The types which we'll be dealing with are:\n{self.mainDataFrame.dtypes}\n"
              f"some very quick stats:\n{self.mainDataFrame.describe(include='all').T}\n"
              f"MISSING VALS: are there any null values?\n{self.mainDataFrame.isnull().sum()}\n"
              f"MISSING VALS: are there any None's?\n{self.mainDataFrame.isna().sum()}\n"
              f"Are there any duplicate values?\n{self.mainDataFrame.duplicated().sum()}\n"
              f"Outliers in: ")

        # sprwadz czy trzeba normaliowac/standaryzowac dane
        self.addCommentToSection(Phases.DATA_DESCRIPTION,
                                 f"The dataset does not contain nulls or missing values - cool, however it"
                                 f"does contain dome duplicate values which we need to deal with")

    @override
    def addAdditionalCorrelationsToDescriptionReport(self):
        return {
            "pearson": {"calculate": True},
            "spearman": {"calculate": True}
        }

    def exploratoryAnalysis(self):

        pass

    def selectFeatures(self):
        pass

    def trainModel(self):
        pass

    def evaluateModel(self):
        pass

    def explainResults(self):
        pass

    def cleanUpDataframe(self):
        pass

    def getDataSet(self) -> (Optional[pandas.DataFrame], str):
        kaggleDownloader = KaggleDownloader()
        dataSetFile = "winequality-dataset-updated.csv"
        if (dataSetFiles := kaggleDownloader.downloadDataSet(dataSetName=self.dataSetKaggleURI)) is None:
            log.critical("💀something really bad happened on the way, check logs")
            raise Exception("💀something really bad happened on the way, check logs")
        if len(dataSetFiles) > 1:
            return pandas.read_csv(self.selectSourceFile(dataSetFiles, dataSetFile)), dataSetFile
        else:
            return pandas.read_csv(dataSetFiles[0]), dataSetFiles[0]

    def selectSourceFile(self, fileList: list, selection: str) -> Path:
        return next((p for p in fileList if p.name == selection), None)
