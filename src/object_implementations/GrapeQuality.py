import logging
from pathlib import Path
import pandas
from typing import Optional
from src.helpers.KaggleDownloader import KaggleDownloader
from src.includes.constants import Phases
from src.objects.AbstractMachineLearning import AbstractMachineLearning


log = logging.getLogger(__name__)

class GrapeQuality(AbstractMachineLearning):

    def __init__(self):
        super().__init__()
        self.dataSetName = "grape-quality-dataset"
        self._dataSetKaggleURI = "mrmars1010/grape-quality"


    def describeDataSet(self):
        print(f"the shape of the dataset is: {self.mainDataFrame.shape}\n"
              f"columns are: {self.mainDataFrame.columns}\n"
              f"a few sample data to visualize:\n{self.mainDataFrame.head()}\n"
              f"The types which we'll be dealing with are:\n{self.mainDataFrame.dtypes}\n"
              f"some very quick stats:\n{self.mainDataFrame.describe(include='all').T}\n"
              f"MISSING VALS: are there any null values?\n{self.mainDataFrame.isnull().sum()}\n"
              f"MISSING VALS: are there any None's?\n{self.mainDataFrame.isna().sum()}\n"
              f"Are there any duplicate values?\n{self.mainDataFrame.duplicated().sum()}\n"
              f"Outliers in: "
              f"Kurtosis for ...")

        # sprwadz czy trzeba normaliowac/standaryzowac dane
        self.addCommentToSection(Phases.DATA_DESCRIPTION,
                                 f"- drop the 'sample_id' column that came with the original dataset\n"
                                 f"- There are no missing values - cool\n"
                                 f"- There are no duplicate rows\n"
                                 f"- reformatted the categorical columns for better handling\n"
                                 f"- quality_score is highly positive correlated with sugar_content and sun_exposure\n"
                                 f"- quality category will help us to categorize as it's highly correlated with quality_score\n"
                                 f"- we don't see any outliers in the set\n"
                                 f"- quality_score has a a kind of normal distribution where median is 2.5\n"
                                 f"- quality_score's Kurtosis is -0.5, thus we don't expect extreme outliers but we shouldn't expect predictable results in range of Q1-Q3 percentile (flat top of chart)\n")

    # @override
    # def addAdditionalCorrelationsToDescriptionReport(self):
    #     return {
    #         "pearson": {"calculate": True},
    #         "spearman": {"calculate": True}
    #     }
    def cleanUpDataframe(self):
        #we don't need the id column that came with the dataset
        self.mainDataFrame.drop('sample_id', axis=1, inplace=True)
        self.addCommentToSection(Phases.DATA_CLEANUP,r"- Dropped column 'sample_id' as we don't need it")

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

    def getDataSet(self) -> (Optional[pandas.DataFrame], str):
        kaggleDownloader = KaggleDownloader()
        dataSetFile = "GRAPE_QUALITY.csv"
        if (dataSetFiles := kaggleDownloader.downloadDataSet(dataSetName=self._dataSetKaggleURI)) is None:
            log.critical("💀something really bad happened on the way, check logs")
            raise Exception("💀something really bad happened on the way, check logs")
        if len(dataSetFiles) > 1:
            return pandas.read_csv(self.selectSourceFile(dataSetFiles, dataSetFile)), dataSetFile
        else:
            return pandas.read_csv(dataSetFiles[0]), dataSetFiles[0]

    def selectSourceFile(self, fileList: list, selection: str) -> Path:
        return next((p for p in fileList if p.name == selection) , None)
