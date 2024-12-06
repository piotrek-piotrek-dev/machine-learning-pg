import logging
import re
from pathlib import Path
import pandas as pandas
from abc import abstractmethod, ABC
from typing import Dict, Optional, Any

from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

import src.helpers.Utils as utils
from ydata_profiling import ProfileReport

from objects.AbstractModel import AbstractModel
from src.helpers.Utils import Attachment
from src.includes.constants import (
    Stages, DESCRIPTION_REPORT_NAME_TEMPLATE, ATTACHMENTS_DIR, AttachmentTypes,
)

log = logging.getLogger(__name__)


class AbstractMachineLearning(ABC):

    def __init__(self):
        utils.createDirectoryStructure()
        self._phaseAttachmentIndexes: Dict[Stages, int] = {p : 0 for p in Stages}
        self._attachmentsList: Dict[Stages, list[utils.Attachment]] = {p : [] for p in Stages}
        self.dontSaveAttachments = False
        self.comments:Dict[Stages, list[str]] = {p : [] for p in Stages}
        self.summary: Dict[Stages, str] = {p : f"### STAGE {p.name}COMMENT:\nPlease provide a comment" for p in Stages}

        self.dataSetName: str = ""
        self.dataSetFile: Path = Path()
        self.mainDataFrame: DataFrame = DataFrame()

        # self.model: AbstractModel = None

        self.currentStage = Stages.INIT


    def run(self):
        log.debug("entering main")
        log.info("Starting Machine Learning for data set '%s'", self.dataSetName)
        self.currentStage = Stages.INIT
        self.separator(f"Starting Machine Learning for data set {self.dataSetName}")

        log.info("Obtaining DataSet")
        self.currentStage = Stages.DATA_GATHERING
        self.separator()
        self.mainDataFrame, self.dataSetFile = self.getDataSet()
        self.cleanDatasetName()

        #this is to set max output colums so functions i.e. describe() won't crop anything out
        pandas.set_option('display.max_columns', self.mainDataFrame.columns.size)
        self.addCommentToSection(f"We'll be using {self.dataSetFile} file for this project")
        self._summarizeSection()

        log.info("Describing DataSet")
        self.currentStage = Stages.DATA_DESCRIPTION
        self.separator()
        self.describeDataSet()
        self.saveDescriptionReport()
        self._summarizeSection()

        log.info("Cleaning up dataframe")
        self.currentStage = Stages.DATA_CLEANUP
        self.separator("Using knowledge from previous step")
        self.cleanUpDataframe()
        self._summarizeSection()

        log.info("Going to EDA")
        self.currentStage = Stages.DATA_EXPLORATION
        self.separator("Using knowledge from previous step")
        self.exploratoryAnalysis()
        self._summarizeSection()

        log.info("Data wrangling")
        self.currentStage = Stages.DATA_WRANGLING
        self.separator("Using knowledge from previous step")
        x, y, preProcessor = self.dataWrangling()
        self._summarizeSection()

        log.info("Modelling")
        self.currentStage = Stages.MODELING
        self.separator()
        model:AbstractModel = self.trainModel(x, y, preProcessor)
        self._summarizeSection()

        log.info("Feature exploring")
        self.currentStage = Stages.FEATURE_SELECTION
        self.separator()
        self.selectFeatures(model)

        log.info("tuning model (on features)")
        self.currentStage = Stages.MODEL_ADJUSTING
        self.separator()
        self.evaluateModel(model, x, y)



    def separator(self, message: str = None):
        print(f"""
        ------ Starting stage: {self.currentStage.name} ------""")
        if message: print(message + "\n")

    def addCommentToSection(self, comment: str) -> None:
        self.comments[self.currentStage].append(comment)

    def saveCommentsFromSection(self) -> None:
        comments = '\n'.join(self.comments[self.currentStage])
        self.summary[self.currentStage] = f"### STAGE {self.currentStage.name} COMMENT:\n{comments}"
        path = self.addAttachment(comments, AttachmentTypes.PLAINTEXT, "comments.txt", "section comments")
        print(f"Comments saved to: {path}\n")

    def cleanDatasetName(self):
        self.dataSetName = utils.camelCaseAstring(self.dataSetName)

    @abstractmethod
    def getDataSet(self) -> (Optional[pandas.DataFrame], str):
        raise NotImplementedError

    @abstractmethod
    def describeDataSet(self):
        """
        check missing vals
        identify duplicates
        verify data types
        check for outliers
        validate numeric ranges
        cross check column dependency
        Check for Inconsistent Data Entry
        """
        raise NotImplementedError

    @abstractmethod
    def exploratoryAnalysis(self):
        raise NotImplementedError

    @abstractmethod
    def dataWrangling(self) -> (DataFrame, DataFrame, ColumnTransformer):
        raise NotImplementedError

    @abstractmethod
    def selectFeatures(self, model:AbstractModel) -> None:
        raise NotImplementedError

    @abstractmethod
    def trainModel(self, x: DataFrame, y: DataFrame, preProcessor: ColumnTransformer) -> AbstractModel:
        raise NotImplementedError

    @abstractmethod
    def evaluateModel(self, model: AbstractModel, x: DataFrame, y: DataFrame):
        raise NotImplementedError

    @abstractmethod
    def explainResults(self):
        raise NotImplementedError

    @abstractmethod
    def cleanUpDataframe(self):
        raise NotImplementedError

    def addAdditionalCorrelationsToDescriptionReport(self) -> Optional[dict]:
        return None

    def saveDescriptionReport(self) -> Path:
        try:
            title=f"This is a description report for dataset: {self.dataSetName}"
            reportName = DESCRIPTION_REPORT_NAME_TEMPLATE.replace("XXX", self.dataSetName)
            additionalCorrelations: dict
            if not (additionalCorrelations := self.addAdditionalCorrelationsToDescriptionReport()):
                additionalCorrelations = {"auto": {"calculate": True}}
            report = self.generateProfileReport(self.mainDataFrame, title, reportName, additionalCorrelations)
            return self.addAttachment(
                report,
                AttachmentTypes.PROFILEREPORT,
                reportName,
                "Data description report")
        except Exception as e:
            log.critical("💀Tried to prepare a description report but got some errors on the way, see exception details:"+ str(e))
            raise e

    def generateProfileReport(self, data: pandas.DataFrame, title: str, reportName: str, correlations: Dict) -> Optional[ProfileReport]:
        path = Path(utils.getPathToRoot(), ATTACHMENTS_DIR, Stages.DATA_DESCRIPTION.name, reportName)
        if not path.exists():
            report = ProfileReport(data, title=title, explorative=True, correlations = correlations)
            print(f"saving report '{title}' to {str(path)}, this can take a while... \ngo and grab yourself a ☕️")
            log.info(f"saving report '{title}' to {str(path)}, this can take a while... \ngo and grab yourself a ☕️")
            return report
                #self.addAttachment(Phases.DATA_DESCRIPTION, report, AttachmentTypes.PROFILEREPORT, reportName)
        else:
            print(f"Report {reportName} already exists in {path}\nskipping... 🎉")
            log.info(f"Report {reportName} already exists in {path}\nskipping... 🎉")
            return None

    def addAttachment(self,
                      attachment: Any,
                      attachmentType: AttachmentTypes = None,
                      fileName: str = None,
                      comment: str = None) -> Optional[Path]:
        """
        Note: you must save the figure before any call to plt.show()
        """
        if self.dontSaveAttachments == True:
            return None

        if fileName is None:
            fileName = '_'.join([self.dataSetName, "fig", str(self._getPhaseAttachmentIndex(self.currentStage))])
        if comment is None and (attachmentType is AttachmentTypes.MATPLOTLIB_CHART or isinstance(attachment, Figure)):
            comment = attachment.gca().get_title()
        elif comment is None and (attachmentType is AttachmentTypes.PLOTLY_CHART or isinstance(attachment, Figure)):
            comment = attachment.layout.title.text

        pathForFile = utils.saveAttachment(self.currentStage, attachment, attachmentType, fileName)
        self._attachmentsList[self.currentStage].append(Attachment(fileName, pathForFile, comment))
        return pathForFile

    def _getPhaseAttachmentIndex(self, phase: Stages) -> int:
        index = self._phaseAttachmentIndexes[phase]
        self._phaseAttachmentIndexes[phase] = index + 1
        return index

    def _summarizeSection(self):
        self.saveCommentsFromSection()
        #%% a cell
        print(self.summary.get(self.currentStage)+"\n###ATTACHMENTS:")
        for a in self._attachmentsList[self.currentStage]:
            print(f"'{a.comment}': {a.fileName}")

    def exposeOutliers(self, *columnName: str) -> {str: DataFrame}:
        """
        check for outliers using interquartile values
        https://www.khanacademy.org/math/cc-sixth-grade-math/cc-6th-data-statistics/cc-6th/a/interquartile-range-review
        :param columnName: strings representing the column to search for outliers
        :return: a new Dataframe object with detected outliers
        """
        ret: {str:DataFrame} = {}
        for c in columnName:
            col = self.mainDataFrame[c]
            quantile1 = col.quantile(0.25)
            quantile3 = col.quantile(0.75)
            iqr = quantile3 - quantile1
            lower_bound = quantile1 - 1.5 * iqr
            upper_bound = quantile3 + 1.5 * iqr
            ret[c] = self.mainDataFrame[(col < lower_bound) | (col > upper_bound)]
        return ret

    def valuesOutsideOfNumericRange(self, columns: [str], ranges: [(int,int)]) -> {str: DataFrame}:
        if len(columns) != len(ranges):
            raise Exception("Columns and ranges must have the same length")
        ret: {str:DataFrame} = {}
        for idx, col in enumerate(columns):
            ret[col] = self.mainDataFrame[~self.mainDataFrame[col].between(*ranges[idx])]
        return ret

    def checkDataFormat(self, columns: [str], format: str) -> {str:DataFrame}:
        ret: {str:DataFrame} = {}
        def checkFormat(payload: Any, scheme: str) -> bool:
            return re.match(str(payload), scheme) is not None
        for column in columns:
            ret[column] = self.mainDataFrame[column].apply(lambda x: checkFormat(x, format))
        return ret
