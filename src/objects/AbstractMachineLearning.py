import logging
from pathlib import Path
import pandas as pandas
from abc import abstractmethod, ABC
from typing import Dict, Optional, Any
from matplotlib.figure import Figure
import src.helpers.Utils as utils
from ydata_profiling import ProfileReport
from src.helpers.Utils import Attachment
from src.includes.constants import (
    Phases, DESCRIPTION_REPORT_NAME_TEMPLATE, ATTACHMENTS_DIR, AttachmentTypes,
)

log = logging.getLogger(__name__)


class AbstractMachineLearning(ABC):

    def __init__(self):
        utils.createDirectoryStructure()
        self._phaseAttachmentIndexes: Dict[Phases, int] = {p : 0 for p in Phases}
        self._attachmentsList: Dict[Phases, list[utils.Attachment]] = {p : [] for p in Phases}
        self.comments:Dict[Phases, list[str]] = {p : [] for p in Phases}
        self.dataSetName: str = ""
        self.dataSetFile: Path = Path()
        self.mainDataFrame: pandas.DataFrame = pandas.DataFrame()
        self.summary: Dict[Phases, str] = {p : f"### STAGE {p.name}COMMENT:\nPlease provide a comment" for p in Phases}

    def run(self):
        log.debug("entering main")
        log.info("Starting Machine Learning for data set '%s'", self.dataSetName)
        self.separator(Phases.ENTER, f"Starting Machine Learning for data set {self.dataSetName}")

        log.info("Obtaining DataSet")
        self.separator(Phases.DATA_GATHERING)
        self.mainDataFrame, self.dataSetFile = self.getDataSet()
        self.cleanDatasetName()

        #this is to set max output colums so functions i.e. describe() won't crop anything out
        pandas.set_option('display.max_columns', self.mainDataFrame.columns.size)
        self.addCommentToSection(Phases.DATA_GATHERING,
                                 f"We'll be using {self.dataSetFile} file for this project")
        self._summarizeSection(Phases.DATA_GATHERING)


        log.info("Describing DataSet")
        self.separator(Phases.DATA_DESCRIPTION)
        self.describeDataSet()
        self.prepareDescriptionReport()
        self._summarizeSection(Phases.DATA_DESCRIPTION)

        log.info("Cleaning up dataframe")
        self.separator(Phases.DATA_CLEANUP, "Using knowledge from previous step")
        self.cleanUpDataframe()
        self._summarizeSection(Phases.DATA_CLEANUP)

        log.info("Data standardization")
        self.separator(Phases.DATA_STANDARDIZATION, "Using knowledge from previous step")
        self.dataStandardization()
        self._summarizeSection(Phases.DATA_STANDARDIZATION)

        log.info("Going to EDA")
        self.separator(Phases.DATA_EXPLORATION, "Using knowledge from previous step")
        self.exploratoryAnalysis()
        self._summarizeSection(Phases.DATA_EXPLORATION)



    def separator(self, phase: Phases, message: str = None):
        print(f"""
        ------ Starting stage: {phase.name} ------""")
        if message: print(message + "\n")

    def addCommentToSection(self, phase: Phases, comment: str) -> None:
        self.comments[phase].append(comment)

    def saveCommentsFromSection(self, phase: Phases) -> None:
        comments = '\n'.join(self.comments[phase])
        self.summary[phase] = f"### STAGE {phase.name} COMMENT:\n{comments}"
        p = self.addAttachment(phase, comments, AttachmentTypes.PLAINTEXT, "comments")
        print(f"Comments saved to: {p}\n")

    def cleanDatasetName(self):
        self.dataSetName = utils.camelCaseAstring(self.dataSetName)

    @abstractmethod
    def getDataSet(self) -> (Optional[pandas.DataFrame], str):
        raise NotImplementedError

    @abstractmethod
    def describeDataSet(self):
        raise NotImplementedError

    @abstractmethod
    def exploratoryAnalysis(self):
        raise NotImplementedError

    @abstractmethod
    def dataStandardization(self):
        raise NotImplementedError

    @abstractmethod
    def selectFeatures(self):
        raise NotImplementedError

    @abstractmethod
    def trainModel(self):
        raise NotImplementedError

    @abstractmethod
    def evaluateModel(self):
        raise NotImplementedError

    @abstractmethod
    def explainResults(self):
        raise NotImplementedError

    @abstractmethod
    def cleanUpDataframe(self):
        raise NotImplementedError

    def addAdditionalCorrelationsToDescriptionReport(self) -> Optional[dict]:
        return None

    def prepareDescriptionReport(self):
        try:
            title=f"This is a description report for dataset: {self.dataSetName}"
            reportName = DESCRIPTION_REPORT_NAME_TEMPLATE.replace("XXX", self.dataSetName)
            additionalCorrelations: dict
            if not (additionalCorrelations := self.addAdditionalCorrelationsToDescriptionReport()):
                additionalCorrelations = {"auto": {"calculate": True}}
            self.generateProfileReport(self.mainDataFrame, title, reportName, additionalCorrelations)

        except Exception as e:
            log.critical("💀Tried to prepare a description report but got some errors on the way, see exception details:"+ str(e))
            raise e

    def generateProfileReport(self, data: pandas.DataFrame, title: str, reportName: str, correlations: Dict):
        path = Path(utils.getPathToRoot(), ATTACHMENTS_DIR, Phases.DATA_DESCRIPTION.name, reportName)
        if not path.exists():
            report = ProfileReport(data, title=title, explorative=True, correlations = correlations)
            print(f"saving report '{title}' to {str(path)}, this can take a while... \ngo and grab yourself a ☕️")
            log.info(f"saving report '{title}' to {str(path)}, this can take a while... \ngo and grab yourself a ☕️")
            self.addAttachment(Phases.DATA_DESCRIPTION, report, AttachmentTypes.PROFILEREPORT, reportName)
        else:
            print(f"Report {reportName} already exists in {path}\nskipping... 🎉")
            log.info(f"Report {reportName} already exists in {path}\nskipping... 🎉")

    def addAttachment(self, stage: Phases,
                      attachment: Any,
                      attachmentType: AttachmentTypes = None,
                      fileName: str = None,
                      comment: str = None) -> Path:
        """
        Note: you must save the figure before any call to plt.show()
        """
        if fileName is None:
            fileName = '_'.join([self.dataSetName, "fig", str(self._getPhaseAttachmentIndex(stage))])
        pathForFile: Path = Path(utils.getPathToRoot(), ATTACHMENTS_DIR, stage.name, fileName)

        if isinstance(attachment, Figure) or attachmentType == AttachmentTypes.MATPLOTLIB_CHART:
            if comment is None:
                comment = attachment.gca().get_title()
            attachment.savefig(pathForFile)
        elif isinstance(attachment, ProfileReport) or attachmentType == AttachmentTypes.PROFILEREPORT:
            attachment.to_file(pathForFile)
        elif isinstance(attachment, str) or attachmentType == AttachmentTypes.PLAINTEXT:
            pathForFile = pathForFile.with_suffix('.txt')
            with open(pathForFile, "w") as file:
                file.write(attachment)

        self._attachmentsList[stage].append(Attachment(fileName, pathForFile, comment))
        return pathForFile

    def _getPhaseAttachmentIndex(self, phase: Phases) -> int:
        index = self._phaseAttachmentIndexes[phase]
        self._phaseAttachmentIndexes[phase] = index + 1
        return index

    def _summarizeSection(self, section: Phases):
        print(self.summary.get(section))
        self.saveCommentsFromSection(section)
