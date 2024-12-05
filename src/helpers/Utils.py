import os
from pathlib import Path
from re import split
from timeit import default_timer
from typing import Any

import plotly.graph_objs
from matplotlib.figure import Figure
from ydata_profiling import ProfileReport

from src.includes.constants import PROJECT_NAME, DATASET_DST_DIR, ATTACHMENTS_DIR, REPORTS_DIR, Stages, AttachmentTypes


def getPathToRoot(start: Path = None):
    if start is None:
        start = Path.cwd()

    idx = 0
    for el in reversed(str(start).split(os.sep)):
        if el == PROJECT_NAME:
            break
        idx+=1
    if idx==0: return Path(start)
    else: return Path(start.parents[idx-1])

def camelCaseAstring(strignToCapitalize: str) -> str:
    return ''.join(
        w.capitalize() for w in split('([^a-zA-Z0-9])', strignToCapitalize) if w.isalnum()
    )

def createDirectoryStructure():
    root = getPathToRoot()
    if not Path(root, DATASET_DST_DIR).exists():
        Path(root, DATASET_DST_DIR).mkdir()
    if not Path(root, REPORTS_DIR).exists():
        Path(root, REPORTS_DIR).mkdir()
    if not Path(root, ATTACHMENTS_DIR).exists():
        Path(root, ATTACHMENTS_DIR).mkdir()
    for p in Stages:
        if not Path(root, ATTACHMENTS_DIR, p.name).exists():
            Path(root, ATTACHMENTS_DIR, p.name).mkdir()

class Attachment():
    def __init__(self, fileName: str, filePath:Path, comment:str):
        self.fileName = fileName
        self.filePath = filePath
        self.comment = comment

def saveAttachment(stage: Stages,
                   attachment: Any,
                   attachmentType: AttachmentTypes = None,
                   fileName: str = None) -> Path:
    pathForFile: Path = Path(getPathToRoot(), ATTACHMENTS_DIR, stage.name, fileName)
    if attachment is not None:
        if isinstance(attachment, Figure) and attachmentType == AttachmentTypes.MATPLOTLIB_CHART:
            attachment.savefig(pathForFile)
        elif isinstance(attachment, ProfileReport) or attachmentType == AttachmentTypes.PROFILEREPORT:
            attachment.to_file(pathForFile)
        elif isinstance(attachment,  plotly.graph_objs.Figure) and attachmentType == AttachmentTypes.PLOTLY_CHART:
            attachment.write_image(pathForFile)
        elif isinstance(attachment, str) or attachmentType == AttachmentTypes.PLAINTEXT:
            with open(pathForFile, "w") as file:
                file.write(attachment)

    return pathForFile

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = default_timer()
        result = func(*args, **kwargs)
        end = default_timer()
        #print(f"{func.__name__}() executed in {(end - start):.6f}s")
        return result, end-start
    return wrapper
