import os
from pathlib import Path
from re import split

from src.includes.constants import PROJECT_NAME, DATASET_DST_DIR, ATTACHMENTS_DIR, REPORTS_DIR, Phases


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
    for p in Phases:
        if not Path(root, ATTACHMENTS_DIR, p.name).exists():
            Path(root, ATTACHMENTS_DIR, p.name).mkdir()

class Attachment():
    def __init__(self, fileName: str, filePath:Path, comment:str):
        self.fileName = fileName
        self.filePath = filePath
        self.comment = comment
