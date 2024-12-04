from abc import ABC
from typing import Any, Dict

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class AbstractModel(ABC):
    def __init__(self, model:Any, preProcessor: ColumnTransformer=None):
        self.model=model
        self.x_train: DataFrame = DataFrame()
        self.y_train: DataFrame = DataFrame()
        self.x_test: DataFrame = DataFrame()
        self.y_test: DataFrame = DataFrame()
        self.modelPreprocessor:ColumnTransformer = preProcessor
        self._modelPipeline: Pipeline = Pipeline([('preprocessor', preProcessor), ('model', model)])

    def splitTestTrain(self, xData:DataFrame, yData:DataFrame, **args) -> None:
        if len(args) == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(xData, yData)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(xData, yData, **args)

    def trainModel(self):
        self._modelPipeline.fit(self.x_train, self.y_train)

    def predict(self, xData:DataFrame = None) -> Any:
        if (x := xData) is None:
            x = self.x_test
        return self._modelPipeline.predict(x)

    def calculateMetrics(self, xTest:DataFrame=None, yTest:DataFrame=None) -> Dict[str,Any]:
        if (xTest is None) != (yTest is None):
            # XOR
            raise Exception("xTest and yTest cannot be both None")
        else:
            if xTest is None and yTest is None:
                xTest = self.x_test
                yTest = self.y_test

        predicted = self.predict(xTest)
        confMatrix = confusion_matrix(yTest, predicted, labels=self.model.classes_)
        # https://www.kdnuggets.com/2022/09/visualizing-confusion-matrix-scikitlearn.html

        return{
            'accuracy': accuracy_score(yTest, predicted),
            'classification_report' : classification_report(yTest, predicted),
            'confusion_matrix_array' : confMatrix
        }

    def getPlotOfConfusionMatrix(self, confusionMatrix:Any)->Any:
        return ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=self.model.classes_)
