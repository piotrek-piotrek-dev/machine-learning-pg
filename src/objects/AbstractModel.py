from abc import ABC
from typing import Any, Dict

from timeit import default_timer
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class AbstractModel(ABC):
    def __init__(self,
                 model:Any,
                 preProcessor: ColumnTransformer=None,
                 xTrain: DataFrame=None,
                 xTest: DataFrame=None,
                 yTrain: DataFrame=None,
                 yTest: DataFrame=None,):
        self.model=model
        self.x_train = xTrain if xTrain is not None else DataFrame()
        self.y_train = yTrain if yTrain is not None else DataFrame()
        self.x_test = xTest if xTest is not None else DataFrame()
        self.y_test = yTest if yTest is not None else DataFrame()
        self.modelPreprocessor:ColumnTransformer = preProcessor
        self._modelPipeline: Pipeline = Pipeline([('preprocessor', preProcessor), ('model', model)])
        self.metrics = Dict[str:Any]

    @staticmethod
    def measure_time(func):
        def wrapper(*args, **kwargs):
            start = default_timer()
            result = func(*args, **kwargs)
            end = default_timer()
            print(f"{func.__name__}() executed in {(end - start):.6f}s")
            return result, end-start
        return wrapper

    def splitTestTrain(self, xData:DataFrame, yData:DataFrame, **args) -> None:
        if len(args) == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(xData, yData)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(xData, yData, **args)

    @measure_time
    def trainModel(self):
        # TODO: capture the return of .fit!!
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

        self.metrics = {
            'accuracy': accuracy_score(yTest, predicted),
            'classification_report' : classification_report(yTest, predicted),
            'confusion_matrix_array' : confMatrix
        }
        return self.metrics

    def getPlotOfConfusionMatrix(self, confusionMatrix:Any)->Any:
        return ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=self.model.classes_)

    def __eq__(self, other):

        return True