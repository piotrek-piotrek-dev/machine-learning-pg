from abc import ABC
from typing import Any, Dict

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from helpers.Utils import measure_time


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
        self.modelPipeline: Pipeline = Pipeline([('preprocessor', preProcessor), ('model', model)])
        self.metrics: Dict[str:Any] = {}

    def splitTestTrain(self, xData:DataFrame, yData:DataFrame, **args) -> None:
        if len(args) == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(xData, yData)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(xData, yData, **args)

    def trainModel(self):
        # TODO: capture the return of .fit!!
        model, execTime = self._fitModel()
        self.metrics['fit_time'] = execTime

    @measure_time
    def _fitModel(self) -> (Any, float):
        self.modelPipeline.fit(self.x_train, self.y_train)

    def predict(self, xData:DataFrame = None) -> Any:
        if (x := xData) is None:
            x = self.x_test
        return self.modelPipeline.predict(x)

    def calculateMetrics(self, xTest:DataFrame=None, yTest:DataFrame=None) -> Dict[str,Any]:
        if (xTest is None) != (yTest is None):
            # XOR
            raise Exception("xTest and yTest cannot be both None")
        else:
            if xTest is None and yTest is None:
                xTest = self.x_test
                yTest = self.y_test

        predicted = self.predict(xTest)
        # https://www.kdnuggets.com/2022/09/visualizing-confusion-matrix-scikitlearn.html
        confMatrix = confusion_matrix(yTest, predicted, labels=self.model.classes_)

        # our metrics would include: accuracy, recall, precision f1
        self.metrics['accuracy'] = accuracy_score(yTest, predicted)
        #self.metrics['MAE'] = mean_absolute_error(yTest, predicted)
        self.metrics['classification_report_dict'] = classification_report(yTest, predicted, output_dict=True)
        self.metrics['classification_report_pretty'] = classification_report(yTest, predicted)
        self.metrics['confusion_matrix_array'] = confMatrix

        return self.metrics

    def getPlotOfConfusionMatrix(self, confusionMatrix:Any)->Any:
        return ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=self.model.classes_)
