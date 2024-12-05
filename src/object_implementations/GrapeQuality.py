import logging
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Optional, Any, Dict
import plotly.express
from matplotlib.figure import Figure
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from includes.constants import DATASET_DST_DIR
from objects.AbstractModel import AbstractModel
from src.helpers.KaggleDownloader import KaggleDownloader
from src.includes.constants import AttachmentTypes
from src.objects.AbstractMachineLearning import AbstractMachineLearning


log = logging.getLogger(__name__)

class GrapeQuality(AbstractMachineLearning):

    def __init__(self):
        super().__init__()
        self.dataSetName = "grape-quality-dataset"
        self._dataSetKaggleURI = "mrmars1010/grape-quality"
        self.dontSaveAttachments = True


    def getDataSet(self) -> (Optional[pandas.DataFrame], str):
        # Kaggle API is having difficult times need to skip...
        kaggleDownloader = KaggleDownloader()
        dataSetFile = "GRAPE_QUALITY.csv"
        return pandas.read_csv(Path(DATASET_DST_DIR, dataSetFile)), dataSetFile

        # if (dataSetFiles := kaggleDownloader.downloadDataSet(dataSetName=self._dataSetKaggleURI)) is None:
        #     log.critical("💀something really bad happened on the way, check logs")
        #     raise Exception("💀something really bad happened on the way, check logs")
        # if len(dataSetFiles) > 1:
        #     return pandas.read_csv(self.selectSourceFile(dataSetFiles, dataSetFile)), dataSetFile
        # else:
        #     return pandas.read_csv(dataSetFiles[0]), dataSetFiles[0]

    def selectSourceFile(self, fileList: list, selection: str) -> Path:
        return next((p for p in fileList if p.name == selection) , None)

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
        self.addCommentToSection(f"- drop the 'sample_id' column that came with the original dataset\n"
                                 f"- There are no missing values - cool\n"
                                 f"- There are no duplicate rows\n"
                                 f"- reformatted the categorical columns for better handling\n"
                                 f"- quality_score is highly positive correlated with sugar_content and sun_exposure\n"
                                 f"- quality category will help us to categorize as it's highly correlated with "
                                 f"quality_score\n"
                                 f"- we don't see any outliers in the set\n"
                                 f"- quality_score has a a kind of normal distribution where median is 2.5\n"
                                 f"- quality_score's Kurtosis is -0.5, thus we don't expect extreme outliers but we "
                                 f"shouldn't expect predictable results in range of Q1-Q3 percentile "
                                 f"(flat top of chart)\n")

    # @override
    # def addAdditionalCorrelationsToDescriptionReport(self):
    #     return {
    #         "pearson": {"calculate": True},
    #         "spearman": {"calculate": True}
    #     }
    def cleanUpDataframe(self):
        #we don't need the id column that came with the dataset
        self.mainDataFrame.drop('sample_id', axis=1, inplace=True)
        self.addCommentToSection(f"- Dropped column 'sample_id' as we don't need it\n"
                                 f"- Other than that, this dataset seems clean")

    def exploratoryAnalysis(self):
        # let's see the heatmap ..
        df_heatmap:Figure = plt.figure(figsize=(12,10))
        sns.heatmap(
            self.mainDataFrame.select_dtypes(include=[numpy.number]).corr(),
            annot=True,
            cmap='coolwarm'
        )
        plt.title("Correlation Heatmap")
        self.addAttachment(df_heatmap,
                           AttachmentTypes.MATPLOTLIB_CHART,
                           "correlation_heatmap.png")
        df_heatmap.show()
        plt.close()
        self.addCommentToSection(
                f"- strong correlation between quality size/category and sugar, sun exposure and berry size\n")

        # let's examine sugar vs quality
        sugar_vs_category: Figure = plt.figure(figsize=(12,10))
        sns.boxplot(x='quality_category',
                    y='sugar_content_brix',
                    data=self.mainDataFrame)
        plt.title('Sugar vs quality category')
        plt.xlabel("quality category")
        plt.ylabel("Sugar content")
        self.addAttachment(sugar_vs_category,
                           AttachmentTypes.MATPLOTLIB_CHART,
                           "sugar_vs_category.png")
        sugar_vs_category.show()
        plt.close()

        self.addCommentToSection(f"- but we'd like to end with an estimation of successes (highest price and "
                                 f"quality) based on location of land")

        # having trouble in presenting this histogram in seaborn/matplotlib
        region_vs_quality = plotly.express.histogram(
            data_frame=self.mainDataFrame.groupby(['region', 'variety', 'quality_category'])[['quality_score']]
                                .sum().reset_index(),
            x='region',
            y='quality_score',
            color='variety',
            title='region vs quality')
        region_vs_quality.update_traces(marker_line_width=1)
        region_vs_quality.update_layout(title = {'font_color' : 'rosybrown', 'x' : 0.43, 'y' : 0.9,
                                                 'xanchor' : 'center', 'yanchor' : 'top'},
                                        font_color = 'lightsalmon', barmode = 'group',
                                        legend_title_font_color = 'fuchsia')

        region_vs_quality.show()
        plt.close()

        print(f"Let's explore correlation in a few columns:\n")
        # self._correlationCoefDeepDive('quality_score', 'berry_size_mm', 'quality_category')
        # self._correlationCoefDeepDive('quality_score', 'cluster_weight_g', 'quality_category')
        # self._correlationCoefDeepDive('sun_exposure_hours', 'berry_size_mm', 'quality_category')
        # self._correlationCoefDeepDive('soil_moisture_percent', 'berry_size_mm', 'quality_category')
        # self._correlationCoefDeepDive('sugar_content_brix', 'berry_size_mm', 'quality_category')
        # self._correlationCoefDeepDive('quality_score', 'sugar_content_brix', 'quality_category')
        # self._correlationCoefDeepDive('quality_score', 'sun_exposure_hours', 'quality_category')
        #
        #
        # print("Looking for outliers:\n")
        # self._detectOutliers('quality_score')
        # self._detectOutliers('berry_size_mm')
        # self._detectOutliers('cluster_weight_g')
        # self._detectOutliers('sun_exposure_hours')
        # self._detectOutliers('soil_moisture_percent')
        # self._detectOutliers('sugar_content_brix')


        self.addCommentToSection(f"- niether corr coef suggest any good relationship between berry size and "
                                 f"quality score\n"
                                 f"- seems obvious, but the greater the berry, the higher quality of wine\n"
                                 f"- this doesn't look like a regression but rather clusterization problem\n")

    def _correlationCoefDeepDive(self, first: str, second: str, hue: str):
        plot: Figure = plt.figure(figsize=(12,10))
        sns.scatterplot(x=first,
                        y=second,
                        hue=hue,
                        data=self.mainDataFrame)
        plt.title(f'{first} vs {second}')
        self.addAttachment(plot,
                           AttachmentTypes.MATPLOTLIB_CHART,
                           f"{first}_vs_{second}.png")
        plot.show()
        plt.close()
        # let's dive deeper:
        print(f"{first} to {second} correlation coef: "
              f"{self.mainDataFrame[first].corr(self.mainDataFrame[second])}\n")
        for category in self.mainDataFrame[hue].unique():
            tmp = self.mainDataFrame[self.mainDataFrame['quality_category'] == category]
            print(f"Corr coef between quality_score and berry_size on {category}:\n\t"
                  f"kendall:  {tmp[first].corr(tmp[second], method='kendall')}\n\t"
                  f"spearman: {tmp[first].corr(tmp[second], method='spearman')}\n\t"
                  f"pearson:  {tmp[first].corr(tmp[second], method='pearson')}\n")

    def _detectOutliers(self, column: str):
        plot: Figure = plt.figure(figsize=(12,10))
        sns.boxplot(y=column,
                    data=self.mainDataFrame)
        plt.title(f'outliers in {column}')
        self.addAttachment(plot,
                           AttachmentTypes.MATPLOTLIB_CHART,
                           f"outliers_{column}.png")
        plot.show()
        plt.close()
        print(f"Mean in {column} is:        {self.mainDataFrame[column].mean()}\n"
              f"Median in {column} is:      {self.mainDataFrame[column].median()}\n"
              f"Skewness in {column} is:    {self.mainDataFrame[column].skew()}\n")

    def dataWrangling(self) -> (DataFrame, DataFrame, ColumnTransformer):
        self.addCommentToSection(f"- Data contains categorical types. need encode: \n"
                                 f"\t- OneHotEncoding for quality category"
                                 f"\t- scaler for numerical - we don't loose anything if we apply it blindly\n")

        x = self.mainDataFrame.drop('quality_category', axis = 1)
        y = self.mainDataFrame['quality_category']

        categorical_column = [cname for cname in x.columns if x[cname].dtype == 'object']
        numerical_column = [cname for cname in x.columns if x[cname].dtype in ['int', 'float']]

        categorical_transformer = Pipeline(steps = [
            ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse_output = True))
        ])
        numerical_transformer = Pipeline(steps = [
            ('scaler', StandardScaler())
        ])

        modelPreprocessor = ColumnTransformer([
            ('cat', categorical_transformer, categorical_column),
            ('num', numerical_transformer, numerical_column)

        ])
        return x, y, modelPreprocessor

    def selectFeatures(self):
        pass

    def trainModel(self, x:DataFrame, y:DataFrame, preProcessor:ColumnTransformer) -> AbstractModel:
        print(f"just reassure if we have all we need:\n {self.mainDataFrame.dtypes}\n"
              f"we'll be checking out and comparing 3 different models:\n"
              f"1. Random Forest Tree\n"
              f"2. Gradient Boost\n"
              f"3. Extreme Gradient Boost (XGB)\n")

        # for an even game, let's settle a common split for all models:
        xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.2, random_state=2)

        randomForest = AbstractModel(RandomForestClassifier(random_state=2898),
                                     preProcessor,
                                     xTrain,
                                     xTest,
                                     yTrain,
                                     yTest)
        self._runModel(randomForest)


        return randomForest

    def _runModel(self, model: AbstractModel) -> Dict[str,Any]:
        modelName:str = model.__class__.__name__
        model.trainModel()
        metrics = model.calculateMetrics()

        print(f"accuracy for {modelName} model is: {metrics['accuracy']}\n"
              f"classification report for {modelName} model is:\n {metrics['classification_report']}\n"
              f"below is the confusion matrix for {modelName} model\n {metrics['confusion_matrix_array']}\n")
        confusionMatrix = model.getPlotOfConfusionMatrix(metrics['confusion_matrix_array'])
        confusionMatrix.plot()
        plt.title(f'{modelName} confusion matrix')
        self.addAttachment(confusionMatrix,
                           AttachmentTypes.MATPLOTLIB_CHART,
                           f"{modelName}_confusion_matrix.png")
        plt.show()
        plt.close()
        return metrics

    def evaluateModel(self):
        pass

    def explainResults(self):
        pass
