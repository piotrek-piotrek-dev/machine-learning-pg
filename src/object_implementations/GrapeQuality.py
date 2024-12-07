import logging
from functools import partial
from pathlib import Path

import numpy
import optuna
import pandas
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Optional, Any
import plotly.express
from matplotlib.figure import Figure
from optuna.terminator import report_cross_validation_scores
from pandas import factorize
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.orm import EXT_CONTINUE
from xgboost import XGBClassifier

from src.includes.constants import DATASET_DST_DIR, Metrics, RANDOM_STATE_MAGIC_NUMBER
from src.objects.AbstractModel import AbstractModel
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
        self.addCommentToSection(f"the shape of the dataset is: {self.mainDataFrame.shape}\n"
              f"columns are: {self.mainDataFrame.columns}\n"
              f"a few sample data to visualize:\n{self.mainDataFrame.head()}\n"
              f"The types which we'll be dealing with are:\n{self.mainDataFrame.dtypes}\n"
              f"some very quick stats:\n{self.mainDataFrame.describe(include='all').T}\n"
              f"MISSING VALS: are there any null values?\n{self.mainDataFrame.isnull().sum()}\n"
              f"MISSING VALS: are there any None's?\n{self.mainDataFrame.isna().sum()}\n"
              f"Are there any duplicate values?\n{self.mainDataFrame.duplicated().sum()}\n"
              f"Outliers in: "
              f"Kurtosis for ...")
        # check if std/norm is necessary

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
    """
   we'd use this method if we'd need to deeply explore interactions on certain columns
   """
    # @override
    # def addAdditionalCorrelationsToDescriptionReport(self):
    #     return {
    #         "pearson": {"calculate": True},
    #         "spearman": {"calculate": True}
    #     }
    def cleanUpDataframe(self):
        #we don't need the id column that came with the dataset
        self.mainDataFrame.drop('sample_id', axis=1, inplace=True)
        self.mainDataFrame["quality_category"] = self.mainDataFrame["quality_category"].astype('category')
        self.mainDataFrame["variety"] = self.mainDataFrame["variety"].astype('category')
        self.mainDataFrame["region"] = self.mainDataFrame["region"].astype('category')
        self.mainDataFrame["harvest_date"] = self.mainDataFrame["harvest_date"].astype('datetime64[ns]')

        self.mainDataFrame.info()
        self.addCommentToSection(f"- Dropped column 'sample_id' as we don't need it\n"
                                 f"- changed column types"
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
        """
        below lines will cook my PC :)
        print(f"Let's explore correlation in a few more columns:\n")
        self._correlationCoefDeepDive('quality_score', 'berry_size_mm', 'quality_category')
        self._correlationCoefDeepDive('quality_score', 'cluster_weight_g', 'quality_category')
        self._correlationCoefDeepDive('sun_exposure_hours', 'berry_size_mm', 'quality_category')
        self._correlationCoefDeepDive('soil_moisture_percent', 'berry_size_mm', 'quality_category')
        self._correlationCoefDeepDive('sugar_content_brix', 'berry_size_mm', 'quality_category')
        self._correlationCoefDeepDive('quality_score', 'sugar_content_brix', 'quality_category')
        self._correlationCoefDeepDive('quality_score', 'sun_exposure_hours', 'quality_category')

        print("Looking for outliers:\n")
        self._detectOutliers('quality_score')
        self._detectOutliers('berry_size_mm')
        self._detectOutliers('cluster_weight_g')
        self._detectOutliers('sun_exposure_hours')
        self._detectOutliers('soil_moisture_percent')
        self._detectOutliers('sugar_content_brix')"""

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
        self.addCommentToSection(f"{first} to {second} correlation coef: "
              f"{self.mainDataFrame[first].corr(self.mainDataFrame[second])}\n")
        for category in self.mainDataFrame[hue].unique():
            tmp = self.mainDataFrame[self.mainDataFrame['quality_category'] == category]
            self.addCommentToSection(f"Corr coef between quality_score and berry_size on {category}:\n\t"
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
        self.addCommentToSection(f"Mean in {column} is:        {self.mainDataFrame[column].mean()}\n"
                                  f"Median in {column} is:      {self.mainDataFrame[column].median()}\n"
                                  f"Skewness in {column} is:    {self.mainDataFrame[column].skew()}\n")

    def dataWrangling(self) -> (DataFrame, DataFrame, ColumnTransformer):
        self.addCommentToSection(f"- Data contains categorical types. need encode: \n"
                                 f"\t- OneHotEncoding for category\n"
                                 f"\t- scaler for numerical - we don't loose anything if we apply it blindly\n"
                                 f"\t- leaving out the date column/format\n"
                                 f"- wrap that in a column transformer\n"
                                 f"- and snap all with a pipeline so it'll be a clean object to pass around")

        x = self.mainDataFrame.drop('quality_category', axis = 1)
        y = self.mainDataFrame['quality_category']

        numerical_column = [cname for cname in x.columns if x[cname].dtype in ['int', 'float']]
        categorical_column = [cname for cname in x.columns if x[cname].dtype == 'category']

        # https://www.kaggle.com/code/marcinrutecki/one-hot-encoding-everything-you-need-to-know
        categorical_transformer = Pipeline(steps = [
            ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
        ])
        numerical_transformer = Pipeline(steps = [
            ('scaler', StandardScaler())
        ])

        # https://www.kdnuggets.com/building-data-science-pipelines-using-pandas
        modelPreprocessor = ColumnTransformer([
            ('cat', categorical_transformer, categorical_column),
            ('num', numerical_transformer, numerical_column),
        ])

        encoded, labels = factorize(y)
        # below conversion will screw up cross validation runs ... ehh..
        #encoded = pandas.DataFrame(data=encoded.T, columns=['category'])

        return x, {'encoded':encoded, 'labels':labels}, modelPreprocessor

    def trainModel(self, x:DataFrame, ySet:{str: DataFrame}, preProcessor:ColumnTransformer) -> AbstractModel:
        print(f"just reassure if we have all we need:\n {x.dtypes}\n"
              f"we'll be checking out and comparing 3 different models:\n"
              f"1. Random Forest Tree\n"
              f"2. Gradient Boost\n"
              f"3. Extreme Gradient Boost (XGB)\n")

        # for an even game, let's settle a common split for all models:
        xTrain, xTest, yTrain, yTest = train_test_split(x,ySet['encoded'], test_size=0.2, random_state=RANDOM_STATE_MAGIC_NUMBER)

        # let's store model parameters for later
        randomForest = AbstractModel(RandomForestClassifier(random_state=RANDOM_STATE_MAGIC_NUMBER),
                                     preProcessor,
                                     xTrain,
                                     xTest,
                                     yTrain,
                                     yTest,
                                     ySet['labels'])
        gradientBoostClassifier = AbstractModel(GradientBoostingClassifier(random_state = RANDOM_STATE_MAGIC_NUMBER,
                                                                           n_estimators = 500,
                                                                           learning_rate = 0.5, ),
                                                preProcessor,
                                                xTrain,
                                                xTest,
                                                yTrain,
                                                yTest,
                                                ySet['labels'])
        xgBoost = AbstractModel(XGBClassifier(random_state=RANDOM_STATE_MAGIC_NUMBER, n_estimators = 500,learning_rate = 0.5),
                                preProcessor,
                                xTrain,
                                xTest,
                                yTrain,
                                yTest,
                                ySet['labels'])

        modelDict = {
            "Random Forest Tree": randomForest,
            "Gradient Boost":  gradientBoostClassifier,
            "Xtreme Gradient Boost (XGB)": xgBoost
        }
        # preparing to do this in parallel ...
        # however now, the model fit times are so low that the parallel FW would be an overhaul
        for name, model in modelDict.items():
            self._runModel(model)

        self.addCommentToSection(f"- Since the XGBC is so good, in the next phase it won't show any progress, so instead"
                                 f"we'll take the RandomForestClassifier for next stage\n")

        return randomForest

    def _runModel(self, model: AbstractModel) -> None:
        modelName:str = model.model.__class__.__name__
        model.trainModel()
        metrics = model.calculateMetrics()

        metricsSummary:str = ""
        metricsSummary += f"Metrics for {modelName}:\n"
        print(metricsSummary)
        for k,v in metrics.items():
            metricsSummary += f"\t{k}:\n\t{v}\n"
            print(metricsSummary)
        metricsSummary+=f"\n\n"
        print(f"\n\n")
        confusionMatrix = model.getPlotOfConfusionMatrix(metrics[Metrics.CONFUSION_MATRIX_ARRAY])
        confusionMatrix.plot()
        plt.title(f'{modelName} confusion matrix')
        self.addAttachment(confusionMatrix,
                           AttachmentTypes.MATPLOTLIB_CHART,
                           f"{modelName}_confusion_matrix.png")
        plt.show()
        # calculate ROC
        # https://www.sharpsightlabs.com/blog/scikit-learn-roc_curve/
        plt.close()

        self.addAttachment(
            metricsSummary,
            AttachmentTypes.PLAINTEXT,
            f"{modelName}_metrics_summary.txt"
            f"metrics for model: {modelName}\n"
        )

    def crossValidation(self, model:AbstractModel, x: DataFrame, ySet: {str:DataFrame}):
        # https://scikit-learn.org/stable/modules/cross_validation.html
        # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

        # let's see if cross validation will increase our metrics
        self.addCommentToSection(f'initial classification report:\n {model.metrics[Metrics.CLASSIFICATION_REPORT]}\n'
              f"fit time: {model.metrics[Metrics.FIT_TIME]}\n"
              f"accuracy: {model.metrics[Metrics.ACCURACY]}\n\n"
              f"we'll try to run cross validation to see if splitting the set can do some good")
        # scoring parameter can be omne of: 'neg_mean_squared_log_error', 'mutual_info_score', 'roc_auc', 'adjusted_mutual_info_score', 'neg_log_loss', 'normalized_mutual_info_score', 'neg_mean_squared_error', 'f1_macro', 'neg_negative_likelihood_ratio', 'recall_micro', 'homogeneity_score', 'fowlkes_mallows_score', 'max_error', 'neg_mean_absolute_percentage_error', 'f1_weighted', 'matthews_corrcoef', 'precision', 'average_precision', 'jaccard_macro', 'jaccard_weighted', 'neg_mean_gamma_deviance', 'precision_samples', 'f1', 'neg_mean_poisson_deviance', 'recall_macro', 'neg_brier_score', 'jaccard', 'f1_micro', 'neg_root_mean_squared_log_error', 'recall_weighted', 'roc_auc_ovr_weighted', 'explained_variance', 'neg_mean_absolute_error', 'rand_score', 'accuracy', 'precision_weighted', 'roc_auc_ovo_weighted', 'jaccard_samples', 'f1_samples', 'top_k_accuracy', 'neg_median_absolute_error', 'adjusted_rand_score', 'completeness_score', 'v_measure_score', 'recall', 'positive_likelihood_ratio', 'roc_auc_ovo', 'precision_micro', 'recall_samples', 'jaccard_micro', 'precision_macro', 'd2_absolute_error_score', 'neg_root_mean_squared_error', 'balanced_accuracy', 'roc_auc_ovr', 'r2'
        scoring = ['accuracy']#, 'average_precision']
        self.addCommentToSection(f'Running cross validation experiments\n')
        for cv in [5, 10, 20]:
            scores = cross_validate(model.modelPipeline, x, ySet['encoded'], cv=cv, scoring=scoring)
            print(f'{scoring} scores for {cv} clusters (input set is {x.shape[0]} long):\n')
            for k, v in scores.items():
                print(f"\t{k}: {v}\n")

        self.addCommentToSection(f"- cross validation shows that dividing the set to 10 clusters is enough to obtain"
                                 f"reasonable time to train and accuracy\n"
                                 f"- would like to implement one more test - see if model is over or under fitted ('When a model performs highly on the training set but poorly on the test set, this is known as overfitting, or essentially creating a model that knows the training set very well but cannot be applied to new problems.')")

        # below line will kill your session :)
        # self._useRFECV(x,ySet['encoded'], model)


        # can't get SHAP to work due to: https://github.com/slundberg/shap/issues/2662
        # _x = model.modelPreprocessor.fit_transform(x)
        #
        # shap.initjs()
        # explainer = shap.TreeExplainer(model.model)
        # shap_values = explainer(_x)
        # shap.summary_plot(shap_values, _x)

    def _useRFECV(self, x:DataFrame, y: DataFrame, model: AbstractModel) -> Any:
        """
        y - must be ySet['encoded']
        rfecv = RFECV(estimator=_model.model,
                      step=1,
                      cv=5,
                      scoring='accuracy',
                      min_features_to_select=1,
                      n_jobs=-1
                      )
        rfecv.fit(_x,ySet['encoded'])
        print(f"Optimal number of features: {rfecv.n_features_}")
        rfecv.support_rfecv_df = DataFrame(data=rfecv.ranking_,index=_x.columns,columns=[‘Rank’]).sort_values(by=’Rank’,ascending=True)
        rfecv_df.head()
        """
        rfecv = RFECV(estimator=model.model)
        pipeline = Pipeline([('feature selection', rfecv), ('model', model.model)])
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE_MAGIC_NUMBER)
        n_scores = cross_val_score(pipeline, x, y, scoring='accuracy', cv=cv, n_jobs=-1)

        pipeline.fit(x,y)
        print(f"Optimal number of features: {rfecv.n_features_}")

        rfecv_df = pandas.DataFrame(data=rfecv.ranking_,index=x.columns,columns=["Rank"]).sort_values(by="Rank",ascending=True)
        print(rfecv_df.head())

    def tuneModelHyperParameters(self, model: AbstractModel, x: DataFrame, ySet: {str:DataFrame}) -> AbstractModel:
        """
        https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
        https://scikit-learn.org/stable/modules/grid_search.html#grid-search
        PCA: https://medium.com/biased-algorithms/shap-values-for-categorical-features-fd4d0ae6edec
        :returns:
        clf = Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
            ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)
        """

        self.addCommentToSection(f"Parameters of our estimator: {model.model.__class__.__name__}"
                                 f" {model.model.get_params()}\n"
                                 f"- we'll be experimenting with GridSearchCV and then with Optuna"
                                 f"let's try optuna...")
        study = optuna.create_study(direction='minimize')

        trials = 10
        method = Metrics.ACCURACY
        objective = partial(self._reduceHyperParams, model=model, method=method)
        study.optimize(objective, n_trials=trials, show_progress_bar=True)

        self.addCommentToSection(f"Optuna results with {trials} and {method.name}\n"
                                 f"Best trial:{study.best_trial}\n"
                                 f"Best hyperparameters: {study.best_params}\n")
        fig = optuna.visualization.plot_optimization_history(study)
        plotly.io.show(fig)
        # below chart wouldb e great but it freezes my PC...
        # fig = optuna.visualization.plot_terminator_improvement(study, plot_error=True)
        # plotly.io.show(fig)

        try:
            bestParameters = AbstractModel(
                model = RandomForestClassifier(**study.best_params),
                preProcessor=model.modelPreprocessor,
                xTrain = model.x_train,
                yTrain = model.y_train,
                xTest = model.x_test,
                yTest = model.y_test,
                labels=model.labels
            )
            return bestParameters
        except Exception as e:
            raise Exception("Trouble in assigning hyperparameters to the model" + repr(e))


    def _reduceHyperParams(self, trial, model: AbstractModel, method: Metrics) -> float:
        n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        rndForestClf = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=RANDOM_STATE_MAGIC_NUMBER)
        pipeline = Pipeline([
            ('preprocessor', model.modelPreprocessor),
            ('model', rndForestClf)
        ])
        pipeline.fit(model.x_train, model.y_train)
        pred = pipeline.predict(model.x_test)
        score = 0.0
        if method == Metrics.PRECISION:
            score = precision_score(model.y_test, pred)
        elif method == Metrics.ACCURACY:
            score = accuracy_score(model.y_test, pred)
        elif method == Metrics.MAE:
            score = mean_absolute_error(model.y_test, pred)
        elif method == Metrics.XVAL:
            scores = cross_val_score(pipeline, model.x_test, model.y_test, cv=KFold(n_splits=10, shuffle=True))
            report_cross_validation_scores(trial, scores)
            score = scores.mean()

        return score

    def explainResults(self):
        pass
