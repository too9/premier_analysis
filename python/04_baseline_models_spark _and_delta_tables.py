# Databricks notebook source
# MAGIC %pip install xgboost
# MAGIC !pip install mlflow --quiet

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='3289544690048618',
  label='Experiment ID'
)


dbutils.widgets.dropdown("outcome","icu",["misa_pt", "multi_class", "death", "icu"])
OUTCOME = dbutils.widgets.get("outcome")

dbutils.widgets.dropdown("demographics", "True", ["True", "False"])
USE_DEMOG = dbutils.widgets.get("demographics")
if USE_DEMOG == "True": DEMOG = True
else: USE_DEMOG = False

dbutils.widgets.dropdown("stratify", "all", ['all', 'death', 'misa_pt', 'icu'])
STRATIFY = dbutils.widgets.get("stratify")

dbutils.widgets.dropdown("experimenting", "False",  ["True", "False"])
EXPERIMENTING = dbutils.widgets.get("experimenting")
if EXPERIMENTING == "True": EXPERIMENTING = True
else: EXPERIMENTING = False

# COMMAND ----------

import mlflow
experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id


# COMMAND ----------

import argparse
import os
import pickle as pkl
from importlib import reload

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import tools.analysis as ta
import tools.preprocessing as tp
import mlflow

# COMMAND ----------

# Setting the globals
#OUTCOME = 'misa_pt'
#USE_DEMOG = True
AVERAGE = 'weighted'
DAY_ONE_ONLY = True
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2022
CHRT_PRFX = ''
LABEL_COLUMN='label'
FEATURES_COLUMN='features'
VALIDATION_COLUMN="validation"

# COMMAND ----------


# Setting the directories and importing the data
# If no args are passed to overwrite these values, use repo structure to construct
# Setting the directories
output_dir = '/dbfs/home/tnk6/premier_output/'
data_dir = '/dbfs/home/tnk6/premier/'

if data_dir is not None:
    data_dir = os.path.abspath(data_dir)

if output_dir is not None:
    output_dir = os.path.abspath(output_dir)

pkl_dir = os.path.join(output_dir, "pkl", "")
stats_dir = os.path.join(output_dir, "analysis", "")
probs_dir = os.path.join(stats_dir, "probs", "")

# COMMAND ----------

# Create analysis dirs if it doesn't exist
[
    os.makedirs(directory, exist_ok=True)
    for directory in [stats_dir, probs_dir, pkl_dir]
]

with open(pkl_dir + CHRT_PRFX + "trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

with open(pkl_dir + "demog_dict.pkl", "rb") as f:
    demog_dict = pkl.load(f)
    demog_dict = {k: v for v, k in demog_dict.items()}

# COMMAND ----------

# Separating the inputs and labels
features = [t[0] for t in inputs]
demog = [t[1] for t in inputs]
cohort = pd.read_csv(os.path.join(output_dir, CHRT_PRFX, 'cohort.csv'))
labels = cohort[OUTCOME]

# Counts to use for loops and stuff
n_patients = len(features)
n_features = np.max(list(vocab.keys()))
n_classes = len(np.unique(labels))
binary = n_classes <= 2

# Converting the labels to an array
y = np.array(labels, dtype=np.uint8)

# COMMAND ----------

# Optionally limiting the features to only those from the first day
# of the actual COVID visit
if DAY_ONE_ONLY:
    features = [l[-1] for l in features]
else:
    features = [tp.flatten(l) for l in features]

# Optionally mixing in the demographic features
if USE_DEMOG:
    new_demog = [[i + n_features for i in l] for l in demog]
    features = [features[i] + new_demog[i] for i in range(n_patients)]
    demog_vocab = {k + n_features: v for k, v in demog_dict.items()}
    vocab.update(demog_vocab)
    n_features = np.max([np.max(l) for l in features])
    all_feats.update({v: v for k, v in demog_dict.items()})

# Converting the features to a sparse matrix
mat = lil_matrix((n_patients, n_features + 1))
for row, cols in enumerate(features):
    mat[row, cols] = 1

# Converting to csr because the internet said it would be faster
X = mat.tocsr()

# Splitting the data; 'all' will produce the same test sample
# for every outcome (kinda nice)
if STRATIFY == 'all':
    outcomes = ['icu', 'misa_pt', 'death']
    strat_var = cohort[outcomes].values.astype(np.uint8)
else:
    strat_var = y

# COMMAND ----------

train, test = train_test_split(range(n_patients),
                               test_size=TEST_SPLIT,
                               stratify=strat_var,
                               random_state=RAND)

# Doing a validation split for threshold-picking on binary problems
train, val = train_test_split(train,
                              test_size=VAL_SPLIT,
                              stratify=strat_var[train],
                              random_state=RAND)

# COMMAND ----------

#
# used for limiting the sample size for testing
#
if EXPERIMENTING == True: 
    ROWS = 1000
    COLS = 100
else:
    ROWS = X.shape[0]
    COLS = X.shape[1]

# COMMAND ----------

#
# when converting sparce/dense matrices to Spark Data Frames, column names are required.#
#
def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names

# COMMAND ----------

#
# If there is enough memory available in the master node,
# this functions proved to be faster
#
def convert_pandas_to_spark_with_vectors(a_dataframe,c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler

    assert isinstance (a_dataframe,  pd.DataFrame)
    assert c_names is not None
    assert len(c_names)>0
    
    number_of_partitions = int(spark.sparkContext.defaultParallelism)*2

    a_rdd = spark.sparkContext.parallelize(a_dataframe.to_numpy(), number_of_partitions)
    
    a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names+[LABEL_COLUMN]) )

    vecAssembler = VectorAssembler(outputCol="features")
    vecAssembler.setInputCols(c_names)
    spark_df = vecAssembler.transform(a_df)

    return spark_df

# COMMAND ----------

#
# If there is NOT enough memory available in the master node,
# this functions proved to be useful
#
def incrementaly_convert_pandas_to_spark_with_vectors(a_dataframe,c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler

    assert isinstance (a_dataframe,  pd.DataFrame)
    assert c_names is not None
    assert len(c_names)>0

    inc=min(10000, a_dataframe.shape[0])
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):

        
        if (i*inc) < a_dataframe.shape[0]:
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
            a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names+LABEL_COLUMN) )

            vecAssembler = VectorAssembler(outputCol="features")
            vecAssembler.setInputCols(c_names)
            a_spark_vector = vecAssembler.transform(a_df)

            if bool == True:
                spark_df = a_spark_vector
                bool = False
            else:
                spark_df = spark_df.union(a_spark_vector)
    
 
    return spark_df

# COMMAND ----------

#
# working around to transform pandas DF to spark DF
#
def pandas_to_spark_via_parquet_files(pDF, c_names, results, index): 
    from pyspark.ml.feature import VectorAssembler
    import time
    
    seconds = time.time()
    
    fileName = "/FileStore/tmp/file"+str(seconds)+".parquet"

    pDF.to_parquet("/dbfs/"+fileName, compression="gzip")  
    sDF=spark.read.parquet(fileName)
    results[index] = VectorAssembler(outputCol=FEATURES_COLUMN)\
                    .setInputCols(c_names)\
                    .transform(sDF).select(LABEL_COLUMN, FEATURES_COLUMN).cache()
    
def convert_pDF_to_sDF_via_parquet_files(list_of_pandas, c_names):
    from threading import Thread

    results = [None] * len(list_of_pandas)
    threads = [None] * len(list_of_pandas)

    for index in range(0,len(threads)):
            threads [index] = Thread(target=pandas_to_spark_via_parquet_files, 
                                     args=(list_of_pandas[index], 
                                           c_names, 
                                           results, 
                                           index))
            threads[index].start()

    for i in range(len(threads)):
        threads[i].join()

    return results


# COMMAND ----------

#
# Since the converstion from Pandas to Spark have been extremelly slow,
# this function allows the converstion to be done in parallel
#
def pandas_to_spark(pDF, c_names, results, index): 
    results[index] = convert_pandas_to_spark_with_vectors(pDF, c_names).select([LABEL_COLUMN,FEATURES_COLUMN]).cache()
        
def convert_pDF_to_sDF(list_of_pandas, c_names):
    from threading import Thread

    results = [None] * len(list_of_pandas)
    threads = [None] * len(list_of_pandas)

    for index in range(0,len(threads)):
            threads [index] = Thread(target=pandas_to_spark, 
                                     args=(list_of_pandas[index], 
                                           c_names, 
                                           results, 
                                           index))
            threads[index].start()

    for i in range(len(threads)):
        threads[i].join()

    return results

# COMMAND ----------

#
# create pandas frames from X
#
c_names = change_columns_names(X)[:COLS]

X_train_pandas = pd.DataFrame(X[train][:ROWS,:COLS].toarray(),columns=c_names)
X_train_pandas[LABEL_COLUMN] = y[train][:ROWS].astype("int")

X_val_pandas = pd.DataFrame(X[val][:ROWS,:COLS].toarray(),columns=c_names)
X_val_pandas[LABEL_COLUMN] = y[val][:ROWS].astype("int")

X_test_pandas = pd.DataFrame(X[test][:ROWS,:COLS].toarray(),columns=c_names)
X_test_pandas[LABEL_COLUMN] = y[test][:ROWS].astype("int")

# COMMAND ----------

results = None
list_of_pandas = [X_train_pandas,X_val_pandas,X_test_pandas]
results = convert_pDF_to_sDF(list_of_pandas,c_names)

X_train_spark = results[0]
X_val_spark   = results[1]
X_test_spark  = results[2]

# COMMAND ----------

#
# .count() forces the cache
#
for i in range(len(results)): 
    print(results[i].count())
    print(results[i].rdd.getNumPartitions())

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS too9_premier_analysis_demo;
# MAGIC drop table IF  EXISTS   too9_premier_analysis_demo.train_data_set;
# MAGIC drop table IF  EXISTS   too9_premier_analysis_demo.val_data_set;
# MAGIC drop table IF  EXISTS   too9_premier_analysis_demo.test_data_set;

# COMMAND ----------

#
# save Spark Data Frames to Delta Tables
# it seems that Spark Frames created from delta tables
# work faster
#
X_train_spark.write.mode("overwrite").format("delta").saveAsTable("too9_premier_analysis_demo.train_data_set")
X_val_spark.write.mode("overwrite").format("delta").saveAsTable("too9_premier_analysis_demo.val_data_set")
X_test_spark.write.mode("overwrite").format("delta").saveAsTable("too9_premier_analysis_demo.test_data_set")


# COMMAND ----------

#
# working around to import data frames from delta tables
# ML algorithms work significally faster
#
from delta.tables import DeltaTable

X_train_dt = spark.table("too9_premier_analysis_demo.train_data_set")
X_val_dt = spark.table("too9_premier_analysis_demo.val_data_set")
X_test_dt = spark.table("too9_premier_analysis_demo.test_data_set")


# COMMAND ----------


### to be used only if the input are spark dataframes
y_val = X_val_dt.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()
y_test = X_test_dt.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()

# COMMAND ----------

#
# Spark ML return predictions as vector of probabilities
# this function return the positive probabilities
#
def get_array_of_probs (predictions_sDF):
    from pyspark.ml.functions import vector_to_array
    import numpy as np

    p = predictions_sDF.select(vector_to_array("probability", "float32").alias("probability")).toPandas()['probability'].to_numpy()
    
    return np.array(list(map(lambda x: x[1], p)))

# COMMAND ----------

#
# this function calculates the statistics from validation and testing predictions
#
def get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name, average=AVERAGE):
    val_gm = ta.grid_metrics(y_val, val_probs)
    cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]
    test_preds = ta.threshold(test_probs, cutpoint)
    stats = ta.clf_metrics(y_test,
                           test_probs,
                           cutpoint=cutpoint,
                           mod_name=mod_name,
                           average=average)
    return stats

# COMMAND ----------

#
# some Spark ML algorithms do not return probabilities
# this function calculates statistics from predictions without probabilities
#
def get_statistics_from_predict(test_predict, y_test, mod_name, average=AVERAGE):
    stats = ta.clf_metrics(y_test,
                           test_predict,
                           mod_name=mod_name,
                           average=average)
    return stats

# COMMAND ----------

#
# this function logs statistics on MLFLow
#
def log_stats_in_mlflow(stats):
    for i in stats:
        if not isinstance(stats[i].iloc[0], str):
            mlflow.log_metric("testing_"+i, stats[i].iloc[0])

# COMMAND ----------

#
# this function logs a few MLFlow parameters about the type of prediction
#
def log_param_in_mlflow():
    mlflow.log_param("average", AVERAGE)
    mlflow.log_param("demographics", USE_DEMOG)
    mlflow.log_param("outcome", OUTCOME)
    mlflow.log_param("stratify", STRATIFY)

# COMMAND ----------

#
# H2O returns predicit probabilities in a different way of Spark ML
# this functions returns the probabilities
#
def get_array_of_probabilities_from_sparkling_water_prediction(predict_sDF):
    p = predict_sDF.select('detailed_prediction').collect()
    probs = list()
    for row in range(len(p)):
        prob = p[row].asDict()['detailed_prediction']['probabilities'][1]
        probs = probs + [prob]
    
    return np.asarray(probs)

# COMMAND ----------

#
# this part user ski-learn 
#

from scipy.sparse import lil_matrix
from sklearn.ensemble import GradientBoostingClassifier as sk_gbc
from sklearn.ensemble import RandomForestClassifier as sk_rfc
from sklearn.linear_model import LogisticRegression as sk_lr
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC as sk_svc

# Loading up some models to try
stats = None
mods = [
    sk_lr(max_iter=5000, multi_class='ovr'),
    sk_rfc(n_estimators=500, n_jobs=-1),
    sk_gbc(),
    sk_svc(class_weight='balanced', max_iter=10000)
]
mod_names = ['lgr', 'rf', 'gbc', 'svm']

# Turning the crank like a proper data scientist
for i, mod in enumerate(mods):
    #
    # add execution parameters to MLFLOW
    #
    mlflow.end_run()
    modelName = mod_names[i]
    run_name=f"sci-learn_{modelName}"

    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    mlflow.log_param("average", AVERAGE)
    mlflow.log_param("demographics", USE_DEMOG)
    mlflow.log_param("outcome", OUTCOME)
    mlflow.log_param("stratify", STRATIFY)
    #
    #
    #
    model_fit = mod.fit(X[train], y[train])
    
    mlflow.sklearn.log_model(model_fit, "model")
    # to make sure model can be found progrmatically, 
    # use "model" as the name of the model
    
    
    mod_name = mod_names[i]
    if DAY_ONE_ONLY:
        mod_name += '_d1'

    if 'predict_proba' in dir(mod):
        if binary:
            val_probs = model_fit.predict_proba(X[val])[:, 1]
            val_gm = ta.grid_metrics(y[val], val_probs)
            cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]
            test_probs = model_fit.predict_proba(X[test])[:, 1]
            test_preds = ta.threshold(test_probs, cutpoint)
            stats = ta.clf_metrics(y[test],
                                   test_probs,
                                   cutpoint=cutpoint,
                                   mod_name=mod_name,
                                   average=AVERAGE)
        else:
            cutpoint = None
            test_probs = model_fit.predict_proba(X[test])
            test_preds = model_fit.predict(X[test])
            stats = ta.clf_metrics(y[test],
                                   test_probs,
                                   mod_name=mod_name,
                                   average=AVERAGE)
    else:
        test_preds = mod.predict(X[test])
        stats = ta.clf_metrics(y[test],
                               test_preds,
                               mod_name=mod_name,
                               average=AVERAGE)
    #
    #
    # add metrics to MLFLow
    #
    log_stats_in_mlflow(stats)


# COMMAND ----------

#
# This part use Spark ML binary classification algoritms#
#
from pyspark.ml.classification import LinearSVC as svc
from pyspark.ml.classification import DecisionTreeClassifier as dtc
from pyspark.ml.classification import GBTClassifier as gbt
from pyspark.ml.classification import RandomForestClassifier as rfc
from pyspark.ml.classification import LogisticRegression as lr

model_class = [lr(maxIter=5000,featuresCol='features',labelCol=LABEL_COLUMN),
              gbt(seed=2022,featuresCol='features',labelCol=LABEL_COLUMN), 
              dtc(seed=2022,featuresCol='features',labelCol=LABEL_COLUMN), 
              rfc (numTrees=500,seed=2022,featuresCol='features',labelCol=LABEL_COLUMN), 
              svc(maxIter=100,featuresCol='features',labelCol=LABEL_COLUMN)]
model_names = ["LogisticRegression",
              "GBTClassifier", 
              "DecisionTreeClassifier", 
              "RandomForestClassifier", 
              "LinearSVC"]
KNOWN_REGRESSORS_THAT_YIELD_PROBABILITIES = ["LogisticRegression", 
                                             "GBTClassifier",
                                             "RandomForestClassifier"]

mlflow.end_run()

for  i in range(len(model_class)):
    modelName = model_names[i]
    run_name="spark_with_delta_tables_"+modelName
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        model = model_class[i]
        model_fit = model.fit(X_train_dt)

        mlflow.spark.log_model(model_fit, "model")
        # to make sure model can be found progrmatically, 
        # use "model" as the name of the model

        predictions_test = model_fit.transform(X_test_dt)

        if modelName in KNOWN_REGRESSORS_THAT_YIELD_PROBABILITIES:
            predictions_val = model_fit.transform(X_val_dt)
            val_probs  = get_array_of_probs (predictions_val)
            test_probs = get_array_of_probs (predictions_test)
            stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name=modelName, average=AVERAGE)
        else:
            y_predict = predictions_test.select('prediction').toPandas()['prediction'].to_numpy()

            stats = get_statistics_from_predict(y_predict, 
                                        y_test, 
                                        str(modelName), 
                                        average=AVERAGE)
        log_stats_in_mlflow(stats)

# COMMAND ----------

#
# This part provides an examples about how to use hyper parameters with Spark ML
#
from pyspark.ml.classification import LogisticRegression as spark_lr
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

mlflow.end_run()
run_name="spark_with_delta_tables_tunned_lr"
with mlflow.start_run(
    run_name=run_name,
    experiment_id=experiment_id,
):
    lr = spark_lr(featuresCol='features',labelCol=LABEL_COLUMN)

    paramGrid = (ParamGridBuilder()
         .addGrid(lr.regParam, [0.01, 0.5, 2.0])
         .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
         .addGrid(lr.maxIter, [100, 500, 1000])
         .build())

    evaluator = BinaryClassificationEvaluator()

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, parallelism=100)

    cvModel = cv.fit(X_train_dt)

    mlflow.spark.log_model(cvModel.bestModel,  "model")
    # to make sure model can be found progrmatically, 
    # use "model" as the name of the model
    
    mlflow.log_param("elasticNetParam", cvModel.bestModel.getElasticNetParam())
    mlflow.log_param("maxIter", cvModel.bestModel.getMaxIter())
    mlflow.log_param("regParam", cvModel.bestModel.getRegParam())

    predictions_test = cvModel.bestModel.transform(X_test_dt)
    predictions_val  = cvModel.bestModel.transform(X_val_dt)
    val_probs  = get_array_of_probs (predictions_val)
    test_probs = get_array_of_probs (predictions_test)
    stats = get_statistics_from_probabilities(val_probs, 
                                              test_probs, 
                                              y_val, 
                                              y_test, 
                                              mod_name="lr", 
                                              average=AVERAGE)

    log_stats_in_mlflow(stats)
    display(stats)

# COMMAND ----------

#
# This park uses XGBoost with Spark Data Frames
# It is provided by Databricks
#
from sparkdl.xgboost import XgboostClassifier as dbr_xgb

mlflow.end_run()
run_name="spark_with_xgboost"
with mlflow.start_run(
    run_name=run_name,
    experiment_id=experiment_id,
):
    
    model = dbr_xgb(missing=0.0, eval_metric='logloss')    
    
    model_fit = model.fit(X_train_dt)
    
    mlflow.spark.log_model(model_fit, "model")
    
    predictions_test = model_fit.transform(X_test_dt)
    predictions_val  = model_fit.transform(X_val_dt)
    
    val_probs  = get_array_of_probs (predictions_val)
    test_probs = get_array_of_probs (predictions_test)
    
    stats = get_statistics_from_probabilities(val_probs, 
                                              test_probs, 
                                              y_val, 
                                              y_test,
                                              mod_name=run_name, 
                                              average=AVERAGE)
    
    log_stats_in_mlflow(stats)
    display(stats)

# COMMAND ----------

#
# This part uses H2O Sparking Water
# Fi1st of all, you must install required libaries
#

!pip install requests
!pip install tabulate
!pip install future
!pip install h2o_pysparkling_3.2

# COMMAND ----------

#
# Import required libraries and start H2O
#
from pysparkling import *

hc = H2OContext.getOrCreate()


# COMMAND ----------

#
# Here you will use H2O XGBoost algorithim
# ATTENTION:
#          For reasons I do not undestand, H2O does not work with Spark Data Frames created from Delta Tables
#
from pysparkling.ml import H2OXGBoostClassifier 

run_name = "SparkingWater_XGBoost"
mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id, 
                 run_name = run_name)

model = H2OXGBoostClassifier(labelCol = LABEL_COLUMN, 
                            stoppingMetric="logloss")

model_fit = model.fit(X_train_dt)

mlflow.spark.log_model(model_fit, "model")
# to make sure model can be found progrmatically, 
# use "model" as the name of the model

prediction_val = model_fit.transform(X_val_dt)
prediction_test = model_fit.transform(X_test_dt)
val_probs  = get_array_of_probabilities_from_sparkling_water_prediction (prediction_val)
test_probs = get_array_of_probabilities_from_sparkling_water_prediction (prediction_test)
stats = get_statistics_from_probabilities(val_probs, 
                                          test_probs, 
                                          y_val, y_test, 
                                          mod_name=run_name, 
                                          average=AVERAGE)

log_stats_in_mlflow(stats)
mlflow.end_run()
display(stats)

# COMMAND ----------

#
# This part uses H2O Deep Learning
#
import mlflow
from pysparkling.ml import H2ODeepLearningClassifier 

X_train_spark = X_train_spark.withColumn("label", X_train_spark.label.cast("string"))
X_val_spark = X_val_spark.withColumn("label", X_val_spark.label.cast("string"))
X_test_spark = X_test_spark.withColumn("label", X_test_spark.label.cast("string"))


run_name = "SparkingWater_DL"

mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id, 
                 run_name = run_name)

model = H2ODeepLearningClassifier (labelCol = LABEL_COLUMN,
                                   stoppingMetric="AUC",
                                   stoppingRounds=10,
                                   activation="RectifierWithDropout",
                                   epochs=1000,
                                   hidden=[1024,1024],
                                   rate=0.001,
                                   l1=0.1,
                                   seed=2022,
                                  splitRatio=.9)

model_fit = model.fit(X_train_dt)

mlflow.spark.log_model(model_fit, "model")
# to make sure model can be found programatically, 
# use "model" as the name of the model

prediction_val = model_fit.transform(X_val_dt)
prediction_test = model_fit.transform(X_test_dt)
val_probs  = get_array_of_probabilities_from_sparkling_water_prediction (prediction_val)
test_probs = get_array_of_probabilities_from_sparkling_water_prediction (prediction_test)
stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name=run_name, average=AVERAGE)

log_stats_in_mlflow(stats)
mlflow.end_run()
display(stats)

# COMMAND ----------

#
# This part use H2O AutoML
#
from pysparkling.ml import H2OAutoMLClassifier

mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id, 
                 run_name = "SparkingWater_AutoMl")

model = H2OAutoMLClassifier(labelCol = LABEL_COLUMN, 
                            maxModels=100, 
                            stoppingMetric="logloss",
                           seed=2022)

model_fit = model.fit(X_train_dt)

bestmodel = automl.getAllModels()[0]
mlflow.spark.log_model(bestmodel,"model")

prediction_val = bestmodel.transform(X_val_dt)
prediction_test = bestmodel.transform(X_test_dt)
val_probs  = get_array_of_probabilities_from_sparkling_water_prediction (prediction_val)
test_probs = get_array_of_probabilities_from_sparkling_water_prediction (prediction_test)
stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name="H2O_sparking_water_AutoMl", average=AVERAGE)

log_stats_in_mlflow(stats)
mlflow.end_run()

# COMMAND ----------


