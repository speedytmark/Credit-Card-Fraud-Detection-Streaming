# Databricks notebook source
# MAGIC %md
# MAGIC # Travis Mark
# MAGIC ### Assignment 2

# COMMAND ----------

sc=spark.sparkContext
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, IntegerType, TimestampType, DateType
from pyspark.sql import functions as f
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import StringIndexer, Bucketizer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC Load Data

# COMMAND ----------

#Create schema
fraudSchema=StructType([StructField('id', LongType(), True)\
    , StructField('trans_date_trans_time', TimestampType(), True)\
    , StructField('cc_num', LongType(), True)\
    , StructField('merchant', StringType(), True)\
    , StructField('category', StringType(), True)\
    , StructField('amt', DoubleType(), True)\
    , StructField('first', StringType(), True)\
    , StructField('last', StringType(), True)\
    , StructField('gender', StringType(), True)\
    , StructField('street', StringType(), True)\
    , StructField('city', StringType(), True)\
    , StructField('state', StringType(), True)\
    , StructField('zip', StringType(), True)\
    , StructField('lat', DoubleType(), True)\
    , StructField('long', DoubleType(), True)\
    , StructField('city_pop', IntegerType(), True)\
    , StructField('occupation', StringType(), True)\
    , StructField('dob', DateType(), True)\
    , StructField('trans_num', StringType(), True)\
    , StructField('unix_time', IntegerType(), True)\
    , StructField('merch_lat', DoubleType(), True)\
    , StructField('merch_long', DoubleType(), True)\
    , StructField('is_fraud', IntegerType(), True)\
])

#Load Files
df1=spark.read.format('csv').option('header',True).schema(fraudSchema).option('mode', 'dropMalformed').load("dbfs:/FileStore/tables/assignment2/fraudTrain.csv")

df2=spark.read.format('csv').option('header',True).schema(fraudSchema).option('mode', 'dropMalformed').load("dbfs:/FileStore/tables/assignment2/fraudTest.csv")

#Combine into a single data frame
df=df1.union(df2)

#Remove unnecessary columns: id, street, city_pop, unix_time
#These will not add relevant information to the model. Removing city population as a possible data element to pass to model due to the nature of its availability in a real-time fraud scenario.
cleaned_df=df.drop('id', 'street', 'city_pop', 'unix_time')

#Remove the "_fraud" prefex from the merchant name. This prefix is present for all entries, so it is not helpful in building or interpreting the model
cleaned_df=cleaned_df.withColumn('merchant', f.col('merchant').substr(7, 100))

#Split into training and test data sets
train_df, test_df=cleaned_df.randomSplit([0.7,0.3], seed=7)

# COMMAND ----------

# MAGIC %md
# MAGIC The instances of fraud in the data are rare. Ensure the split obtained an even likelihood of fraud in both the training and test data

# COMMAND ----------

train_fraud_per=train_df.filter(f.col('is_fraud')==1).count()/train_df.count()
test_fraud_per=test_df.filter(f.col('is_fraud')==1).count()/test_df.count()
print(f'Percent of fraud transactions in training set: {round(train_fraud_per*100,4)}')
print(f'Percent of fraud transactions in test set: {round(test_fraud_per*100,4)}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Pipeline Stages

# COMMAND ----------

#Feature Engineering
#Parse out the time components of the transaction datetime field. 
train_df=train_df.withColumn('transaction_hour', f.hour('trans_date_trans_time'))
train_df=train_df.withColumn('transaction_minute', f.minute('trans_date_trans_time')) 

#Create an age field using the dob and transaction date fields
train_df=train_df.withColumn('age', f.year('trans_date_trans_time')-f.year('dob'))

#Bucketize age
age_buckets=Bucketizer(splits=[-float('inf'),0,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,float('inf')], inputCol='age', outputCol='age_bin')

#Bucketize the minutes
minutes_buckets=Bucketizer(splits=[-float('inf'),5,10,15,20,25,30,35,40,45,50,55,float('inf')], inputCol='transaction_minute', outputCol='transaction_minute_bin')

#Categorical Encoding
merchant_index=StringIndexer(inputCol='merchant', outputCol='merchant_index')
category_index=StringIndexer(inputCol='category', outputCol='category_index')
gender_index=StringIndexer(inputCol='gender', outputCol='gender_index')
city_index=StringIndexer(inputCol='city', outputCol='city_index')
state_index=StringIndexer(inputCol='state', outputCol='state_index')
zip_index=StringIndexer(inputCol='zip', outputCol='zip_index')
occupation_index=StringIndexer(inputCol='occupation', outputCol='occupation_index')

#Label index
label_index=StringIndexer(inputCol='is_fraud', outputCol='label').fit(df)

#Vectorize inputs
features_full=['amt', 'age_bin', 'transaction_hour', 'transaction_minute_bin', 'merchant_index', 'category_index', 'gender_index', 'city_index', 'state_index', 'zip_index', 'lat', 'long', 'occupation_index', 'merch_lat', 'merch_long']

vectorize=VectorAssembler(inputCols=features_full, outputCol='features')

#Random Forest model parameters
rf=RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=51, featureSubsetStrategy='sqrt', maxBins=1000, seed=36)

#Assemble stages
steps=[age_buckets,minutes_buckets,merchant_index,category_index,gender_index,city_index,state_index,zip_index,occupation_index,label_index,vectorize,rf]

#Make pipeline
pipeline=Pipeline(stages=steps)

# COMMAND ----------

# MAGIC %md
# MAGIC # Fit Models

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full Model
# MAGIC Using all features, train a random forest model. With a larger number of features limit the feature subset selection to choose from the square root of the number of features.

# COMMAND ----------

rf_model=pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### X-Small model 
# MAGIC Remove extra features. Limit to Age bin, transaction hour, category, gender, and amt.
# MAGIC With limited features, allow more variety in feature subset selection

# COMMAND ----------

#Random Forest small model features
features_xs=['amt', 'age_bin', 'transaction_hour', 'merchant_index', 'category_index', 'gender_index']
vectorize_xs=VectorAssembler(inputCols=features_xs, outputCol='features')

#Random Forest small model parameters
rf_xs=RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=51, featureSubsetStrategy='auto', maxBins=1000, seed=36)

#Assemble stages
steps_xs=[age_buckets,merchant_index,category_index,gender_index,label_index,vectorize_xs,rf_xs]

#Make pipeline
pipeline_xs=Pipeline(stages=steps_xs)

#Fit
rf_xs_model=pipeline_xs.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### XG Boost
# MAGIC Gradient Boosted Extra Trees model. Using the limited set of features and 30 iterations.

# COMMAND ----------

#XGBoost model parameters
gbt=GBTClassifier(featuresCol='features', labelCol='label', maxIter=30, maxBins=1000, seed=36)

#Assemble stages
steps_gbt=[age_buckets,merchant_index,category_index,gender_index,label_index,vectorize_xs,gbt]

#Make pipeline
pipeline_gbt=Pipeline(stages=steps_gbt)

#Fit
gbt_model=pipeline_gbt.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate models
# MAGIC Perform predictions of the training data set for each model to select the best performing one to use on the test data.

# COMMAND ----------

#Run the full model on the training data set for validation
valid=rf_model.transform(train_df).persist()

#Run the small model on the training data set for validation
valid_xs=rf_xs_model.transform(train_df).persist()

#Run the Gradient Boost model on the training data set for validation
valid_gbt=gbt_model.transform(train_df).persist()

#Instantiate a binary classification evaluator to calculate ROC on each model
evaluator=BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')

#Calculate and compare the area under the ROC of predictions for each model
print(f'Full Model Area Under ROC curve of predictions on training data: {round(evaluator.evaluate(valid),4)}')

print(f'Small Model Area Under ROC curve of predictions on training data: {round(evaluator.evaluate(valid_xs),4)}')

print(f'Area Under ROC curve of predictions on training data: {round(evaluator.evaluate(valid_gbt),4)}')

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate and compare the accuracy, precision, recall, specificity for each model on the validation data. Also compare the false positive rate and false negative rate for each.

# COMMAND ----------

#Calculations for full model
full_tp=valid.filter((f.col('prediction')==1) & (f.col('label')==1)).count()
full_fp=valid.filter((f.col('prediction')==1) & (f.col('label')==0)).count()
full_tn=valid.filter((f.col('prediction')==0) & (f.col('label')==0)).count()
full_fn=valid.filter((f.col('prediction')==0) & (f.col('label')==1)).count()

print('Full Model Metrics:')
print(f"Full Model Accuracy: {round((full_tp+full_tn)/valid.count(),5)}")
print(f"Full Model Precision: {round(full_tp/(full_tp+full_fp),5)}")
print(f"Full Model Recall: {round(full_tp/(full_tp+full_fn),5)}")
print(f"Full Model Specificity: {round(full_tn/(full_tn+full_fp),5)}")
print(f"Full Model False Negative Rate: {1-round(full_tp/(full_tp+full_fn),5)}")
print(f"Full Model False Positive Rate: {1-round(full_tn/(full_tn+full_fp),5)}")
print(f"Full Model F1 Score: {round(full_tp/(full_tp+(.5*(full_fp+full_fn))),5)}")

#Calculations for small model
xs_tp=valid_xs.filter((f.col('prediction')==1) & (f.col('label')==1)).count()
xs_fp=valid_xs.filter((f.col('prediction')==1) & (f.col('label')==0)).count()
xs_tn=valid_xs.filter((f.col('prediction')==0) & (f.col('label')==0)).count()
xs_fn=valid_xs.filter((f.col('prediction')==0) & (f.col('label')==1)).count()

print('\nSmall Model Metrics:')
print(f"Small Model Accuracy: {round((xs_tp+xs_tn)/valid_xs.count(),5)}")
print(f"Small Model Precision: {round(xs_tp/(xs_tp+xs_fp),5)}")
print(f"Small Model Recall: {round(xs_tp/(xs_tp+xs_fn),5)}")
print(f"Small Model Specificity: {round(xs_tn/(xs_tn+xs_fp),5)}")
print(f"Small Model False Negative Rate: {1-round(xs_tp/(xs_tp+xs_fn),5)}")
print(f"Small Model False Positive Rate: {1-round(xs_tn/(xs_tn+xs_fp),5)}")
print(f"Small Model F1 Score: {round(xs_tp/(xs_tp+(.5*(xs_fp+xs_fn))),5)}")

#Calculations for GBT model
gbt_tp=valid_gbt.filter((f.col('prediction')==1) & (f.col('label')==1)).count()
gbt_fp=valid_gbt.filter((f.col('prediction')==1) & (f.col('label')==0)).count()
gbt_tn=valid_gbt.filter((f.col('prediction')==0) & (f.col('label')==0)).count()
gbt_fn=valid_gbt.filter((f.col('prediction')==0) & (f.col('label')==1)).count()

print('\nGBT Model Metrics:')
print(f"GBT Model Accuracy: {round((gbt_tp+gbt_tn)/valid_gbt.count(),5)}")
print(f"GBT Model Precision: {round(gbt_tp/(gbt_tp+gbt_fp),5)}")
print(f"GBT Model Recall: {round(gbt_tp/(gbt_tp+gbt_fn),5)}")
print(f"GBT Model Specificity: {round(gbt_tn/(gbt_tn+gbt_fp),5)}")
print(f"GBT Model False Negative Rate: {1-round(gbt_tp/(gbt_tp+gbt_fn),5)}")
print(f"GBT Model False Positive Rate: {1-round(gbt_tn/(gbt_tn+gbt_fp),5)}")
print(f"GBT Model F1 Score: {round(gbt_tp/(gbt_tp+(.5*(gbt_fp+gbt_fn))),5)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate model on test data

# COMMAND ----------

#Feature Engineering
#Parse out the time components of the transaction datetime field. 
test_df=test_df.withColumn('transaction_hour', f.hour('trans_date_trans_time'))

#Create an age field using the dob and transaction date fields
test_df=test_df.withColumn('age', f.year('trans_date_trans_time')-f.year('dob'))

#Run the model on the test data set for predictions
preds=gbt_model.transform(test_df).persist()
preds.select('trans_num','merchant','amt','label','probability','prediction').show(10)
#Show a sample of fraud flagged transactions
preds.where(f.col('prediction')==1).select('trans_num','merchant','amt','label','probability','prediction').show(10)

#Calculate precision of predictions on training data
test_evaluator=BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
precision=BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderPR')

print(f'Area Under ROC curve of predictions on test data: {round(test_evaluator.evaluate(preds),4)}')
print(f'Area Under Precision curve of predictions on test data: {round(precision.evaluate(preds),4)}')

# COMMAND ----------

#Calculations for test data
test_tp=preds.filter((f.col('prediction')==1) & (f.col('label')==1)).count()
test_fp=preds.filter((f.col('prediction')==1) & (f.col('label')==0)).count()
test_tn=preds.filter((f.col('prediction')==0) & (f.col('label')==0)).count()
test_fn=preds.filter((f.col('prediction')==0) & (f.col('label')==1)).count()

print('\nTest Date Metrics:')
print(f"Accuracy: {round((test_tp+test_tn)/preds.count(),5)}")
print(f"Precision: {round(test_tp/(test_tp+test_fp),5)}")
print(f"Recall: {round(test_tp/(test_tp+test_fn),5)}")
print(f"Specificity: {round(test_tn/(test_tn+test_fp),5)}")
print(f"False Negative Rate: {1-round(test_tp/(test_tp+test_fn),5)}")
print(f"False Positive Rate: {1-round(test_tn/(test_tn+test_fp),5)}")
print(f"F1 Score: {round(test_tp/(test_tp+(.5*(test_fp+test_fn))),5)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming

# COMMAND ----------

#Partition test data 
stream_test_df=test_df.repartition(numPartitions=500).orderBy('trans_date_trans_time').persist()
#Save test files based on partitions
dbutils.fs.mkdirs("/FileStore/tables/assignment2/batches")
dbutils.fs.rm("/FileStore/tables/assignment2/batches/", True)
stream_test_df.write.format('csv').option('header', True).save("/FileStore/tables/assignment2/batches")

# COMMAND ----------

#Setup reduced feature list schema
streamSchema=StructType([StructField('trans_date_trans_time', TimestampType(), True)\
    , StructField('cc_num', LongType(), True)\
    , StructField('merchant', StringType(), True)\
    , StructField('category', StringType(), True)\
    , StructField('amt', DoubleType(), True)\
    , StructField('first', StringType(), True)\
    , StructField('last', StringType(), True)\
    , StructField('gender', StringType(), True)\
    , StructField('city', StringType(), True)\
    , StructField('state', StringType(), True)\
    , StructField('zip', StringType(), True)\
    , StructField('lat', DoubleType(), True)\
    , StructField('long', DoubleType(), True)\
    , StructField('occupation', StringType(), True)\
    , StructField('dob', DateType(), True)\
    , StructField('trans_num', StringType(), True)\
    , StructField('merch_lat', DoubleType(), True)\
    , StructField('merch_long', DoubleType(), True)\
    , StructField('is_fraud', IntegerType(), True)\
    , StructField('transaction_hour', IntegerType(), True)\
    , StructField('transaction_minute', IntegerType(), True)\
    , StructField('age', IntegerType(), True)\
])

# COMMAND ----------

#Streamming With Data Prep
#Read in one file at a time
sourceStream = spark.readStream.format('csv').option('header', True).schema(streamSchema).option('maxFilesPerTrigger', 1).load("dbfs:///FileStore/tables/assignment2/batches")

#Feature Engineering
#Parse out the time components of the transaction datetime field. 
sourceStream=sourceStream.withColumn('transaction_hour', f.hour('trans_date_trans_time'))
#Create an age field using the dob and transaction date fields
sourceStream=sourceStream.withColumn('age', f.year('trans_date_trans_time')-f.year('dob'))

#Apply pipeline and trained model to new streamming data
preds_stream=rf_model.transform(sourceStream)

#Create a fraud alerts stream
fraud_alerts=preds_stream.filter(f.col('prediction')==1)

#Create sink and display fraud transactions as they are identified
sinkStream=fraud_alerts.writeStream.outputMode('append').format('memory').queryName('fraud_alerts').start()

# COMMAND ----------

#Review results
#Create query
current=spark.sql("SELECT * FROM fraud_alerts")
print(current.select('trans_num', 'trans_date_trans_time', 'first', 'last', 'merchant', 'amt', 'probability').orderBy(f.col('trans_date_trans_time').desc()).show())

print(f'Number of fraud flags: {current.count()}')
