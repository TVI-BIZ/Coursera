
# coding: utf-8

# This is the second assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# 
# Just execute all cells one after the other and you are done - just note that in the last one you have to update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"

# In[1]:


credentials_1 = {'password':"""4b5403df0d792637f53845b5b93c089d8fc7f225ba85306f0deeaa4518df1804""",
                 'custom_url':'https://ab28a05d-e0f8-43a2-9930-cfa9e2737b8f-bluemix:4b5403df0d792637f53845b5b93c089d8fc7f225ba85306f0deeaa4518df1804@ab28a05d-e0f8-43a2-9930-cfa9e2737b8f-bluemix.cloudant.com',
                 'username':'ab28a05d-e0f8-43a2-9930-cfa9e2737b8f-bluemix' }




# Let's create a SparkSession object and put the Cloudant credentials into it

# In[13]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_1['custom_url'].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .getOrCreate()


# Now it’s time to have a look at the recorded sensor data. You should see data similar to the one exemplified below….
# 

# In[28]:


df=spark.read.load('shake_classification', "org.apache.bahir.cloudant").withColumnRenamed('CLASS', 'label')
#df = spark.read.load('shake_classification', "org.apache.bahir.cloudant")

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# Please create a VectorAssembler which consumed columns X, Y and Z and produces a column “features”
# 

# In[29]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ["X","Y","Z"], outputCol= "features")


# Please insatiate a classifier from the SparkML package and assign it to the classifier variable. Make sure to either
# 1.	Rename the “CLASS” column to “label” or
# 2.	Specify the label-column correctly to be “CLASS”
# 

# In[53]:


from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier 
from pyspark.ml.feature import StringIndexer


#indexer = StringIndexer(inputCol='label',outputCol='label')
#normalizer = Normalizer(inputCol= 'features', outputCol='features_norm',p=1.0)
#classifier = LogisticRegression(maxIter=20,regParam=0.3,elasticNetParam=0.8)
classifier = DecisionTreeClassifier(labelCol='label',featuresCol='features')

#pipeline = Pipeline(stages=[vectorAssembler,classifier])
#model = pipeline.fit(df_train)


# Let’s train and evaluate…
# 

# In[59]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, classifier])


# In[60]:


model = pipeline.fit(df)


# In[61]:


prediction = model.transform(df)


# In[62]:


prediction.show()


# In[63]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("label")
    
binEval.evaluate(prediction) 


# If you are happy with the result (I’m happy with > 0.55) please submit your solution to the grader by executing the following cells, please don’t forget to obtain an assignment submission token (secret) from the Courera’s graders web page and paste it to the “secret” variable below, including your email address you’ve used for Coursera. (0.55 means that you are performing better than random guesses)
# 

# In[64]:


get_ipython().system(u'rm -Rf a2_m2.parquet')


# In[65]:


prediction = prediction.repartition(1)
prediction.write.json('a2_m2.json')


# In[ ]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[66]:


import zipfile

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('a2_m2.json.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('a2_m2.json', zipf)
zipf.close()


# In[67]:


get_ipython().system(u'base64 a2_m2.json.zip > a2_m2.json.zip.base64')


# In[70]:


from rklib import submit
key = "J3sDL2J8EeiaXhILFWw2-g"
part = "G4P6f"
email = "###"
secret = "###"

with open('a2_m2.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)

