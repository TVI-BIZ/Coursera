
# coding: utf-8

# This is the third assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# Just execute all cells one after the other and you are done - just note that in the last one you must update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"
# 

# In[1]:


credentials_1 = {'password':"""4b5403df0d792637f53845b5b93c089d8fc7f225ba85306f0deeaa4518df1804""",
                 'custom_url':'https://ab28a05d-e0f8-43a2-9930-cfa9e2737b8f-bluemix:4b5403df0d792637f53845b5b93c089d8fc7f225ba85306f0deeaa4518df1804@ab28a05d-e0f8-43a2-9930-cfa9e2737b8f-bluemix.cloudant.com',
                 'username':'ab28a05d-e0f8-43a2-9930-cfa9e2737b8f-bluemix' }



# Let's create a SparkSession object and put the Cloudant credentials into it

# In[2]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_1['custom_url'].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .getOrCreate()


# Now it’s time to have a look at the recorded sensor data. You should see data similar to the one exemplified below….
# 

# In[3]:


df=spark.read.load('shake_classification', "org.apache.bahir.cloudant")

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# Let’s check if we have balanced classes – this means that we have roughly the same number of examples for each class we want to predict. This is important for classification but also helpful for clustering

# In[4]:


spark.sql("SELECT count(class), class from df group by class").show()


# Let's create a VectorAssembler which consumes columns X, Y and Z and produces a column “features”
# 

# In[5]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["X","Y","Z"],
                                  outputCol="features")


# Please insatiate a clustering algorithm from the SparkML package and assign it to the clust variable. Here we don’t need to take care of the “CLASS” column since we are in unsupervised learning mode – so let’s pretend to not even have the “CLASS” column for now – but it will become very handy later in assessing the clustering performance. PLEASE NOTE – IN REAL-WORLD SCENARIOS THERE IS NO CLASS COLUMN – THEREFORE YOU CAN’T ASSESS CLASSIFICATION PERFORMANCE USING THIS COLUMN 
# 
# 

# In[9]:


from pyspark.ml.clustering import KMeans 
from pyspark.ml import Pipeline

clust = KMeans().setK(13).setSeed(1)
pypeline = Pipeline (stages=[vectorAssembler,clust])


# Let’s train...
# 

# In[10]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, clust])
model = pipeline.fit(df)


# ...and evaluate...

# In[11]:


prediction = model.transform(df)
prediction.show()


# In[12]:


prediction.createOrReplaceTempView('prediction')
spark.sql('''
select max(correct)/max(total) as accuracy from (

    select sum(correct) as correct, count(correct) as total from (
        select case when class != prediction then 1 else 0 end as correct from prediction 
    ) 
    
    union
    
    select sum(correct) as correct, count(correct) as total from (
        select case when class = prediction then 1 else 0 end as correct from prediction 
    ) 
)
''').rdd.map(lambda row: row.accuracy).collect()[0]


# If you reached at least 55% of accuracy you are fine to submit your predictions to the grader. Otherwise please experiment with parameters setting to your clustering algorithm, use a different algorithm or just re-record your data and try to obtain. In case you are stuck, please use the Coursera Discussion Forum. Please note again – in a real-world scenario there is no way in doing this – since there is no class label in your data. Please have a look at this further reading on clustering performance evaluation https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment
# 

# In[13]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[14]:


prediction= prediction.repartition(1)
prediction.write.json('a2_m3.json')


# In[15]:


import zipfile

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('a2_m3.json.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('a2_m3.json', zipf)
zipf.close()


# In[16]:


get_ipython().system(u'base64 a2_m3.json.zip > a2_m3.json.zip.base64')


# In[17]:


from rklib import submit
key = "pPfm62VXEeiJOBL0dhxPkA"
part = "EOTMs"
email = "###"
secret = "####"

with open('a2_m3.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)

