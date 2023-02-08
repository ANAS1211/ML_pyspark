
#utiliser les commandes linux sur python
import os
#Construire une session Spark nommée spark
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
#pour utiliser la fonction "col"
from pyspark.sql.functions import col
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName("exam").getOrCreate()
spark
#a= str(os.system("ls"))
#print(a, type(a))
#créer un dataframe et afficher un echantillon
df_raw = spark.read.csv('creditcard.csv', header=True)
print(df_raw.columns)
a=df_raw.sample(False, .001, seed = 2).toPandas()
print(len(df_raw.columns))
#afficher le type de variable de la BD
df_raw.printSchema()
#Créer un DataFrame df à partir de df_raw en changeant 
#les colonnes des variables cibles en double et la variable cible, Class, en int
#Afficher le schéma des variables de df
print(df_raw.columns[0:30])
#changer le type des 30 premières colonnes en double
exprs = [col(c).cast("double") for c in df_raw.columns[:30]]
#créer un dataframe en ajoutant les colonnes "exprs"
# changer le type de la colonne CLass en int
df = df_raw.select(df_raw.Class.cast('int'), *exprs)
df.printSchema()
# DataFrame.filter(items=None, like=None, regex=None, axis=None)[source]
# Subset the dataframe rows or columns according to the specified index labels.
# Supprimer les lignes contenant des valeurs manquantes du DataFrame df
isnull=0
for i in range(30):
    isnull= isnull+ df.filter(df[1].isNull()).count()
if isnull:
    print('il y a des valaurs manquates')
else:
    print('il n y pas de valeurs manquantes')
# Créer un rdd rdd_ml séparant la variable à expliquer des features 
# (à mettre sous forme DenseVector)
from pyspark.ml.linalg import DenseVector
# Créer un DataFrame df_ml contenant notre base de données 
# sous deux variables : 'labels' et 'features'
# lambda arguments : expression
rdd_ml = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
print(type(rdd_ml))
df_ml = spark.createDataFrame(rdd_ml, ['label', 'features'])
df_ml.show(5)
# Créer deux DataFrames appelés train et test contenant 
# chacun respectivement 80% et 20% des données
# Créer un classificateur Random Forest appelé clf

# Apprendre le modèle des forêts aléatoires sur l'ensemble d'entraînement
from pyspark.ml.classification import RandomForestClassifier
train, test = df_ml.randomSplit([.8, .2], seed=2)

clf = RandomForestClassifier (labelCol="label",
featuresCol="features",
predictionCol='prediction',seed = 2)
model = clf.fit(train)

# Calculer la précision, accuracy, du modèle entraîné
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predicted = model.transform(test)
predicted.show(10)
evaluator = MulticlassClassificationEvaluator (metricName= 'accuracy', labelCol='label', predictionCol='prediction')
accuracy = evaluator.evaluate(predicted)
print("l'accuracy est de :",accuracy)