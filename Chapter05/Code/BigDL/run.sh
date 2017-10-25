#!/bin/bash

SPARK_HOME=/home/ubuntu/software/spark-2.2.0-bin-hadoop2.7
BigDL_HOME=/home/ubuntu/software/BigDL
PYTHON_API_ZIP_PATH=${BigDL_HOME}/bigdl-python.zip
BigDL_JAR_PATH=${BigDL_HOME}/bigdl-SPARK.jar
export PYTHONPATH=${PYTHON_API_ZIP_PATH}:${BigDL_HOME}/conf/spark-bigdl.conf:$PYTHONPATH


${SPARK_HOME}/bin/spark-submit \
      --master spark://ip-172-31-1-246.us-west-2.compute.internal:7077 \
       --driver-cores 5  \
      --driver-memory 5g  \
      --total-executor-cores 16  \
      --executor-cores 8  \
      --executor-memory 10g \
       --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/BigDL-MNIST.py\
       --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
       --jars ${BigDL_JAR_PATH} \
       --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
       --conf spark.executor.extraClassPath=bigdl-SPARK.jar \
       ${BigDL_HOME}/BigDL-MNIST.py


