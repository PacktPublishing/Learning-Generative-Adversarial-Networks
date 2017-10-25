1. First download the BigDL distribution from the following website-

https://bigdl-project.github.io/master/#release-download/

2. Extract the zip file.

unzip dist-spark-<version>-scala-<version>-linux64-0.2.0-dist.zip

For example
unzip dist-spark-2.1.1-scala-2.11.8-linux64-0.2.0-dist.zip



3. Copy the bigdl jar and python zip files from lib folder inside the extract and to current location.


4. Edit run.sh file based on your Spark home and BigDl home path along with the jar and zip version of BigDL packages.

5. Finally execute the run.sh file to submit deep learning (lenet) job to your running spark cluster.

./runs.sh





