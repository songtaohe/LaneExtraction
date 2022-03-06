import sys 
from subprocess import Popen 

Popen("mkdir dataset_training", shell=True).wait()
#Popen("mkdir dataset_evaluation", shell=True).wait()

Popen("python3 hdmapeditor/create_dataset_for_training.py ../dataset/regions.json ../dataset/ dataset_training/", shell=True).wait()
Popen("python3 hdmapeditor/create_dataset_for_training_vectors.py ../dataset/regions.json ../dataset/ dataset_training/", shell=True).wait()
