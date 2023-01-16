###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
#import ConfigParser
import configparser
import tensorflow as tf 

#config file to read from
#config = ConfigParser.RawConfigParser()
config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print("Error: " + e)

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results
result_dir = name_experiment
print ("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print( "Dir already existing")
elif sys.platform=='win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' +result_dir)

print( "copy the configuration file in the results folder")
if sys.platform=='win32':
    os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')
else:
    os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

# run the experiment
if nohup:
    print( "\n2. Run the training on GPU with nohup")
    os.system(run_GPU +' nohup python3.8 -u ./example_NN.py > ' +'./'+name_experiment+'/'+name_experiment+'.nohup')
else:
    print( "\n2. Run the training on GPU (no nohup)")
    os.system(run_GPU +' python3.8 ./example_NN.py')

#Prediction/testing is run with a different script
