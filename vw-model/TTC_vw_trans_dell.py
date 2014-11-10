# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:02:58 2014

@author: ivan
"""
import pandas as pd
import numpy as np
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from csv import DictReader
import math
from glob import glob

# Data locations
loc_train = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/train.csv"
loc_test = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/test.csv"
loc_labels = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/trainLabels.csv"
loc_best = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/preds/submission_D22_L011_y33_y6_y12.csv" # best submission

loc_model_prefix = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/model"
loc_preds_prefix = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/preds"

# Will be created
loc_test_vw = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/test.vw"
loc_train_vw = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/train_y10.vw"
loc_train_vw_temp = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/train_yn_temp.vw" # used for relabelling

loc_kaggle_submission = "C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/preds/submission_D22_L011_y33_y6_y12_y7.csv"

def load_data(loc_csv, nrows=0):
  print("\nLoading data at %s shaped:"%loc_csv)
  if nrows > 0:
    df = pd.read_csv(loc_csv, nrows=nrows)
  else:
    df = pd.read_csv(loc_csv)
  print(df.shape)
  return df

def to_vw(loc_csv, loc_out, y, y_nr=33, train=True):
  print("\nConverting %s"%loc_csv)
  with open(loc_out,"wb") as outfile:
    distribution = 0
    
    for linenr, row in enumerate( DictReader(open(loc_csv,"rb")) ):
      hash_features = ["x35","x91","x94","x95","x34","x4","x65","x64","x61","x3"]
      yes_no_features = ["x92","x93","x101","x103","x130","x102","x10","x11","x12","x13","x14","x25","x24","x26","x32","x33","x30","x31","x141","x140","x142","x45","x44","x43","x42","x41","x2","x1","x55","x56","x57","x129","x128","x127","x126","x105","x63","x62","x87","x86","x85","x116","x117","x115","x104","x74","x75","x72","x73","x71"]
      pos_features = ["x23","x22","x113","x114","x53","x54","x138","x139"]
      float_features = ["x70","x77","x96","x97","x98","x99","x107","x135","x100","x137","x132","x19","x16","x29","x28","x36","x37","x38","x39","x122","x144","x145","x47","x40","x110","x119","x60","x120","x121","x123","x124","x125","x59","x52","x50","x7","x6","x8","x9","x40","x144","x145","x122","x39","x38","x37","x36"]
      
      n_h = ""
      n_b = ""
      n_p = ""
      n_f = ""
      n_r = ""
      
      for k in row:
        if k is not "id":
          if k in hash_features:
            n_h += " %s_%s"%(k,row[k])
          elif k in yes_no_features:
            n_b += " %s_%s"%(k,row[k])
          elif k in pos_features:
            n_p += " %s_%s"%(k,row[k])
          elif k in float_features and row[k] is not "":
            n_f += " %s:%s"%(k,row[k])
          elif k in float_features and row[k] is "":
            n_f += " %s_%s"%(k,row[k])
          else:
            n_r += " %s_%s"%(k,row[k])
            
      if train:
        label = y[linenr][y_nr-1]
        
        if label == 1:
          distribution += 1
        else:
          label = -1
      else:
        label = 1

      id = row["id"]
      outfile.write("%s '%s |h%s |b%s |p%s |f%s |r%s\n"%(label,id,n_h,n_b,n_p,n_f,n_r) )

      if linenr % 10000 == 0:
        print("%s\t%s"%(linenr,distribution))
    print(distribution)

def relabel_vw(loc_vw, loc_out, loc_labels, y, y_i = 0):
  print("Relabelling to dataset %s..."%loc_out)
  start = datetime.now()
  
  with open(loc_out,"wb") as outfile:
    for e, line in enumerate( open( loc_vw, "rb") ):
      if y[e][y_i-1] == 0:
        new_id = -1
      else:
        new_id = 1
      outfile.write( "%s %s\n"%(new_id," ".join(line.strip().split()[1:])) )
  print("\ncompleted in :( %s\n"%(str(datetime.now()-start)))    
    
def sigmoid(x):
  return 1 / (1 + math.exp(-x))    
    
def to_kaggle(loc_preds, loc_best_sub, loc_out_sub, y_nr):    
  preds = {}  
  for e, line in enumerate( open(loc_preds,"rb") ):
    preds[line.strip().split()[1]] = sigmoid(float(line.strip().split()[0]))

  with open(loc_out_sub,"wb") as outfile:  
    for e, line in enumerate( open(loc_best_sub,"rb") ):
      row = line.strip().split(",")
      if e == 0:
        outfile.write(line)
      elif "y"+str(y_nr)+"," not in line:
        outfile.write(line)
      else:
        outfile.write("%s,%s\n"%(row[0],preds[row[0].replace("_y"+str(y_nr),"")]))
  print("Finished writing Kaggle submission: %s"%loc_out_sub)
        
if __name__ == "__main__":
  #Load labels, remove the id
  y = load_data(loc_labels)
  y = np.array(y.drop("id", axis=1))
  print(y.shape)
  print(np.sum(y, axis=0))
  
  #Create train set for label y33, and a test set with dummy labels
  to_vw(loc_train, loc_train_vw, y, y_nr=10, train=True)
  to_vw(loc_test, loc_test_vw, y, train=False)
  
  #Train and test VW now
  
  #Add the VW predictions to our best submission file
  to_kaggle("C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Tradeshift-Text-Classification/preds/preds_y7.p.txt",loc_best, loc_kaggle_submission, y_nr=7)
