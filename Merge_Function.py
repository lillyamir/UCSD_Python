#!/usr/bin/env python
# coding: utf-8

# This function will:
# 
# - Take x datasets (starting w/ 2 in this trial run)
# - Create an ID/Time variable for the survey/participant
# - Create N/A's for columns that exist in one survey and not the other
# - Not drop missing data
# - Takes in a list of CSV files in one version and a folder/relative path to read the CSV in another.

# In[35]:


import pandas as pd
import os


# Version 1: Take in a list of CSVs

# In[32]:


def merge_csv(lst):
    #Warning message
    if type(lst) != list:
        return "Must pass list of .csv files"
    
    #Create a place for the dfs to go
    dfs = []
    for file in lst:
        dfs.append(pd.read_csv(file))
      
    #Create ID column in each so we keep track after merging
    
    for survey_number, df in enumerate(dfs, start=1):
        #Create a unique ID that indicates survey number and row number
        df["id"] = [f"{survey_number}-{i+1}" for i in range(len(df))]
        # Move 'id' column to leftmost
        col = df.pop('id')
        df.insert(0, col.name, col)
        
        
    #Merge them together
    
    data = pd.concat(dfs, ignore_index = True)
    
    return data


# Version 2: Take in a directory of CSVs as input

# In[40]:


def merge_directory(path):
    #Account for people who need to learn to read directions
    if type(path) != str:
        return "Must pass file path"
    
    files = os.listdir(path)
    # Filter for CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    # Full path adjustment
    csv_files = [os.path.join(path, file) for file in csv_files]
    
    #Create a place for the dfs to go
    dfs = []
    for file in csv_files:
        dfs.append(pd.read_csv(file))
      
    #Create ID column in each so we keep track after merging
    
    for survey_number, df in enumerate(dfs, start=1):
        #Create a unique ID that indicates survey number and row number
        df["id"] = [f"{survey_number}-{i+1}" for i in range(len(df))]
        # Move 'id' column to leftmost
        col = df.pop('id')
        df.insert(0, col.name, col)
        
        
    #Merge them together
    
    data = pd.concat(dfs)
    
    return data

