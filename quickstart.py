import os.path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import numpy as np
import json
import re
import pandas as pd
from IPython.display import display
import clusteringfunctions as cf
from sklearn.cluster import KMeans
from datetime import timedelta
from credcollect import collectcreds
import time

def main():
  """Shows basic usage of the Gmail API.
  Lists the user's Gmail labels.
  """
  creds = collectcreds()
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.

  try:
    print('The process may take 1-2 minutes to complete')
    starttime = time.perf_counter()
    # Call the Gmail API
    service = build("gmail", "v1", credentials=creds)
    # define # of emails to process in list
    # default to 1k emails from inbox, adjust as needed
    numMessages = 1000
    df = getPreData(service, numMessages)
    results = performKMeans(df)    
    data = postprocessArray(results)
    listOfAddressResult = data.loc[(data['Label'] == 1.0) & (data['Proportion'] != 1.0)][['Email', 'Unread', 'Total']]
    print(listOfAddressResult)
    duration = timedelta(seconds = time.perf_counter()-starttime)
    print('Job took: ', duration)

  except HttpError as error:
    print(f"An error occurred: {error}")

'''
Input: df - dataframe holding message data
Return: NumPy Array
Desc: performs k-means clustering and returns a numpy array with the group labels
'''
def performKMeans(df):
  data = cf.df_to_np(df)
  emailAddresses = df.index
  emailAddresses = emailAddresses.to_numpy().reshape(-1, 1)
  kmean = cf.kmeans(data, False)
  kmean_labels = kmean.labels_.reshape(-1, 1)
  data = cf.getTable(data, kmean_labels, False)
  data = np.concatenate((emailAddresses, data), axis=1)
  return data

'''
Input: service - main Gmail API service object
numResults - # of messages to search for
Return: Pandas dataframe
Desc: collects a list of message IDs from Gmail API and calls preprocessing functions
'''
def getPreData(service, numResults):
  results = service.users().messages().list(userId="me", maxResults=numResults).execute()
  dataArr = getAddressesAndLabels(service, results)
  df = preprocessArray(dataArr)
  return df

'''
Input: service - main Gmail API service object
results - List of message IDs in user's inbox (python dictionary)
Return: Python dict() of email address -> (# of unread emails, # of total emails)
Desc: iterates over the list of message IDs and collects the data for the dictionary
'''
def getAddressesAndLabels(service, results):
  ret = dict()
  timer = 0
  unread = False
  for res in results['messages']:
    timer += 1
    emailGet = service.users().messages().get(userId="me", id=res['id']).execute()
    labelIDs = emailGet['labelIds']
    headersList = emailGet['payload']['headers']
    unread = True if "UNREAD" in labelIDs else False
    addList = [x for x in headersList if x['name'] == "From"]
    regAddress = re.search("<.*>", str(addList[0]['value']))
    if (regAddress == None):
      continue
    address = regAddress.group(0)
    if address not in ret:
      ret[address] = [1 if unread is False else 0, 1]
    else:
        ret[address][0] = ret.get(address)[0] + (1 if unread is False else 0)
        ret[address][1] = ret.get(address)[1] + 1
    if (timer >= 200):
      timer = 0
      # used to prevent overexecution of Gmail API
      time.sleep(1)
  return ret    

'''
Input: data - python dictionary that maps email address of a sender -> (# of unread emails, # of total emails); acquired from getAddressesAndLabels
Return: boolean stating whether preprocessing was complete
Desc: creates a Pandas dataframe from the input data, then creates a new column storing the proportion of unread emails to the total # of emails.
Creates a new csv to perform k-means clustering on
'''
def preprocessArray(data):
  df = pd.DataFrame.from_dict(data, orient='index')
  df.columns = ['Unread', 'Total']
  df['Proportion'] = df['Unread'].div(df['Total'])
  df['Proportion'] = df['Proportion'].round(3)
#   df.to_csv('out2.csv')
#   display(df)
  return df
  
'''
unused for now
'''
def postprocessArray(data):
  df = pd.DataFrame(data, columns=['Email','Unread','Total','Proportion', 'Label'])
  return df

if __name__ == "__main__":
  main()


