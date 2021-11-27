import requests
import json
import pandas as pd
import datetime
from pandas.io import gbq
import numpy as np
from google.oauth2 import service_account
import pandas_gbq


def temp_request_api():
    url = "https://covid19.th-stat.com/json/covid19v2/getTimeline.json"
    response = requests.post(url)
    response_result = (json.dumps(response.json(), indent=4))
    # print(response_result)
    data = json.loads(response.content)
    jsonData = data["Data"]
    today = datetime.datetime.today().strftime('%m/%d/%Y')
    for a in jsonData:
        if a['Date'] == today:
            #print("Match")
            #print(dict(a))
            df = pd.DataFrame(a, index=[0])
            print(df)
            if df == None :
                print("No Update")
            else:
                print("Updated")

def covid_all():
    url = "https://covid19.th-stat.com/json/covid19v2/getTimeline.json"
    response = requests.post(url)
    response_result = (json.dumps(response.json(), indent=4))
    data = json.loads(response.content)
    jsonData = data["Data"]
    df = pd.json_normalize(jsonData)
    #print(jsonData)
    # today = datetime.datetime.today().strftime('%m/%d/%Y')
    # for a in jsonData:
    #     if a['Date'] == today:
    #         print("Match")
    #         print(dict(a))
    #         df = pd.DataFrame(a, index=[0])
    #         print(df)
    return (df)


#print(temp_request_api())
#print(covid_all())
credentials = service_account.Credentials.from_service_account_file('datateam.json')
url = "https://covid19.th-stat.com/json/covid19v2/getTimeline.json"
response = requests.post(url)
response_result = (json.dumps(response.json(), indent=4))
print(response_result)
data = json.loads(response.content)
jsonData = data["Data"]
today = datetime.datetime.today().strftime('%m/%d/%Y')
for a in jsonData:
    if a['Date'] == today:
        #print("Match")
        #print(dict(a))
        df = pd.DataFrame(a, index=[0])
        print(df)
        #pandas_gbq.context.credentials = credentials
        #pandas_gbq.to_gbq(df, 'Covid19_THA.covid19_statistic', project_id='datateam-316802', if_exists='append')
        #print("Updated")
    else:
        pass
           