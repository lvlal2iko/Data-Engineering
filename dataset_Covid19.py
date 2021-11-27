import pandas as pd
from pandas.io import gbq
import numpy as np
from google.oauth2 import service_account
import pandas_gbq
import requests
import json
from requests.sessions import default_headers

credentials = service_account.Credentials.from_service_account_file('datateam.json')	


def convert_format_time(a):
    b = pd.to_datetime(a['Date'])
    a['Date'] = b.dt.strftime("%d-%m-%y")
    return a
    
def get_vaccine() :
    url = "https://raw.githubusercontent.com/wiki/djay/covidthailand/cases_briefings.csv"
    df = pd.read_csv(url)
    covid = df.rename(columns={'Cases Area Prison': 'Cases_Area_Prison',
                               'Cases Asymptomatic': 'Cases_Asymptomatic',
                               'Cases Imported':'Cases_Imported',
                               'Cases Local Transmission': 'Cases_Local_Transmission',
                               'Cases Proactive': 'Cases_Proactive',
                               'Cases Symptomatic': 'Cases_Symptomatic',
                               'Cases Walkin': 'Cases_Walkin',
                               'Deaths Age Max' : 'Deaths_Age_Max',
                               'Deaths Age Median' : 'Deaths_Age_Median',
                               'Deaths Age Min': 'Deaths_Age_Min',
                               'Deaths Comorbidity None':'Deaths_Comorbidity_None',
                               'Deaths Female':'Deaths_Female',
                               'Deaths Male' : 'Deaths_Male',
                               'Deaths Risk Family':'Deaths_Risk_Family',
                               'Hospitalized Field':'Hospitalized_Field',
                               'Hospitalized Hospital' : 'Hospitalized_Hospital',
                               'Hospitalized Respirator' : 'Hospitalized_Respirator',
                               'Hospitalized Severe': 'Hospitalized_Severe',
                               'Source Cases' : 'Source_Cases',
                               'Vac Given' : 'Vaccine_Given',
                               'Vac Given 1' : 'Vaccine_Given_1',
                               'Vac Given 1 Cum' : 'Vac_Given_1_Cum',
                               'Vac Given 2' : 'Vac_Given_2',
                               'Vac Given 2 Cum' : 'Vac_Given_2_Cum',
                               'Vac Given Cum' : 'Vac_Given_Cum'
                               })
    return covid
    
def get_confirmed_covid19_infections() :
    url = "https://covid19.th-stat.com/json/covid19v2/getTimeline.json"
    response = requests.post(url)
    response_result = (json.dumps(response.json(), indent=4))
    data = json.loads(response.content)
    jsonData = data["Data"]
    covid = pd.json_normalize(jsonData)
    return covid
    
    
def get_vaccine_per_province() :
    url = "https://raw.githubusercontent.com/wiki/djay/covidthailand/vaccinations.csv"
    df = pd.read_csv(url)
    df_clean = df.rename(columns={'Vac Allocated AstraZeneca 1': 'Vac_Allocated_AstraZeneca_1',
                               'Vac Allocated AstraZeneca 2': 'Vac_Allocated_AstraZeneca_2',
                               'Vac Allocated Sinovac 1':'Vac_Allocated_Sinovac_1',
                               'Vac Allocated Sinovac 2': 'Vac_Allocated_Sinovac_2',
                               'Vac Given': 'Vac_Given',
                               'Vac Given 1 Cum': 'Vac_Given_1_Cum',
                               'Vac Given 2 %' : 'Vac_Given_2_percent',
                               'Vac Given 2 Cum': 'Vac_Given_2_Cum',
                               'Vac Given AstraZeneca' : 'Vac_Given_AstraZeneca',
                               'Vac Given AstraZeneca Cum' : 'Vac_Given_AstraZeneca_Cum',
                               'Vac Given Cum' : 'Vac_Given_Cum',
                               'Vac Given Sinopharm': 'Vac_Given_Sinopharm',
                               'Vac Given Sinopharm Cum' : 'Vac_Given_Sinopharm_Cum',
                               'Vac Given Sinovac': 'Vac_Given_Sinovac',
                               'Vac Given Sinovac Cum' : 'Vac_Given_Sinovac_Cum'
                               })
    covid = df_clean[['Date','Vac_Allocated_AstraZeneca_1','Vac_Allocated_AstraZeneca_2','Vac_Allocated_Sinovac_1','Vac_Allocated_Sinovac_2',
                     'Vac_Given','Vac_Given_1_Cum','Vac_Given_2_percent','Vac_Given_2_Cum','Vac_Given_AstraZeneca','Vac_Given_AstraZeneca_Cum',
                     'Vac_Given_Cum','Vac_Given_Sinopharm','Vac_Given_Sinopharm_Cum','Vac_Given_Sinovac','Vac_Given_Sinovac_Cum'
                     ]]
    a = pd.to_datetime(covid['Date'])
    covid['Date'] = a.dt.strftime("%d-%m-%y")
    return covid
    
def get_covid_cases_by_province() :
    url = "https://raw.githubusercontent.com/wiki/djay/covidthailand/cases_by_province.csv"
    covid = pd.read_csv(url)
    return covid

def get_all_covid_data() :
    url = "https://raw.githubusercontent.com/wiki/djay/covidthailand/combined.csv"
    covid = pd.read_csv(url)
    return covid
    
def get_covid_UK() :  
    covid = pd.read_csv("Dataset_Covid_UK/UK_infected.csv")
    return covid
    
    
    
    
#get_vaccine()
#get_confirmed_covid19_infections()


raw_infected_uk = pd.read_csv("Dataset_Covid_UK/UK_infected.csv")
infected_uk = raw_infected_uk[['date','newCasesBySpecimenDate','cumCasesBySpecimenDate']]
infected_uk = infected_uk.rename(columns={'date': 'Date','newCasesBySpecimenDate': 'newCasesInfected','cumCasesBySpecimenDate':'cumCasesInfected'})
infected_uk = convert_format_time(infected_uk)

raw_death_uk = pd.read_csv("Dataset_Covid_UK/UK_death.csv")
death_uk = raw_death_uk[['date','newDailyNsoDeathsByDeathDate','cumDailyNsoDeathsByDeathDate']]
death_uk = death_uk.rename(columns={'date': 'Date'})
death_uk = convert_format_time(death_uk)


raw_vaccine1st_uk = pd.read_csv("Dataset_Covid_UK/UK_vaccine_1st.csv")
vaccine1st_uk = raw_vaccine1st_uk[['date','newPeopleVaccinatedFirstDoseByPublishDate','cumPeopleVaccinatedFirstDoseByPublishDate']]
vaccine1st_uk = vaccine1st_uk.rename(columns={'date': 'Date'})
vaccine1st_uk = convert_format_time(vaccine1st_uk)


raw_vaccine2nd_uk = pd.read_csv("Dataset_Covid_UK/UK_vaccine_2nd.csv")
vaccine2nd_uk = raw_vaccine2nd_uk[['date','newPeopleVaccinatedSecondDoseByPublishDate','cumPeopleVaccinatedSecondDoseByPublishDate']]
vaccine2nd_uk = vaccine2nd_uk.rename(columns={'date': 'Date'})
vaccine2nd_uk = convert_format_time(vaccine2nd_uk)

infected_death = pd.merge(infected_uk, death_uk, how="left", on="Date")
vaccine = pd.merge(vaccine1st_uk, vaccine2nd_uk, how="left", on="Date")
covid_uk = pd.merge(infected_death, vaccine, how="left", on="Date")

print(covid_uk)



pandas_gbq.context.credentials = credentials
pandas_gbq.to_gbq(covid_uk, 'Covid19_UK.covid_dataset', project_id='datateam-316802', if_exists='replace')