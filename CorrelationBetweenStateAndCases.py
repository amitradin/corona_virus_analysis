"""lets start by reading the Us-countries-2020 to see the trend of deaths at the end of the pandemic.
We are going to check if there is a country that was out of the ordinary"""
import datetime
import operator

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chisquare

#first lest explore the data set abit to see what it contains
cases_2020=pd.read_csv('corona virus deaths/covid-19-data/to be commited/us-counties-2020.csv')
print(cases_2020.head())
print(cases_2020.tail())
print(cases_2020.shape)
print(cases_2020.info())
"""there are 884737 entries! thats a lot. and with the info() method we can see that there are almost no null values. except some values in the death column
how ever those null values account for 2000/884737 which is just 2.2 percent of the total values. Lets call it a rounding error for now
it is very difficult to see trends with that much information. so lets start by graphing the total per date to see the trend
the data accually inclued the amount of cases for that day, and not the new cases that day, which means that if we sum the total number of cases
we get more than 1.7 billion. which of course is not the amount of cases in the US becuase of the overlap. but it does paint a picuture of the cases
state in each country"""

cases_2020['date']=pd.to_datetime(cases_2020['date'])
cases_pivot=pd.pivot_table(cases_2020,values='cases',index='date',aggfunc=np.sum)
plt.plot(cases_pivot)
plt.ticklabel_format(axis='y',style='plain')
plt.title('Corona virus cases in the US in the year 2020',color='red')
plt.show()

"""there is definitely a clear uptrend, the plot is skewed massively to the left.
most of the cases happened towards the end of the year, and the cases does not seem to slow down.
my objective is to check if there are countries that deviate from this trend. maybe in some countries the trend is the opposite
to check this we are going to plot each country trend and see if there is something intersting we can find"""

"""we only need to run this loop once becuase it saves all the graphs to my computer. The images are attached
we save the graphs instead of showing them becuase it alot faster to look at the images side by side 
instead of looking at one graph at a time."""
"""for state in cases_2020['state'].unique():
    plt.figure(figsize=(10,6))
    cases_2020_state=cases_2020[['cases','date']][cases_2020['state']==state]
    cases_2020_state.dropna(inplace=True)
    cases_2020_state=cases_2020_state.sort_values('date')
    plt.plot(cases_2020_state['date'],cases_2020_state['cases'])
    plt.title(state)
    plt.savefig('C:\myPythonProjects\corona virus deaths\covid-19-data\images\{}'.format(state))
"""
"""most of the states follow a similar pattern: a sort of exponential growth
there are however a few states where this exponential growth only starts at the end of the year. Those countries include:
Wyoming, North Dakota, New Mexico.
Lets graph those 3 countries side by side with the rest
"""
states_3=cases_2020[cases_2020['state'].isin(['Wyoming','North Dakota','New Mexico'])]
states_other=cases_2020[~cases_2020['state'].isin(['Wyoming','North Dakota','New Mexico'])]
#of course we need to accually scale this graph so it will be relativly the same so instead of number growth lets look at percentage growth
plt.figure(figsize=(10,6))
states_3_pivot=pd.pivot_table(states_3,values='cases',index='date',aggfunc=np.sum)
states_3=states_3_pivot.reset_index()
states_3['percentage']=states_3['cases'].pct_change()
plt.plot(states_3['date'],states_3['percentage'],label='Wyoming-North Dakota-NewMexico')
plt.ticklabel_format(axis='y',style='plain')
other_pivot=pd.pivot_table(states_other,values='cases',index='date',aggfunc=np.sum)
states_other=other_pivot.reset_index()
states_other['percentage']=states_other['cases'].pct_change()
plt.plot(states_other['date'],states_other['percentage'],color='green',label='rest')
plt.ticklabel_format(axis='y',style='plain')
plt.legend()
plt.show()
"""despite my believes it seems that the graphs look almost identical when plotted side by side. This means that this states did not have a different trend
compared to the other states. if we want to find other patterns we need to think of another method to explore the data
One option we can use is to explore the death rate compare to the. First lets see if there is a corraltion"""
print(cases_2020['deaths'].corr(cases_2020['cases']))#A 0.75 correlation. Lets investigate further
death_cases_pivot=pd.pivot_table(cases_2020,values=['deaths','cases'],index='date',aggfunc=np.sum).reset_index()
death_cases_pivot['death_percent']=death_cases_pivot['deaths'].pct_change()
death_cases_pivot['cases_percent']=death_cases_pivot['cases'].pct_change()
plt.plot(death_cases_pivot['date'],death_cases_pivot['death_percent'],label='Death Percentile')
plt.plot(death_cases_pivot['date'],death_cases_pivot['cases_percent'],label='cases Percentile')
plt.legend()
plt.show()
"""If we consider the months that come after march (which did not include much activity), 
we can see a clear correlation between the death rate and the cases rate. which means that the 
corona cases actually killed proportionally to the cases rate.  
"""
"""now lest look at the p_value that comes out of the chisquare check to see how likely each state is 
to have the number of cases it has
"""
cases_pivot_state=pd.pivot_table(cases_2020,values='cases',index='date',columns='state',aggfunc=np.sum).reset_index()
cases_pivot_state.fillna(0,inplace=True)
cases_pivot_state.drop('date',axis=1,inplace=True)
p_values_cases={}
for name in cases_pivot_state:
    col=cases_pivot_state[name]
    p_values_cases[name]=chisquare(col)[1]
print(p_values_cases)#all zeros
"""we can conclude that the values for the cases are unlikely statistically which probably means
that the amount of corona virus cases could not have been predicted beforehand
Lets look at the same statistic for the death values"""
deaths_pivot_state=pd.pivot_table(cases_2020,values='deaths',index='date',columns='state',aggfunc=np.sum).reset_index()
deaths_pivot_state.fillna(0,inplace=True)
deaths_pivot_state.drop('date',axis=1,inplace=True)
p_values_deaths={}
for name in cases_pivot_state:
    col=deaths_pivot_state[name]
    p_values_deaths[name]=chisquare(col)[1]
print(p_values_deaths)
"""The deaths looks almost identical to the cases, all the p_values are 0 except Northern Mariana Islands which has a p_value of 1
which means that the result for that state is actually not significant. which means that the number of deaths there was 
attributed to chance. for the other states there is a different reason for this outcome. Of course to find those reasons
A large scale research has to be conducted on the matter. I just showcased my first statistical impression from the data"""

"""now lets see which country has the highest percentage of infected people"""
states_pivot=pd.pivot_table(cases_2020,values='cases',index='date',columns='state',aggfunc=np.sum).reset_index()
#lest start from march
states_pivot=states_pivot[states_pivot['date'] >= datetime.datetime(2020, 0o3, 0o1)]
states_pivot.fillna(0,inplace=True)
states_pivot.drop('date',inplace=True,axis=1)
print(states_pivot)
pop=pd.read_csv('corona virus deaths/covid-19-data/to be commited/NST-EST2022-ALLDATA.csv')
pop=pop[['NAME','ESTIMATESBASE2020']]
pop.rename(columns={'NAME':'state'},inplace=True)
states_pop=pd.Series(cases_2020['state'].unique(),name='state')
states_pop=pd.merge(left=states_pop,right=pop,how='inner')
#here we downloded another csv file containing information about the population of the states in the US according to information in 2020
#now we'll calculate the mean size and scale our answers to this size
states_pop_mean=states_pop['ESTIMATESBASE2020'].mean()
states_pop['multiplier']=states_pop_mean/states_pop['ESTIMATESBASE2020']
print(states_pop)
infected_percentage=dict()
for name1,pop in zip(states_pop['state'],states_pop['ESTIMATESBASE2020']):
    state_series=states_pivot[name1]
    infected_avg=np.mean(state_series)
    infected_percentage[name1]=infected_avg/pop
max=0
name=''
for key in infected_percentage:
    if infected_percentage[key]>max:
        max=infected_percentage[key]
        name=key
print(name,infected_percentage[name])#North Dakota: 0.0280
"""to conclude we can say that on average North Dakota had highest percentage of her population that was infected. 
meaning that in the time of the covid in 2020 if you lived in North Dakota you had the highest likelihood to be infected"""

"""Thats if for now. we got some insight to the state of states in the US in the year 2020"""