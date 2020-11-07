#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:48:43 2020

@author: jonathanmbuyi

In this project, I scrape data sport data from Heisman Trophy website: https://www.heisman.com/heisman-winners/.
From this website, I pulled data of all heisman trophy winners from the first winner in 1935 to the most recent winner in 2019.

"""
#-----------------------------------------------
# Import required libraries
#------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import streamlit as st
from wordcloud import WordCloud

#------------------------------------------------
# Headers
#------------------------------------------------
st.title('Heisman Trophy Winners 🏆 ')
st.subheader('Scraping sport data from a website')
st.write('\n\n')

#------------------------------------------------
# REtrieve sport data
#------------------------------------------------

#URL link
url = ('https://www.heisman.com/heisman-winners/')

#request function
@st.cache(allow_output_mutation=True)
def reqst_n_content():
    r = requests.get(url)
    soup = BeautifulSoup(r.content)
    
    return soup


# Retrieve content
soup = reqst_n_content()

# Retrieve raw table data
raw_tdsets = soup.find(id='heisman_winner_table', class_='table_roster') 
tr_data = raw_tdsets.find_all('div', {'class':'row'})
final_data = [i.get_text().strip().split('\n\n\n') for i in tr_data]

# Extract columns headers
table_headers = soup.find(class_="content-padder").find(class_='table_headings').find(class_='row')
headers = table_headers.find_all('div')
final_headers = [i.text.strip() for i in headers]

# Create dataframe
df = pd.DataFrame(final_data, columns=final_headers[1:])
st.write('### Heisman Trophy Winners List')
st.write(df.style.hide_index())


#------------------------------------------------
# Explore the retrieved sport data table
#------------------------------------------------
st.write('## Exploring Heisman Winners Dataset')
st.write(df['Player'].value_counts().rename('Trophies by Player'))
st.write('##### Archie Griffin appears to be the only player to have won this award twice (1974 & 1975) since its introduction in 1935.')
st.image('https://www.heisman.com/wp-content/uploads/1974/05/74-75-ARCHIE-GRIFFIN.jpg?x84391',caption='Archie Griffin')

#Let's group different backs position under running back (RB), then plot the chart
df['Pos'] = np.where(df['Pos'].isin(['RB','HB','FB']),'RB',df['Pos'])
# Plot # of heisman winners by field position
pos = df['Pos'].value_counts()
plt.figure(figsize=(16,7))
pos.plot(kind='barh', color='goldenrod')
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.xlabel('Total (and % of) Winners')
plt.ylabel('Position', fontsize=12)
plt.title('\n\n Heisman Winners by Position', fontsize=16)
for a,b in zip(range(len(pos.index)), pos.values): 
    plt.text(b, a, str(b) + '(' + str(round(b/len(df)*100)) +'%)' 
            ,fontsize=10
            ,weight='bold'
            ,va='center'
            ,ha='center'
            ,color='black')
st.pyplot()
st.write('##### RB and QB positions have produced far more winners than any other position, with RB claiming over half of overall winners (when combined RB, FB and HB)')

st.write('\n\n\n')
st.write('##### Charles Woodson is the only defensive player to have won the Heisman trophy back in 1997.')
cb_winner = df[df['Pos'] == 'CB']
cb_winner.set_index('Player', inplace=True)
st.write(cb_winner)
st.image('https://www.heisman.com/wp-content/uploads/1997/05/97-CHARLES-WOODSON.jpg?x84391', caption='Charles Woodson')

# Let's see organisations with the most heisman winners
#Generate word cloud to visualise most influential schools in relation to this prestigious award. 
wc = WordCloud(width=600, height=300, max_words=500, background_color='whitesmoke').generate_from_frequencies(df['School'].value_counts())

plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.title('\n\n Heisman Winners by School', fontsize=16)
plt.axis('off')
st.pyplot()

st.write('##### Colleges (universities) that stand out from the plot above are Oklahoma, Notre Dame, Ohio State and USC.')

st.write(df['School'].value_counts().rename('Winners by School'))

# Cleanse Class field
df['Class'] = [x.replace('*','') for x in df['Class']]
df['Class'].value_counts()

st.write('\n\n\n')
# Plot position against class
xtab = pd.crosstab(df.Pos,df.Class)
xtab = xtab[['Freshman','Sophomore','Junior','Senior']]
plt.figure(figsize=(12,5))
sns.heatmap(xtab,cmap='YlGnBu', annot=True, cbar=False, vmin=-10)
plt.title('Winners by Position and Class', fontsize=16)
plt.xlabel('Year of School (Class)', fontsize=12)
plt.ylabel('Position', fontsize=12)
st.pyplot()
st.write('##### As expected, the majority of players (just over 90%) have won the award in their Junior or Senior year of school.')
st.write('##### Just a handful of players have won this trophy in their early years of college (as Freshman or Sophomore), with 5 out of 6 winners being Quaterbacks and the other winner being a running back. These fastest winners are:')
st.write('\n')
yng_wnrs = df[df['Class'].isin (['Freshman','Sophomore'])].reset_index(drop=True)
yng_wnrs.set_index('Player', inplace=True)
st.write(yng_wnrs)
         