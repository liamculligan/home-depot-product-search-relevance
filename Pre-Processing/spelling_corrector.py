"""
Home Depot Product Search Relevance

For each search term in training and testing data, determine if Google corrects the search.
Save those searches that are corrected to a dictionary so that potential spelling
errors in the search term can be corrected.

Original Author: Kaggle user steubk
Updated By: Tyrone Cragg & Liam Culligan

Date: April 2016
"""

#Import required packages and functions
import pandas as pd
import requests
import re
import time
from random import randint
import csv

#Read in search terms
train = pd.read_csv("train.csv", header = 0, usecols = ['search_term'], encoding = "ISO-8859-1")

test = pd.read_csv("test.csv", header = 0, usecols = ['search_term'], encoding = "ISO-8859-1")

#Concatenate train and test
search_term = pd.concat((train, test)).reset_index(drop=True)

#Drop duplicated search terms
search_term = search_term.drop_duplicates()

#Convert to list
search_term = search_term['search_term'].tolist()

#Define the function
start_spell_check = "<span class=\"spell\">Showing results for</span>"
end_spell_check = "<br><span class=\"spell_orig\">Search instead for"

html_codes = (
		("'", '&#39;'),
		('"', '&quot;'),
		('>', '&gt;'),
		('<', '&lt;'),
		('&', '&amp;'),
)

def spell_check(s):
    #Takes a string as input and returns Google's corrected result
    q = '+'.join(s.split())
    time.sleep(  randint(2,5) ) #relax and don't let google be angry
    r = requests.get("https://www.google.co.uk/search?q="+q)
    content = r.text
    start = content.find(start_spell_check) 
    if (start > -1):
        start = start + len(start_spell_check)
        end=content.find(end_spell_check)
        search = content[start:end]
        search = re.sub(r'<[^>]+>', '', search)
        for code in html_codes:
            search = search.replace(code[1], code[0])
        correction = search[1:]
    else:
        correction = ""
    return correction

search_term = search_term[3:15]

#Loop through each unique search term and apply the above function
corrections = {}
for search in search_term:
    google_result = spell_check(search)
    if google_result != "":
        print(search + ": " + google_result)
        corrections[search] = google_result

#Save to CSV
with open('correction_playing.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['search_term_old','search_term_new'])
    for row in corrections.items():
        writer.writerow(row)
