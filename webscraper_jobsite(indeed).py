from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd


start_time = datetime.now()

ser = Service("C:\\python codes\\chromedriver2\\chromedriver.exe")
op = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=ser, options=op)
driver.maximize_window()

list_url = [] 
job_titles = []
companies= []
locations = []
date_posted = []
scrapping_dates = []
for page in range(0, 30, 10):

	url = 'https://ca.indeed.com/jobs?q=data+scientist&l=Ontario&start=0' + str(page)
	list_url.append(url)

	

for webpage in list_url:
	driver.get(webpage)

	


	all_jobs = driver.find_elements(By.CLASS_NAME, 'slider_container')
#driver.find_element(By.CLASS_NAME, "quiz_button")

	

	for job in all_jobs:
		result_html = job.get_attribute('innerHTML')
		soup = BeautifulSoup(result_html, 'html.parser')

		try:
			title = soup.find('h2', class_ = 'jobTitle').text.replace('\n', '')

		except:
			title = "None"

		try:
			company = soup.find('span', class_='companyName').text

		except:
			company = "None"

		try:
			location = soup.find('div', class_='companyLocation').text
		except:
			location = "None"
		try:
			date = soup.find('span', class_= 'date').text.lstrip("Posted")
	#print(date.text.lstrip("Posted"))
		except:
			date = "None"
		
		scrapping_dates.append(datetime.now())
		job_titles.append(title)
		companies.append(company)
		locations.append(location)
		date_posted.append(date)

	time.sleep(3)



driver.quit()

df = pd.DataFrame()

df["Job"] = job_titles
df["Company"] = companies
df["Location"] = locations
df["Date Posted"] = date_posted
df['Scrapping Date'] = scrapping_dates
pd.set_option('display.max_columns', None) 

print(df)
df.to_csv(r'C:\python codes\file2.csv', header = True)
print(df)