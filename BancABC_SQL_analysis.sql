--Data cleaning starts with examining the given data which consists of two tables:
--Table 1:
--Aggregator IP List.csv imported as dbo.aggregators which contains a list of all the whitelisted IPs 
--that belong to known financial aggregators that are allowed by Banc ABC to access their customer’s accounts.
--Table 2:
--Login Transactions.csv imprted as dbo.Login Transactions
--This file contains all the logins observed Shape over a 24 hour period on Banc ABC’s special Aggregator ONLY endpoint. 
--This endpoint was set up specifically to process transactions for the whitelisted aggregators so as not to interfere with regular customer traffic which uses a different endpoint


SELECT * FROM [dbo].[Login Transactions];
SELECT * from dbo.Aggregators ORDER BY Aggregator;

--Checking the number of white-listed IPs for each aggregator
--It shows that YoungOnes has the highest number of white-listed IPs allowed by Bank ABC
SELECT Aggregator, count(IP) FROM dbo.Aggregators
GROUP BY Aggregator;

--First question asked by CISO is>> Which aggregators are accessing Bank ABC system?

--The Excel file had already shown that Funtown is written in two ways: FunTown & Funtown, so I updated the table to correct it

SELECT DISTINCT Aggregator FROM dbo.[Aggregators]
ORDER BY Aggregator; 

UPDATE
     dbo.[Aggregators]
SET
    Aggregator = REPLACE(Aggregator, 'FunTown', 'Funtown')
WHERE
Aggregator IS NOT NULL;

SELECT count(DISTINCT IP) FROM dbo.[Aggregators];

--Analyzing IP from Login Transactions table to see any errors in the IP addresses column like spaces in the IP address or commas instead of periods or any alphabets in the IP.

SELECT  IP from dbo.[Login Transactions]
WHERE IP LIKE '%[abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,/; ]%';

--Examining all distinct IP addresses
SELECT DISTINCT IP FROM [dbo].[Login Transactions];

--Checking the LoginSuccess column to see if there are more than the two expected unique values: Fail & Success

SELECT DISTINCT LoginSuccess FROM [dbo].[Login Transactions]

WHERE LoginSuccess <> 'Fail' AND  LoginSuccess <> 'Success';

--Now I will be joining the login transactions and aggregators tables as left join to match the IP with the aggregator AND inserting the data into a new table for ease of running queries later for further calculations

 INSERT INTO dbo.[logins_aggregators]
 SELECT dbo.[Login Transactions].column1, dbo.[Login Transactions].IP, dbo.[Login Transactions].LoginSuccess, dbo.[Login Transactions].AccountName, dbo.[Login Transactions].timestamp, dbo.Aggregators.Aggregator

 FROM dbo.[Login Transactions]
 LEFT JOIN
 dbo.Aggregators
 ON dbo.[Login Transactions].IP = dbo.Aggregators.ip;
 
 SELECT * FROM [dbo].[logins_aggregators];

--I will replace NULL values in Aggregators column of the joined table with 'unfamiliar IP'

UPDATE
     dbo.logins_aggregators
SET
    aggregator = 'unfamiliar IP'
WHERE
aggregator IS NULL;

--Two of the IP addresses for Funtown from the Aggregators table are 68.142.128.0/24 and 68.142.133.0/24
--What 68.142.128.0/24 means is that If you're given the entire /24 block, you can have 254 routers (x.x.x.1 to x.x.x.254) each with private networks behind them
--So I will replace all unfamiliar IP addresses that fall in this range with Funtown


SELECT DISTINCT IP, aggregator FROM dbo.logins_aggregators
WHERE aggregator = 'unfamiliar IP' AND IP LIKE '68.142.128%' OR IP LIKE '68.142.133%'
ORDER BY IP;

UPDATE
     dbo.logins_aggregators
SET
    aggregator = 'Funtown'

WHERE aggregator = 'unfamiliar IP' AND IP LIKE '68.142.128%' OR IP LIKE '68.142.133%';

--Now that I have matched a range of IP addresses with Funtown I will now count the number of occurences of unfamiliar IP that do not match any of the aggregators. I could call them anomalies. There are three such IPs
-- two of these IPs appear 2 and 5 times and they both failed to login as per the loginsuccess column so these can be ignored
--108.39.243.2  appears 26191 times but the loginsuccess column shows this IP failed to login everytime so it can be ignored

SELECT IP, COUNT(IP) AS login_count FROM dbo.logins_aggregators
WHERE aggregator = 'unfamiliar IP'
GROUP BY IP
ORDER BY IP;

--Now that I have pretty much cleaned up the data from all the tables I can move to the second question
--How much volume are the aggregators sending?
--This would be the total login attempts by each aggregator so it would be a simple group by aggregator

SELECT * FROM dbo.logins_aggregators; --first taking a look at the table

SELECT aggregator, COUNT(LoginSuccess) AS Total_volume_per_aggregator
FROM dbo.logins_aggregators
GROUP BY aggregator
ORDER BY Total_volume_per_aggregator;

--Next question is how many individual user accounts are being accessed by the aggregators?

WITH accounts_accessed AS
(SELECT aggregator, LoginSuccess, count(DISTINCT Accountname) AS user_accounts_accessed FROM dbo.logins_aggregators
GROUP BY aggregator, LoginSuccess
)

SELECT aggregator, SUM(user_accounts_accessed) AS Total_user_accounts_accessed
FROM accounts_accessed
GROUP BY aggregator
ORDER BY Total_user_accounts_accessed;

--WHAT IS THE LOGIN SUCCESS RATE OF THESE AGGREGATORS AND IS THIS IN LINE WITH WHAT WOULD BE EXPECTED IN YOUR OPINION?
--here login success rate will be the percentage of successful logins from the total login attempts by each aggregator


SELECT aggregator, COUNT(CASE  WHEN LoginSuccess = 'Success' THEN 1 ELSE NULL END) AS successful_logins,  count(LoginSuccess) AS Total_login_attempts,
((COUNT(CASE  WHEN LoginSuccess = 'Success' THEN 1 ELSE NULL END) * 100.0)/count(LoginSuccess)) AS Login_success_rate
FROM dbo.logins_aggregators
GROUP BY aggregator
ORDER BY Login_success_rate; 

--What is the average number of transactions each aggregator sends per 10min interval?

--I will change the data type of epoch time stamp to bigINT to perform datetime conversion on it before finding
--the rolling average

ALTER TABLE dbo.logins_aggregators
ALTER COLUMN timestamps bigint;

SELECT *, DATEADD(ss, timestamps/1000, '1970/1/1 00:00:00') AS Date
FROM dbo.logins_aggregators; --this is to check if datetime conversion is successful

--Here I added another column to the table to include the converted readable date_time format
ALTER TABLE dbo.logins_aggregators
ADD date_time as DATEADD(ss, timestamps/1000, '1970/1/1 00:00:00');

SELECT * from DBO.logins_aggregators;

--I created a new table for ease of use for myself and I grouped the data with count of loginsuccess as the aggregation over the datetime rounded to minutes and the aggregator
SELECT  DATETRUNC(minute, date_time) AS minuts,aggregator,  count(LoginSuccess) AS transactions
FROM dbo.logins_aggregators
GROUP BY DATETRUNC(minute, date_time), aggregator
ORDER by minuts, aggregator;

--I checked what the table looked like for a single aggregator
SELECT * FROM [dbo].[Groupedbytime_aggregators_transactions]
WHERE Aggregator = 'Fintech'
ORDER BY Truncated_date;

--Here I used window functions to find the rolling average of login_transactions over 10 minutes intervals. I partitioned the table by aggregator and placed the condition that the first 10 rows of 
--the partitions should be null because they do not fit into the row range of 10 rows preceding hence rolling average over these should not be counted.
--I casted the column data of login_transactions as a float so I can get float as the result of the average and not the rounded whole numbers since the result follows the datatype of the original data in the column.

SELECT Truncated_date, Aggregator, 
CASE 
   WHEN ROW_NUMBER() OVER(PARTITION BY Aggregator ORDER BY Truncated_date) > 10 THEN ROUND(AVG(CAST(Login_transactions AS FLOAT)) OVER (PARTITION BY Aggregator ORDER BY Truncated_date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW), 2)
   ELSE NULL
   END AS moving_avg_10_min 
FROM [dbo].[Groupedbytime_aggregators_transactions];

--What is the max number of transactions each aggregator sends per 10 min interval?

SELECT Truncated_date, Aggregator, 
CASE 
   WHEN ROW_NUMBER() OVER(PARTITION BY Aggregator ORDER BY Truncated_date) > 10 THEN MAX(Login_transactions) OVER (PARTITION BY Aggregator ORDER BY Truncated_date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)
   ELSE NULL
   END AS max_transactions_per_10_min 
FROM [dbo].[Groupedbytime_aggregators_transactions];


--What would be the impact of the CISO’s proposal to limit each aggregator to 1 login per account per 10 min interval?

--First I check for any uppercase letters in the accountnames to check for repetition since sql server DISTINCT statement is case insensitive
SELECT Accountname FROM [dbo].[logins_aggregators]
WHERE Accountname LIKE '%[ABCDEFGHIJKLMNOPQRSTUVWXYZ]%'
COLLATE Latin1_General_CS_AS; 

--I also used multiple filters to test for alphabets only accountnames and accountnames with characters in them
SELECT Accountname FROM [dbo].[logins_aggregators]
WHERE Accountname NOT LIKE '%[abcdefghijklmnopqrstuvwxyz1234567890.,;\/]%'
ORDER BY Accountname;

SELECT DISTINCT Accountname FROM [dbo].[logins_aggregators]
ORDER BY Accountname;

INSERT INTO [dbo].[grouped_accountnames]

SELECT DATETRUNC(minute, date_time), aggregator, Accountname, Count(LoginSuccess)
FROM [dbo].[logins_aggregators]
GROUP BY DATETRUNC(minute, date_time), aggregator, Accountname
ORDER BY DATETRUNC(minute, date_time), aggregator, Accountname;

select * from [dbo].[grouped_accountnames];

--i had to create a new table that contains the sum of the count of logins for each individual account for each minute of the time as this shows the total login attempts by each aggregator for individual accounts.

INSERT INTO [dbo].[Sum_accounts_logins]

SELECT Truncated_date, Aggregator, SUM(Login_attempts)
FROM [dbo].[grouped_accountnames]
GROUP BY Truncated_date, Aggregator
ORDER BY Truncated_date, Aggregator;

SELECT * FROM [dbo].[Sum_accounts_logins]
ORDER BY Truncated_date, Aggregator;

--Now this query shows the AVERAGE NUMBER OF TRANSACTIONS per 10 minute interval for each aggregator

select Truncated_date, Aggregator, accounts_logged_avg_10_min from 
(SELECT Truncated_date, Aggregator,
CASE 
   WHEN ROW_NUMBER() OVER(PARTITION BY Aggregator ORDER BY Truncated_date) > 10 THEN ROUND(AVG(CAST(Sum_accounts_logins AS FLOAT)) OVER (PARTITION BY Aggregator ORDER BY Truncated_date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW), 2)
   ELSE NULL
   END AS accounts_logged_avg_10_min 
FROM [dbo].[Sum_accounts_logins]) t
where t.Aggregator = 'Unfamiliar IP';

--Now to find the maximum of total account logins per 10 minute interval for each aggregator
select Truncated_date, Aggregator, accounts_logged_max_10_min from 
(SELECT Truncated_date, Aggregator,
CASE 
   WHEN ROW_NUMBER() OVER(PARTITION BY Aggregator ORDER BY Truncated_date) > 10 THEN MAX(Sum_accounts_logins) OVER (PARTITION BY Aggregator ORDER BY Truncated_date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)
   ELSE NULL
   END AS accounts_logged_max_10_min 
FROM [dbo].[Sum_accounts_logins]) max_t
where max_t.Aggregator = 'Unfamiliar IP';

--I will also find the total transactions per 10 minute interval:

SELECT Truncated_date, Aggregator, total_transactions_per_10_min FROM
(SELECT Truncated_date, Aggregator,
CASE 
   WHEN ROW_NUMBER() OVER(PARTITION BY Aggregator ORDER BY Truncated_date) > 10 THEN SUM(Sum_accounts_logins) OVER (PARTITION BY Aggregator ORDER BY Truncated_date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)
   ELSE NULL
   END AS total_transactions_per_10_min 
FROM [dbo].[Sum_accounts_logins]) t
WHERE t.Aggregator = 'PayTM';


--Now I will create a table to group the count of total login transactions by aggregators and accountnames.
--I will also truncate the date to round it to minutes since I will use the minutes to define rolling 10 min intervals 

INSERT INTO [dbo].[grouped_accountnames]

SELECT DATETRUNC(minute, date_time), aggregator, Accountname, Count(LoginSuccess)
FROM [dbo].[logins_aggregators]
GROUP BY DATETRUNC(minute, date_time), aggregator, Accountname
ORDER BY DATETRUNC(minute, date_time), aggregator, Accountname;

select * from [dbo].[grouped_accountnames];

--I will calculate the running count of unique accounts per 10 min interval as this is equal to 1 login
--per account per 10 min interval as per CISO proposal

---AWS ACCOUNTS
with AWS_accounts as (select * from [dbo].[grouped_accountnames] where Aggregator = 'AWS')

SELECT d.Truncated_date, oa.Accountcount
from( select distinct t1.Truncated_date from AWS_accounts as t1) as d
outer apply (
               select count(distinct t2.Accountname) as Accountcount from AWS_accounts as t2 where t2.Truncated_date between dateadd(MINUTE, -10, d.Truncated_date)
			   AND d.Truncated_date) AS oa
order by d.Truncated_date ASC;

select count(distinct Accountname) from [dbo].[grouped_accountnames]
where aggregator = 'AWS' and Truncated_date BETWEEN '2018-10-28 05:00:00.000' AND '2018-10-28 05:02:00.000';

---FUNTOWN ACCOUNTS
with Funtown_accounts as (select * from [dbo].[grouped_accountnames] where Aggregator = 'Funtown')

SELECT d.Truncated_date, oa.Accountcount
from( select distinct t1.Truncated_date from Funtown_accounts as t1) as d
outer apply (
               select count(distinct t2.Accountname) as Accountcount from Funtown_accounts as t2 where t2.Truncated_date between dateadd(MINUTE, -10, d.Truncated_date)
			   AND d.Truncated_date) AS oa
order by d.Truncated_date ASC;



---FINTECH ACCOUNTS
with Fintech_accounts as (select * from [dbo].[grouped_accountnames] where Aggregator = 'Fintech')

SELECT d.Truncated_date, oa.Accountcount
from( select distinct t1.Truncated_date from Fintech_accounts as t1) as d
outer apply (
               select count(distinct t2.Accountname) as Accountcount from Fintech_accounts as t2 where t2.Truncated_date between dateadd(MINUTE, -10, d.Truncated_date)
			   AND d.Truncated_date) AS oa
order by d.Truncated_date ASC;



--YOUNGONES ACCOUNTS
with YoungOnes_accounts as (select * from [dbo].[grouped_accountnames] where Aggregator = 'YoungOnes')

SELECT d.Truncated_date, oa.Accountcount
from( select distinct t1.Truncated_date from YoungOnes_accounts as t1) as d
outer apply (
               select count(distinct t2.Accountname) as Accountcount from YoungOnes_accounts as t2 where t2.Truncated_date between dateadd(MINUTE, -10, d.Truncated_date)
			   AND d.Truncated_date) AS oa
order by d.Truncated_date ASC;


------INSIGHT ACCOUNTS
with Insight_accounts as (select * from [dbo].[grouped_accountnames] where Aggregator = 'Insight')

SELECT d.Truncated_date, oa.Accountcount
from( select distinct t1.Truncated_date from Insight_accounts as t1) as d
outer apply (
               select count(distinct t2.Accountname) as Accountcount from Insight_accounts as t2 where t2.Truncated_date between dateadd(MINUTE, -10, d.Truncated_date)
			   AND d.Truncated_date) AS oa
order by d.Truncated_date ASC;


--PAYTM ACCOUNTS
with PayTM_accounts as (select * from [dbo].[grouped_accountnames] where Aggregator = 'PayTM')

SELECT d.Truncated_date, oa.Accountcount
from( select distinct t1.Truncated_date from PayTM_accounts as t1) as d
outer apply (
               select count(distinct t2.Accountname) as Accountcount from PayTM_accounts as t2 where t2.Truncated_date between dateadd(MINUTE, -10, d.Truncated_date)
			   AND d.Truncated_date) AS oa
order by d.Truncated_date ASC;

--UNFAMILIAR IPs
with unfamiliarIP_accounts as (select * from [dbo].[grouped_accountnames] where Aggregator = 'Unfamiliar IP')

SELECT d.Truncated_date, oa.Accountcount
from( select distinct t1.Truncated_date from unfamiliarIP_accounts as t1) as d
outer apply (
               select count(distinct t2.Accountname) as Accountcount from unfamiliarIP_accounts as t2 where t2.Truncated_date between dateadd(MINUTE, -10, d.Truncated_date)
			   AND d.Truncated_date) AS oa
order by d.Truncated_date ASC;

--I will now move to Tableau and visualize my data analysis in a dashboard