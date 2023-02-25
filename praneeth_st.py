import pandas as pd
import numpy as np
import time
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import warnings
warnings.filterwarnings("ignore")
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
#!pip install autots
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("yfinance")
install("yahoofinancials")
import yfinance as yf
from yahoofinancials import YahooFinancials
install("autots")
from autots import AutoTS, load_daily
from os import path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.title('Foundational Project')
import os

input_file = True #st.sidebar.file_uploader("Please upload your data file here:", type=['csv'])

if input_file:
    class WebDriver(object):

        def __init__(self):
            self.options = Options()

            self.options.binary_location = '/opt/headless-chromium'
            self.options.add_argument('--headless')
            self.options.add_argument('--no-sandbox')
            self.options.add_argument('--start-maximized')
            self.options.add_argument('--start-fullscreen')
            self.options.add_argument('--single-process')
            self.options.add_argument('--disable-dev-shm-usage')

        def get(self):
            driver = webdriver.Chrome(ChromeDriverManager().install())
            return driver
    def get_tickers(driver):
        """return the number of tickers available on the webpage"""
        TABLE_CLASS = "W(100%)"  
        tablerows = len(driver.find_elements(By.XPATH, value="//table[@class= '{}']/tbody/tr".format(TABLE_CLASS)))
        return tablerows
    def parse_ticker(rownum, table_driver):
        """Parsing each Ticker row by row and return the data in the form of Python dictionary"""
        Symbol = table_driver.find_element(By.XPATH, value="//tr[{}]/td[1]".format(rownum)).text
        Name = table_driver.find_element(By.XPATH, value="//tr[{}]/td[2]".format(rownum)).text
        LastPrice = table_driver.find_element(By.XPATH, value="//tr[{}]/td[3]".format(rownum)).text
        MarketTime = table_driver.find_element(By.XPATH, value="//tr[{}]/td[4]".format(rownum)).text
        Change = table_driver.find_element(By.XPATH, value="//tr[{}]/td[5]".format(rownum)).text
        PercentChange = table_driver.find_element(By.XPATH, value="//tr[{}]/td[6]".format(rownum)).text	
        Volume = table_driver.find_element(By.XPATH, value="//tr[{}]/td[7]".format(rownum)).text
        MarketCap = table_driver.find_element(By.XPATH, value="//tr[{}]/td[8]".format(rownum)).text	

        return {
        'Symbol': Symbol,
        'Name': Name,
        'LastPrice': LastPrice,
        'MarketTime': MarketTime,
        'Change': Change,
        'PercentChange': PercentChange,
        'Volume': Volume,
        'MarketCap': MarketCap
        }
    YAHOO_FINANCE_URL = "https://finance.yahoo.com/screener/unsaved/d0ac6574-dc65-4bec-a860-428098c86c2c?offset=0&count=100" 
    tab_titles=['Fetch Data from Yahoo Finance!','Target Variable', 'Plot','Understanding Cummulative Return','Peformance']
    tabs=st.tabs(tab_titles)
    instance_ = WebDriver()
    driver = instance_.get()
    driver.get(YAHOO_FINANCE_URL)
    print('Fetching the page')
    table_rows = get_tickers(driver)
    print('Found {} Tickers'.format(table_rows))
    print('Parsing Trending tickers')
    table_data = [parse_ticker(i, driver) for i in range (1, table_rows + 1)]
    driver.close()
    driver.quit()
    type(table_data)
    table_data_df=pd.DataFrame(table_data)
    table_data_df =table_data_df.drop_duplicates("Name")
    table_data_df['Change']=table_data_df['Change'].str.rstrip("%")
    table_data_df = table_data_df.astype({'Change':'float'})
    names=table_data_df[table_data_df["Change"]==table_data_df["Change"].max()]["Name"] 
    Symbol=table_data_df[table_data_df["Change"]==table_data_df["Change"].max()]["Symbol"]
    yahoo_financials = YahooFinancials('DIVISLAB.BO')
    data=yahoo_financials.get_historical_price_data("2022-06-10", "2023-03-25", "daily")
    btc_df = pd.DataFrame(data['DIVISLAB.BO']['prices'])
    btc_df = btc_df.drop('date', axis=1).set_index('formatted_date')
    btc_df.dropna(how='any',inplace=True)
    change= btc_df['adjclose'].pct_change()
    change.plot(title=" stock price")
    df_s = btc_df[[ 'adjclose']]

    df_s.index = pd.to_datetime(df_s.index)
    df_s = df_s.sort_values('formatted_date')
    train_df_s = df_s.iloc[:172]
    test_df_s = df_s.iloc[172:]
    with tabs[0]:
        st.table(table_data_df)
    with tabs[1]:
        st.table(df_s)
    with tabs[2]:
        plt.title('Divis Lab', size=20)
        train_df_s.adjclose.plot(figsize=(15,8), title= 'Train Data', fontsize=14, label='Train')
        test_df_s.adjclose.plot(figsize=(15,8), title= 'Test Data', fontsize=14, label='Test',color='orange')
        plt.legend()
        plt.grid()
        plt.show()
    model = AutoTS(
    forecast_length=5,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
    )
    long = False 
    model = model.fit(
    train_df_s,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
    )
    prediction = model.predict()
    yahoo_financials = YahooFinancials('DIVISLAB.BO')
    data=yahoo_financials.get_historical_price_data('2012-01-01','2022-01-01',"daily")
    btc_df_l = pd.DataFrame(data['DIVISLAB.BO']['prices'])
    btc_df_l = btc_df_l.drop('date', axis=1).set_index('formatted_date')
    btc_df_l= btc_df_l[['adjclose']]
    btc_df_l['Stock_Returns']=btc_df_l['adjclose'].pct_change()
    btc_df_l['Stock_cumRETURNS']=btc_df_l['Stock_Returns'].cumsum().apply(np.exp)
    btc_df_l.dropna(how='any',inplace=True)
    with tabs[3]:
        btc_df_l
        sns.set_style('whitegrid')
        btc_df_l['Stock_cumRETURNS'].plot(figsize=(8,8),label="Stock")
        plt.title('Equity Curves')
        plt.ylabel("Cumulative Returns")
        plt.xlabel("Index")
        plt.legend(loc='upper left')
        plt.show()
    df = btc_df_l[[ 'adjclose']]
    df = df.sort_values('formatted_date')
    df.index = pd.to_datetime(df.index)
    train_df = df.iloc[:2218]
    test_df = df.iloc[2218:]
    with tabs[4]:
        plt.title('DIVIS', size=20)
        train_df.adjclose.plot(figsize=(15,8), title= 'Train Data', fontsize=14, label='Train')
        test_df.adjclose.plot(figsize=(15,8), title= 'Test Data', fontsize=14, label='Test',color='orange')
        plt.legend()
        plt.grid()
        plt.show()
    model = AutoTS(
    forecast_length=246,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
    )
    long = False
    model = model.fit(
    train_df,
    date_col='datetime' if long else None,
    value_col='value' if long else None,
    id_col='series_id' if long else None,
    )
    prediction = model.predict()

    #plot a sample
    prediction.plot(model.df_wide_numeric,
                    series=model.df_wide_numeric.columns[0],
                        start_date="2021-05-30")
    # point forecasts dataframe
    forecasts_df = prediction.forecast
    # upper and lower forecasts
    forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast
    with tabs[5]:
        st.table(forecasts_df)
        model_results = model.results()
        # and aggregated from cross validation
        validation_results = model.results("validation")
        st.table(model_results)
        st.table(validation_results)