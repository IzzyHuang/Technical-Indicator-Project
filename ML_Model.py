# Class to train the ML model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from scipy.stats import mstats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, TimeSeriesSplit, GridSearchCV
import pickle
import os
sns.set()

# copying code from ipynb file so model can be easily called in strategy
class ML_Model():

    def __init__(self):
        self.data = pd.read_csv('/Users/miascarpati/Desktop/Northwestern/Project/Technical-Indicator-Project/MSFT_2022_Minute_data.csv')
    
    def prep_data(self):
        df_list = []
        folder_path = '/Users/miascarpati/Desktop/Northwestern/Project/Technical-Indicator-Project/MSFT_data/'
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                df_list.append(df)
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df = merged_df.sort_values('ts_event')
        merged_df.to_csv("MSFT_2022_Minute_data.csv")
        all_data = self.data
        all_data = all_data.rename(columns={'open':'Open','close': 'Close', 'high': 'High', 'low': 'Low', 'volume':'Volume', 'ticker': 'symbol'})

        # dictionary to convert time to readable format
        time_lookup = {}
        temp = [[i,j] for i,j in zip(all_data.index, all_data["ts_event"])]
        for k in temp:
            time_lookup[k[0]] = k[1]

        #Creating Return column
        all_data['return'] = all_data.groupby('symbol')['Close'].pct_change() 

        ###
        ### Simple Moving Average 
        ###
        all_data['SMA_5'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 5).mean())
        all_data['SMA_15'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 15).mean())
        all_data['SMA_ratio'] = all_data['SMA_15'] / all_data['SMA_5']

        ###
        ### Simple Moving Average Volume
        ###
        all_data['SMA5_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 5).mean())
        all_data['SMA15_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 15).mean())
        all_data['SMA_Volume_Ratio'] = all_data['SMA5_Volume']/all_data['SMA15_Volume']

        ###
        ### Wilder's Smoothing
        ###
        def Wilder(data, periods):
            start = np.where(~np.isnan(data))[0][0] #Check if nans present in beginning
            Wilder = np.array([np.nan]*len(data))
            Wilder[start+periods-1] = data[start:(start+periods)].mean() #Simple Moving Average
            for i in range(start+periods,len(data)):
                Wilder[i] = (Wilder[i-1]*(periods-1) + data[i])/periods #Wilder Smoothing
            return(Wilder)

        ###
        ### Average True Range (ATR)
        ###
        all_data['prev_close'] = all_data.groupby('symbol')['Close'].shift(1)
        all_data['TR'] = np.maximum((all_data['High'] - all_data['Low']), 
                            np.maximum(abs(all_data['High'] - all_data['prev_close']), 
                            abs(all_data['prev_close'] - all_data['Low'])))
        for i in all_data['symbol'].unique():
            TR_data = all_data[all_data.symbol == i].copy()
            all_data.loc[all_data.symbol==i,'ATR_5'] = Wilder(TR_data['TR'], 5)
            all_data.loc[all_data.symbol==i,'ATR_15'] = Wilder(TR_data['TR'], 15)

        all_data['ATR_Ratio'] = all_data['ATR_5'] / all_data['ATR_15']

        ###
        ### Average Directional Index (ADX)
        ###
        all_data['prev_high'] = all_data.groupby('symbol')['High'].shift(1)
        all_data['prev_low'] = all_data.groupby('symbol')['Low'].shift(1)

        all_data['+DM'] = np.where(~np.isnan(all_data.prev_high),
                                np.where((all_data['High'] > all_data['prev_high']) & 
                (((all_data['High'] - all_data['prev_high']) > (all_data['prev_low'] - all_data['Low']))), 
                                                                        all_data['High'] - all_data['prev_high'], 
                                                                        0),np.nan)

        all_data['-DM'] = np.where(~np.isnan(all_data.prev_low),
                                np.where((all_data['prev_low'] > all_data['Low']) & 
                (((all_data['prev_low'] - all_data['Low']) > (all_data['High'] - all_data['prev_high']))), 
                                            all_data['prev_low'] - all_data['Low'], 
                                            0),np.nan)

        for i in all_data['symbol'].unique():
            ADX_data = all_data[all_data.symbol == i].copy()
            all_data.loc[all_data.symbol==i,'+DM_5'] = Wilder(ADX_data['+DM'], 5)
            all_data.loc[all_data.symbol==i,'-DM_5'] = Wilder(ADX_data['-DM'], 5)
            all_data.loc[all_data.symbol==i,'+DM_15'] = Wilder(ADX_data['+DM'], 15)
            all_data.loc[all_data.symbol==i,'-DM_15'] = Wilder(ADX_data['-DM'], 15)

        all_data['+DI_5'] = (all_data['+DM_5']/all_data['ATR_5'])*100
        all_data['-DI_5'] = (all_data['-DM_5']/all_data['ATR_5'])*100
        all_data['+DI_15'] = (all_data['+DM_15']/all_data['ATR_15'])*100
        all_data['-DI_15'] = (all_data['-DM_15']/all_data['ATR_15'])*100

        all_data['DX_5'] = (np.round(abs(all_data['+DI_5'] - all_data['-DI_5'])/(all_data['+DI_5'] + all_data['-DI_5']) * 100))

        all_data['DX_15'] = (np.round(abs(all_data['+DI_15'] - all_data['-DI_15'])/(all_data['+DI_15'] + all_data['-DI_15']) * 100))

        for i in all_data['symbol'].unique():
            ADX_data = all_data[all_data.symbol == i].copy()
            all_data.loc[all_data.symbol==i,'ADX_5'] = Wilder(ADX_data['DX_5'], 5)
            all_data.loc[all_data.symbol==i,'ADX_15'] = Wilder(ADX_data['DX_15'], 15)

        ###
        ### Stochastic Oscillators
        ###
        all_data['Lowest_5D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 5).min())
        all_data['High_5D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 5).max())
        all_data['Lowest_15D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 15).min())
        all_data['High_15D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 15).max())

        all_data['Stochastic_5'] = ((all_data['Close'] - all_data['Lowest_5D'])/(all_data['High_5D'] - all_data['Lowest_5D']))*100
        all_data['Stochastic_15'] = ((all_data['Close'] - all_data['Lowest_15D'])/(all_data['High_15D'] - all_data['Lowest_15D']))*100

        all_data['Stochastic_%D_5'] = all_data['Stochastic_5'].rolling(window = 5).mean()
        all_data['Stochastic_%D_15'] = all_data['Stochastic_5'].rolling(window = 15).mean()

        all_data['Stochastic_Ratio'] = all_data['Stochastic_%D_5']/all_data['Stochastic_%D_15']

        ###
        ### RSI
        ###
        all_data['Diff'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.diff())
        all_data['Up'] = all_data['Diff']
        all_data.loc[(all_data['Up']<0), 'Up'] = 0

        all_data['Down'] = all_data['Diff']
        all_data.loc[(all_data['Down']>0), 'Down'] = 0 
        all_data['Down'] = abs(all_data['Down'])

        all_data['avg_5up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=5).mean())
        all_data['avg_5down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=5).mean())

        all_data['avg_15up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=15).mean())
        all_data['avg_15down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=15).mean())

        all_data['RS_5'] = all_data['avg_5up'] / all_data['avg_5down']
        all_data['RS_15'] = all_data['avg_15up'] / all_data['avg_15down']

        all_data['RSI_5'] = 100 - (100/(1+all_data['RS_5']))
        all_data['RSI_15'] = 100 - (100/(1+all_data['RS_15']))

        all_data['RSI_ratio'] = all_data['RSI_5']/all_data['RSI_15']

        ###
        ### Moving Average Convergence Divergence (MACD)
        ###
        all_data['5Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
        all_data['15Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=15, adjust=False).mean())
        all_data['MACD'] = all_data['15Ewm'] - all_data['5Ewm']

        ###
        ### Bollinger Bands
        ###
        all_data['15MA'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=15).mean())
        all_data['SD'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=15).std())
        all_data['upperband'] = all_data['15MA'] + 2*all_data['SD']
        all_data['lowerband'] = all_data['15MA'] - 2*all_data['SD']

        ###
        ### Rate of Change
        ###
        all_data['RC'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = 15)) 

        # 1 hr prediction interval
        all_data['Close_Shifted'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.shift(-59))
        all_data['Target'] = ((all_data['Close_Shifted'] - all_data['Open'])/(all_data['Open']) * 100).shift(-1)
        all_data['Target_Direction'] = np.where(all_data['Target']>0,1,0)
        all_data = all_data.dropna().copy()

        # target variables for random forest
        Target_variables = ['SMA_ratio','ATR_5','ATR_15','ATR_Ratio',
                        'ADX_5','ADX_15','SMA_Volume_Ratio','Stochastic_5','Stochastic_15','Stochastic_Ratio',
                        'RSI_5','RSI_15','RSI_ratio','MACD']
        for variable in Target_variables:
            all_data.loc[:,variable] = mstats.winsorize(all_data.loc[:,variable], limits = [0.1,0.1])

        return all_data

    def returnTargetVars(self):
        Target_variables = ['SMA_ratio','ATR_5','ATR_15','ATR_Ratio',
                       'ADX_5','ADX_15','SMA_Volume_Ratio','Stochastic_5','Stochastic_15','Stochastic_Ratio',
                      'RSI_5','RSI_15','RSI_ratio','MACD']
        return Target_variables

    def train_model(self):
        all_data = self.prep_data()
        Target_variables = self.returnTargetVars()

        # train test split
        train_data = all_data.loc[:128787,]
        test_data = all_data.loc[128787:] 
        X_train = train_data.loc[:,Target_variables]
        Y_train = train_data.loc[:,['Target_Direction']]

        # validation curve
        rf = RandomForestClassifier()
        train_scoreNum, test_scoreNum = validation_curve(rf,
                                        X = X_train[:], y = Y_train.loc[:,'Target_Direction'], 
                                        param_name = 'n_estimators', 
                                        param_range = [3,4,7,10,12,15,20,25,30], cv = TimeSeriesSplit(n_splits = 3))

        train_scores_mean = np.mean(train_scoreNum, axis=1)
        train_scores_std = np.std(train_scoreNum, axis=1)
        test_scores_mean = np.mean(test_scoreNum, axis=1)
        test_scores_std = np.std(test_scoreNum, axis=1)

        co_data = all_data.copy()
        co_train = co_data[:128787]
        co_train = co_train.dropna().copy()
        co_train

        X_train = co_train.loc[:,Target_variables]

        Y_train = co_train.loc[:,['Target_Direction']]

        #Define paramters from Validation Curve
        params = {'max_depth': [5, 7],
                'max_features': ['sqrt'],
                'min_samples_leaf': [10, 15, 20],
                'n_estimators': [5, 7, 9],
                'min_samples_split':[20, 25, 30]} #Using Validation Curves

        rf = RandomForestClassifier()

        #Perform a TimeSeriesSplit on the dataset
        time_series_split = TimeSeriesSplit(n_splits = 3)


        rf_cv = GridSearchCV(rf, params, cv = time_series_split, n_jobs = -1, verbose = 20)

        #Fit the random forest with our X_train and Y_train
        rf_cv.fit(X_train, Y_train)
                
        #Save the fitted variable into a Pickle file
        file_loc = f'{os.getcwd()}\\Pickle_Files\\Cluster_'    
        pickle.dump(rf_cv, open(file_loc,'wb'))

    # to do : define predictor function. that's what will be called in the strategy and a trend will be determined
    def predictor(self, day_data):
        self.train_model()
        
        Target_variables = self.returnTargetVars()
        all_data = self.prep_data()
        time_lookup = {}
        temp = [[i,j] for i,j in zip(all_data.index, all_data["ts_event"])]
        for k in temp:
            time_lookup[k[0]] = k[1]

        try:
            pred_for_tomorrow = pd.DataFrame({'Date':[],
                                            'company':[],
                                            'prediction':[]})

            #Predict each stock using the 2nd January Data
            rf_cv =  pickle.load(open(os.getcwd() + f'\\Pickle_Files\\Cluster_', 'rb'))
            best_rf = rf_cv.best_estimator_
            cluster_data = day_data.copy()
            cluster_data = cluster_data.dropna()
            if (cluster_data.shape[0]>0):
                X_test = cluster_data.loc[:,Target_variables]
                

                pred_for_tomorrow = pred_for_tomorrow.append(pd.DataFrame({'Date':cluster_data.index,
                                                                            'company':cluster_data['symbol'],
                                                                            'prediction':best_rf.predict_proba(X_test)[:,1]}), ignore_index = True)
            
            top_10_pred = pred_for_tomorrow.sort_values(by = ['prediction'], ascending = False).head(10)

            for selected_company in top_10_pred['company']:
                actual = all_data[all_data.symbol == selected_company].loc['2019-01-02','Target_Direction']
                pct_change = all_data[all_data.symbol == selected_company].loc['2019-01-02','Target']
                top_10_pred.loc[top_10_pred.company == selected_company,'actual'] = actual
                top_10_pred.loc[top_10_pred.company == selected_company,'pct_change'] = pct_change
            
        except:
            pass
        top_10_pred["Time"] = [time_lookup[i] for i in top_10_pred['Date']]

        return top_10_pred

