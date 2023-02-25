# Mia Scarpati and Izzy Huang
# Technical Indicator Strategy
import collections
import math
import numpy as np
import ML_Model

class TechStrat():
    def __init__(self, **kwargs):
        """
        Constructor for our strategy which synthesizes a variety of technical indicators
        """
        self.data={} # blank?
        self.orders={} # stores our orders based on indicators
        self.buyTimes={} # buy times
        self.typicalPrices={} # may not be important
        self.dayData={} # open, high, low, close, volume
        self.BBands={} # done
        self.SMARatio={} # done
        self.ATRRatio={} # done
        self.SMAVolRatio={} # done
        self.ADX5={} # done
        self.ADX15={} # done
        self.RSIRatio={} # done
        self.MACD={} # done
        self.StochasticRatio={} # done
        # self.RC={}
        tickers=kwargs["tickers"]
        for ticker in tickers:
            self.data[ticker]=[]
            self.orders[ticker]=[]
            self.typicalPrices[ticker]=[]
            self.dayData[ticker]=[-1, -1, math.inf, -1, -1] # open, high, low, close, volume
            self.BBands[ticker]=[0, 0] # upper, lower
            self.SMARatio[ticker]=[]
            self.ATRRatio[ticker]=[]
            self.ADX5[ticker]=[]
            self.ADX15[ticker]=[]
            self.RSIRatio[ticker]=[]
            self.MACD[ticker]=[]
            self.StochasticRatio[ticker]=[]
        
        # do we need below?
        self.ticks=0
        self.ticksPerDay=kwargs["dayConst"]
        self.daysInMovingAverage = kwargs["MAL"]
        # stores how many standard deviations we want to use to calculate our upper and lower bands. By default set to 2
        self.nextDayStart = kwargs["dayConst"]
        # variable to clear our least recent orders so that the length of the list doesnt get too big. By default set to 10,000
        self.clearDataLen = kwargs["clearDataLen"]

    def clear_orders(self, ticker):
        """
        Clears all current orders and logs relevant information.
        """
        print(f"Trashing %d orders.", len(self.orders[ticker]))
        self.orders[ticker] = []

    def updateSingle(self, **kwargs):
        """
        Update method called every tick. Just takes in the stock's price at the time of the tick. \n
        Inputs:
        "newDay": boolean, whether or not we're at a new day
        "ticker": string, the ticker of the stock we're updating
        "price": float, the price of the stock at the time of the tick
        """
        ticker = kwargs["ticker"]
        price = kwargs["newPrice"]
        if kwargs["newDay"]:
            # handle all new day management stuff
            # if we're at a new day, add our current day's data to TP and reset the dayData
            # print(ticker, self.dayData[ticker])
            self.typicalPrices[ticker].append(sum(self.dayData[ticker]) / 4)
            self.dayData[ticker] = [-1, price, price, price]

        # open, high, low, close — update all accordingly
        # update open only once
        if self.dayData[ticker][0] == -1:
            self.dayData[ticker][0] = price
        self.dayData[ticker][1] = max(self.dayData[ticker][1], price)
        self.dayData[ticker][2] = min(self.dayData[ticker][2], price)
        self.dayData[ticker][3] = price

    def update(self, newData):
        """
        generic update method that calls update on every ticker
        """

        newDay = False
        # if the tick we're at is equal to our defined time for when the next "day" starts, make note of the fact that we're
        # on a new "day" and update the time of the next "day" start.
        if self.ticks == self.nextDayStart:
            self.nextDayStart = self.ticks + self.ticksPerDay
            newDay = True

        for ticker, newPrice in newData.items():
            self.updateSingle(ticker=ticker, newPrice=newPrice, newDay=newDay)
        self.ticks += 1
    
    # calculate SMA ratio from SMA_5 and SMA_15, both in mins
    def SMA(self, ticker):
        """
        Calculates the SMA for a given ticker
        """
        # if we have enough data to calculate the SMA
        if len(self.dayData[ticker]) >= 15:
            # calculate the SMA
            SMA5 = self.dayData[ticker][3].transform(lambda x: x.rolling(window=5).mean())
            SMA15 = self.dayData[ticker][3].transform(lambda x: x.rolling(window=15).mean())
            SMARatio = SMA5/SMA15
            # store the SMA
            self.SMARatio[ticker].append(SMARatio)
    
    # calculate SMA vol ratio from SMA_5 and SMA_15, both in mins
    def SMAVol(self, ticker):
        """
        Calculates the SMA for a given ticker
        """
        # if we have enough data to calculate the SMA
        if len(self.dayData[ticker]) >= 15:
            # calculate the SMA
            SMA5 = self.dayData[ticker][4].transform(lambda x: x.rolling(window=5).mean())
            SMA15 = self.dayData[ticker][4].transform(lambda x: x.rolling(window=15).mean())
            SMAVolRatio = SMA5/SMA15
            # store the SMA
            self.SMAVolRatio[ticker].append(SMAVolRatio)
    
    # Wilder's Smoothing
    def Wilder(data, periods):
        start = np.where(~np.isnan(data))[0][0] #Check if nans present in beginning
        Wilder = np.array([np.nan]*len(data))
        Wilder[start+periods-1] = data[start:(start+periods)].mean() #Simple Moving Average
        for i in range(start+periods,len(data)):
            Wilder[i] = (Wilder[i-1]*(periods-1) + data[i])/periods #Wilder Smoothing
        return(Wilder)

    # calculate ATR ratio from ATR_5 and ATR_15, both in mins
    def ATR(self, ticker):
        """
        Calculates the ATR for a given ticker
        """
        # if we have enough data to calculate the ATR
        if len(self.dayData[ticker]) >= 15:
            # calculate the ATR
            prevClose = self.dayData[ticker][3].shift(1)
            TR = np.maximum(self.dayData[ticker][1]-self.dayData[ticker][2], np.maximum(abs(self.dayData[ticker][1]-prevClose), abs(prevClose-self.dayData[ticker][2])))
            ATR5 = self[ticker].Wilder(TR, 5) # lowkey these might be wrong
            ATR15 = self[ticker].Wilder(TR, 15)
            ATRRatio = ATR5/ATR15
            # store the ATR
            self.ATRRatio[ticker].append(ATRRatio)
    
    # calculate ADX_5 and ADX_15, both in mins
    def ADX(self, ticker):
        """
        Calculates the ADX for a given ticker
        """
        # if we have enough data to calculate the ADX
        if len(self.dayData[ticker]) >= 15:
            # calculate the ADX
            prevHigh = self.dayData[ticker][1].shift(1)
            prevLow = self.dayData[ticker][2].shift(1)
            prevClose = self.dayData[ticker][3].shift(1)

            # general up and down indicators
            upDM = np.where(~np.isnan(prevHigh), np.where(self.dayData[ticker][1]>prevHigh and self.dayData[ticker][1]-prevHigh>prevLow-self.dayData[ticker][2], self.dayData[ticker][1]-prevHigh, 0), np.nan)
            downDM = np.where(~np.isnan(prevLow), np.where(self.dayData[ticker][2]<prevLow and prevLow-self.dayData[ticker][2]>self.dayData[ticker][1]-prevHigh, prevLow-self.dayData[ticker][2], 0), np.nan)
            
            # wilder smooth the above to 5 and 15 min windows
            upDM5 = self[ticker].Wilder(upDM, 5)
            downDM5 = self[ticker].Wilder(downDM, 5)
            upDM15 = self[ticker].Wilder(upDM, 15)
            downDM15 = self[ticker].Wilder(downDM, 15)

            # ATR calculations
            TR = np.maximum(self.dayData[ticker][1]-self.dayData[ticker][2], np.maximum(abs(self.dayData[ticker][1]-prevClose), abs(prevClose-self.dayData[ticker][2])))
            ATR5 = self[ticker].Wilder(TR, 5)
            ATR15 = self[ticker].Wilder(TR, 15)

            # dir indicators, 5 and 15 min
            upDI5 = 100*upDM5/ATR5
            downDI5 = 100*downDM5/ATR5
            upDI15 = 100*upDM15/ATR15
            downDI15 = 100*downDM15/ATR15

            # convert into ADX format and store into array
            DX5 = 100*abs(upDI5-downDI5)/(upDI5+downDI5)
            DX15 = 100*abs(upDI15-downDI15)/(upDI15+downDI15)
            ADX5 = self[ticker].Wilder(DX5, 5)
            ADX15 = self[ticker].Wilder(DX15, 15)
            self.ADX5[ticker].append(ADX5)
            self.ADX15[ticker].append(ADX15)

    # calculate RSI ratio from RSI_5 and RSI_15, both in mins
    def RSI(self, ticker):
        """
        Calculates the RSI for a given ticker
        """
        # if we have enough data to calculate the RSI
        if len(self.dayData[ticker]) >= 15:
            # calculate the RSI
            diff = self.dayData[ticker][3].diff()
            up = diff.clip(lower=0)
            down = abs(diff.clip(upper=0))

            # 5 and 15 min windows
            up5 = up.transform(lambda x: x.rolling(window=5).mean())
            down5 = down.transform(lambda x: x.rolling(window=5).mean())
            up15 = up.transform(lambda x: x.rolling(window=15).mean())
            down15 = down.transform(lambda x: x.rolling(window=15).mean())

            # RSI calculations
            RS5 = up5/down5
            RS15 = up15/down15
            RSI5 = 100 - (100/(1+RS5))
            RSI15 = 100 - (100/(1+RS15))
            self.RSIRatio[ticker].append(RSI5/RSI15)
    
    # calculate MACD
    def MACDFunc(self, ticker):
        """
        Calculates the MACD for a given ticker
        """
        # if we have enough data to calculate the MACD
        if len(self.dayData[ticker]) >= 15:
            # calculate the MACD
            EMA5 = self.dayData[ticker][3].transform(lambda x: x.ewm(span=5, adjust=False).mean())
            EMA15 = self.dayData[ticker][3].transform(lambda x: x.ewm(span=15, adjust=False).mean())
            MACD = EMA5 - EMA15
            self.MACD[ticker].append(MACD)

    # calculate BBands
    def BollBands(self, ticker):
        """
        Calculates the Bollinger Bands for a given ticker
        """
        # if we have enough data to calculate the Bollinger Bands
        if len(self.dayData[ticker]) >= 15:
            # calculate the Bollinger Bands
            upperBand = self.dayData[ticker][3].transform(lambda x: x.rolling(window=5).mean() + 2*x.rolling(window=5).std())
            lowerBand = self.dayData[ticker][3].transform(lambda x: x.rolling(window=5).mean() - 2*x.rolling(window=5).std())
            self.BBands[ticker][0].append(upperBand)
            self.BBands[ticker][1].append(lowerBand)
    
    # calculate stochastic ratio
    def Stochastic(self, ticker):
        """
        Calculates the Stochastic for a given ticker
        """
        # if we have enough data to calculate the Stochastic
        if len(self.dayData[ticker]) >= 15:
            # calculate the Stochastic
            high5 = self.dayData[ticker][1].transform(lambda x: x.rolling(window=5).max())
            low5 = self.dayData[ticker][2].transform(lambda x: x.rolling(window=5).min())
            high15 = self.dayData[ticker][1].transform(lambda x: x.rolling(window=15).max())
            low15 = self.dayData[ticker][2].transform(lambda x: x.rolling(window=15).min())
            stochastic5 = (100*(self.dayData[ticker][3]-low5)/(high5-low5)).rolling(window=5).mean()
            stochastic15 = (100*(self.dayData[ticker][3]-low15)/(high15-low15)).rolling(window=15).mean()
            self.StochasticRatio[ticker].append(stochastic5/stochastic15)
    
    # calculate the trend of the last hour
    # this is bad, need to figure out how we're using indicators. prediction ?
    def Trend(self, ticker):
        """
        Calculates the trend of the last hour for a given ticker, based on technical indicators
        """
        # if we have enough data to calculate the trend
        if len(self.dayData[ticker]) >= 60:
            priceTrend = np.gradient(self.dayData[ticker][3][-60:])
            RSIRatio = self.RSIRatio[ticker][-60:]
            ADX = [self.ADX5[ticker][-60:],self.ADX15[ticker][-60:]]
            MACD = self.MACD[ticker][-60:]
            BBands = [self.BBands[ticker][0][-60:],self.BBands[ticker][1][-60:]]
            StochasticRatio = self.StochasticRatio[ticker][-60:]
            ATRRatio = self.ATRRatio[ticker][-60:]
            SMA = self.SMARatio[ticker][-60:]
            SMAVol = self.SMAVolRatio[ticker][-60:]

            # determine the trend
            if (priceTrend[-1] > 0 and RSIRatio[-1] > 1 and ADX[0][-1] > ADX[1][-1] and MACD[-1] > 0 and BBands[0][-1] > BBands[1][-1] and StochasticRatio[-1] > 1 and ATRRatio[-1] > 1 and SMA[-1] > 1 and SMAVol[-1] > 1):
                self.orders[ticker].append("BUY")
            elif (priceTrend[-1] < 0 and RSIRatio[-1] < 1 and ADX[0][-1] < ADX[1][-1] and MACD[-1] < 0 and BBands[0][-1] < BBands[1][-1] and StochasticRatio[-1] < 1 and ATRRatio[-1] < 1 and SMA[-1] < 1 and SMAVol[-1] < 1):
                self.orders[ticker].append("SELL")

    # top ten trade windows...unsure how we use this yet
    def Predict(self, ticker):
        # using random forest model, which we should probably train in a separate class
        Model = ML_Model()
        top = Model.predict(ticker)
        for i in range(10):
            self.buyTimes[ticker].append(top[i]['Time'])
