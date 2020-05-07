import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
import yaml
import logging

import Util

PORT_NAME = 'Portfolio'
logger = logging.getLogger('PortfolioAnalyzer')
logger.setLevel(logging.DEBUG)

class PortfolioAnalyzer:
    def __init__(self, valuesFile, baselineSymbol):
        self.valuesFile = valuesFile
        self.baselineSymbol = baselineSymbol
        with open("config.yaml") as file:
            mycfg = yaml.full_load(file)
        self.symbol_file = mycfg["symbol_file"]
        self.bar_size = mycfg["bar_size"]
        self.data_type = mycfg["data_type"]
        self.date_analysis_start = dt.datetime.strptime(mycfg["date_analysis_start"], "%Y-%m-%dT%H:%M:%S")
        self.date_analysis_end = dt.datetime.strptime(mycfg["date_analysis_end"], "%Y-%m-%dT%H:%M:%S")
        self.n_bar_total_max = int(mycfg["n_bar_total_max"])
        self.symbol_file = mycfg["symbol_file"]
        self.reference_ind = mycfg["reference_ind"]
        self.starting_cash = int(mycfg["starting_cash"])
        self.n_band_width = float(mycfg["n_band_width"])
        self.n_bar_to_look_back = int(mycfg["n_bar_to_look_back"])
        self.bs_strategy = mycfg["bs_strategy"]

    def run(self):
        logger.debug("Start analyizing %s ..." % self.valuesFile)
        with open (self.valuesFile, 'r') as fin:
            reader = csv.reader(fin)
            dates = []
            values = []
            for row in reader:
                date = dt.datetime.strptime(row[0],'%Y-%m-%dT%H:%M:%S')
                dates.append(date)
                value = float(row[-1])
                values.append(value)

        self.df_price = pd.DataFrame(data=values, index=dates, columns=[PORT_NAME])
        startDate = self.df_price.index[0]
        endDate = self.df_price.index[-1]

        def receive(symbol, df):
            logger.debug("Baseline data received")
            self.df_price[symbol] = pd.Series(data=df['close'].values, index=self.df_price.index)
            self.analyze()

        import HistoricalData
        HistoricalData.request(self.baselineSymbol, startDate, endDate, self.data_type, self.bar_size, receive)


    def evaluate(self, na_normalized_price):
        na_rets = na_normalized_price.copy()
        Util.returnize0(na_rets)
        vol = np.std(na_rets)
        daily_ret = np.mean(na_rets)
        cum_ret = na_normalized_price[-1] / na_normalized_price[0]
        sharpe = np.sqrt(252) * daily_ret / vol

        return sharpe, cum_ret, vol, daily_ret

    def analyze(self):
        na_normalized_price = self.df_price.values / self.df_price.values[0,:]

        dates = list(map(lambda dt: pd.to_datetime(dt).to_pydatetime(), self.df_price.index.values))
        # Evaluate both fund and baseline
        sharpe, cum_ret, vol, daily_ret = self.evaluate(self.df_price[PORT_NAME])
        sharpe_b, cum_ret_b, vol_b, daily_ret_b = self.evaluate(self.df_price[self.baselineSymbol])

        plt.clf()
        plt.plot(dates, na_normalized_price)
        plt.legend(self.df_price.columns.values)
        plt.ylabel('Normalized close')
        plt.xlabel('Date')
        plt.savefig(self.valuesFile + '-analyzed.pdf', format='pdf')
        plt.text('Total Return of Fund: %f' % cum_ret, (0.,0.1),'figure fraction')
        plt.text('Total Return of ' + self.baselineSymbol + ': %f' % cum_ret_b, (0.,0.),'figure fraction')

        logger.info ('Details of the performance of the portfolio:')
        logger.info ('')
        logger.info ('Date Range: %s to %s' %(dates[0].strftime("%Y-%m-%dT%H:%M:%S"), dates[-1].strftime("%Y-%m-%dT%H:%M:%S")))
        logger.info ('')
        logger.info ('Sharpe Ratio of Fund: %f' % sharpe)
        logger.info ('Sharpe Ratio of ' + self.baselineSymbol + ': %f' % sharpe_b)
        logger.info ('')
        logger.info ('Total Return of Fund: %f' % cum_ret)
        logger.info ('Total Return of ' + self.baselineSymbol + ': %f' % cum_ret_b)
        logger.info ('')
        logger.info ('Standard Deviation of Fund: %f' % vol)
        logger.info ('Standard Deviation of ' + self.baselineSymbol + ': %f' % vol_b)
        logger.info ('')
        logger.info ('Average Daily Return of Fund: %f' % daily_ret)
        logger.info ('Average Daily Return of ' + self.baselineSymbol + ': %f' % daily_ret_b)
