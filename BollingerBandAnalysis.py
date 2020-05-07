import numpy as np
import pandas as pd
import csv
import math
import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb

logger = logging.getLogger('BollingerBandAnalysis')
logger.setLevel(logging.DEBUG)

class BollingerBandAnalysis:

    def __init__(self, df_price: pd.DataFrame, s_bollinger_index_out_file: str, s_plot_out_file_prefix: str,
                 s_orders_out_file: str, s_values_out_file: str, f_commission = 1.0):
        self.df_price = df_price
        self.s_bollinger_index_out_file = s_bollinger_index_out_file
        self.s_plot_out_file_prefix = s_plot_out_file_prefix
        self.s_orders_out_file = s_orders_out_file
        self.s_values_out_file = s_values_out_file
        self.f_commission = f_commission # percent of order

    def run(self, n_band_width, n_bar_to_look_back, f_initial_cash, f_amount_per_trade, strategy=None):
        f_cash = f_initial_cash
        stop_loss = False

        logger.debug('Calculating bollinger index')
        #logger.debug('what is my data for price' + str(self.df_price.tail(5)))
        # NB : the first n_bar_to_look_back values are NaN
        df_rolling = self.df_price.rolling(n_bar_to_look_back)
        df_std = df_rolling.std()
        df_mean = df_rolling.mean()
        df_upper = df_mean + df_std * n_band_width
        df_lower = df_mean - df_std * n_band_width
        df_bollinger = (self.df_price - df_mean) / (df_std * n_band_width)
        #logger.debug('what is my data bol calc %f %f %f %f ' % (df_rolling.first(),df_std.first(),df_mean.first(),df_upper.first(),df_lower.first()))
        #logger.debug('what is my data for bollinger' + str(df_bollinger.tail(5)))

        df_bollinger.to_csv(self.s_bollinger_index_out_file)

        ls_dates = list(map(lambda dt: pd.to_datetime(dt).to_pydatetime(), self.df_price.index.values))
        ls_symbols = self.df_price.columns.values

        dls_upper_intersection = {}
        dls_lower_intersection = {}
        dls_buy = {}
        dls_sell = {}
        stop = {}

        d_shares = {}
        for s_symbol in ls_symbols:
            d_shares[s_symbol] = 0
            dls_upper_intersection[s_symbol] = []
            dls_lower_intersection[s_symbol] = []
            dls_buy[s_symbol] = []
            dls_sell[s_symbol] = []
            stop[s_symbol] = []

        logger.debug('Studying bollinger index over %d days to generate orders and values', len(ls_dates))
        with open(self.s_orders_out_file, 'w') as f_out_orders:
            order_writer = csv.writer(f_out_orders)

            with open(self.s_values_out_file, 'w') as f_out_values:
                value_writer = csv.writer(f_out_values)

                for idx in range(n_bar_to_look_back + 1, len(ls_dates)):
                    order_sell, order_buy = False, False
                    dt_date = ls_dates[idx]

                    for s_symbol in ls_symbols:
                        f_price_now = self.df_price[s_symbol][idx]

                        # if df_bollinger[s_symbol][idx] >= 1 and df_bollinger[s_symbol][idx-1] < 1:
                        #     dls_upper_intersection[s_symbol].append(dt_date)
                        # elif df_bollinger[s_symbol][idx] < 1 and df_bollinger[s_symbol][idx-1] >= 1:
                        #     dls_upper_intersection[s_symbol].append(dt_date)
                        # elif df_bollinger[s_symbol][idx] <= -1 and df_bollinger[s_symbol][idx-1] > -1:
                        #     dls_lower_intersection[s_symbol].append(dt_date)
                        # elif df_bollinger[s_symbol][idx] >= -1 and df_bollinger[s_symbol][idx-1] < -1:
                        #     dls_lower_intersection[s_symbol].append(dt_date)

                        if strategy == 'jlu1':
                            if df_bollinger[s_symbol][idx-1] >= 1 and \
                                    df_bollinger[s_symbol][idx] <= df_bollinger[s_symbol][idx - 1]:
                                # Went over 1.0 and decreasing
                                order_sell = True
                            elif (df_bollinger[s_symbol][idx] <= -1 or df_bollinger[s_symbol][idx-1] <= -1) and \
                                    df_bollinger[s_symbol][idx] >= df_bollinger[s_symbol][idx-1]:
                                # Went below -1.0 and increasing
                                order_buy = True
                        if strategy == 'jlu2':
                            if df_bollinger[s_symbol][idx] >= 1 and df_bollinger[s_symbol][idx-1] < 1:
                                # Went over 1.0 and re-enter neutral zone
                                order_sell = True
                            elif df_mean[s_symbol][idx-n_bar_to_look_back] <= df_mean[s_symbol][idx] and \
                                    (df_bollinger[s_symbol][idx] <= -1 or df_bollinger[s_symbol][idx-1] <= -1) and \
                                    df_mean[s_symbol][idx]-df_mean[s_symbol][idx-1] >= -df_std[s_symbol][idx]*0.1:
                                # Went below -1.0 and increasing and trend positive or neutral
                                order_buy = True
                        if strategy == 'jlu3':
                            possible_sell = False
                            if idx > n_bar_to_look_back+5:
                                for jdx in range(idx-5, idx+1):
                                    if df_bollinger[s_symbol][jdx] >= 1 and df_bollinger[s_symbol][jdx-1] <= 1:
                                        # Went over 1.0 in the near past or presently
                                        possible_sell = True
                            if possible_sell and self.df_price[s_symbol][idx] <= self.df_price[s_symbol][idx-1]\
                                and self.df_price[s_symbol][idx-1] <= self.df_price[s_symbol][idx-2]:
                                # Went over 1.0 and decreasing
                                order_sell = True
                            elif df_mean[s_symbol][idx - n_bar_to_look_back] <= df_mean[s_symbol][idx] and \
                                     (df_bollinger[s_symbol][idx] <= -1 or df_bollinger[s_symbol][idx - 1] <= -1) and \
                                     df_bollinger[s_symbol][idx] >= df_bollinger[s_symbol][idx - 1]:
                                # Went below -1.0 and increasing
                                order_buy = True
                        if strategy == 'jlu4':
                            possible_sell = False
                            if idx > n_bar_to_look_back+5:
                                for jdx in range(idx-5, idx+1):
                                    if df_bollinger[s_symbol][jdx] >= 1 and df_bollinger[s_symbol][jdx-1] <= 1:
                                        # Went over 1.0 in the near past or presently
                                        possible_sell = True
                            if possible_sell and self.df_price[s_symbol][idx] <= self.df_price[s_symbol][idx-1]\
                                and self.df_price[s_symbol][idx-1] <= self.df_price[s_symbol][idx-2]:
                                # Went over 1.0 and decreasing
                                order_sell = True
                            elif (df_bollinger[s_symbol][idx] <= -1 or df_bollinger[s_symbol][idx - 1] <= -1) and \
                                    df_bollinger[s_symbol][idx] >= df_bollinger[s_symbol][idx - 1] and\
                                    df_mean[s_symbol][idx]-df_mean[s_symbol][idx-1] >= -df_std[s_symbol][idx]*0.1:
                                # Went below -1.0 and increasing and mean trend neutral or positive
                                order_buy = True
                        else:
                            if df_bollinger[s_symbol][idx] >= 1 and df_bollinger[s_symbol][idx-1] < 1:
                                # Went over 1.0 and re-enter neutral zone
                                order_sell = True
                            elif df_bollinger[s_symbol][idx] >= -1 and df_bollinger[s_symbol][idx-1] < -1:
                                # Went below -1.0 and re-enter neutral zone
                                order_buy = True

                        if stop_loss and d_shares[s_symbol] > 0 and f_price_now <= stop[s_symbol][-1]:
                            order_sell = True

                        if order_buy:
                            order_buy = False
                            f_quantity = math.ceil(min(f_amount_per_trade, f_cash) / f_price_now)
                            if d_shares[s_symbol] <= f_quantity*2.5 and f_quantity * f_price_now > 1500:
                                logger.debug('Buying %d shares of %s on %s at %f (%f,%f)',
                                             f_quantity, s_symbol, dt_date, f_price_now,
                                             df_bollinger[s_symbol][idx - 1], df_bollinger[s_symbol][idx])
                                dls_buy[s_symbol].append(dt_date)
                                stop[s_symbol].append(f_price_now - df_std[s_symbol][idx] * n_band_width)
                                order_writer.writerow([dt_date.strftime('%Y-%m-%dT%H:%M:%S'), s_symbol, 'Buy', f_quantity])
                                d_shares[s_symbol] += f_quantity
                                f_cash -= f_price_now * f_quantity
                                f_cash -= self.f_commission * math.ceil(f_price_now * f_quantity / 100)
                        elif order_sell:
                            order_sell = False
                            f_quantity = f_amount_per_trade/ f_price_now
                            if True or f_quantity*1.3 >= d_shares[s_symbol]:
                                f_quantity = d_shares[s_symbol]
                            if f_quantity > 0:
                                dls_sell[s_symbol].append(dt_date)
                                logger.debug('Selling %d shares of %s on %s at %f (%f,%f)',
                                             f_quantity, s_symbol, dt_date, f_price_now,
                                             df_bollinger[s_symbol][idx-1], df_bollinger[s_symbol][idx])
                                order_writer.writerow(
                                    [dt_date.strftime('%Y-%m-%dT%H:%M:%S'), s_symbol, 'Sell', f_quantity])
                                d_shares[s_symbol] -= f_quantity
                                f_cash += f_price_now * f_quantity
                                f_cash -= self.f_commission * math.ceil(f_price_now * f_quantity / 100)
                    f_value = f_cash
                    for s_symbol in ls_symbols:
                        f_value += d_shares[s_symbol] * self.df_price[s_symbol][idx]

                    row = [dt_date.strftime('%Y-%m-%dT%H:%M:%S'), f_value]
                    # row.append('CASH')
                    # row.append(str(f_cash))
                    # for s_symbol in ls_symbols:
                        # row.append(s_symbol)
                        # row.append(str(d_shares[s_symbol]))
                        # row.append(str(self.df_price[s_symbol][idx]))

                    value_writer.writerow(row)

        logger.debug('Strategy complete')

        if len(ls_symbols) < 20:
            logger.debug('Plotting bollinger bands')
            for s_symbol in ls_symbols:

                # Plot bollinger band
                plt.clf()
                plt.xlabel('Date')
                ax1 = plt.subplot(211)
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * .8, box.height])
                ax1.plot(ls_dates, self.df_price[s_symbol].values)
                ax1.plot(ls_dates, df_mean[s_symbol].values)
                ax1.fill_between(np.array(ls_dates),
                                 df_lower[s_symbol].squeeze(),
                                 df_upper[s_symbol].squeeze(),
                                 facecolor='gray',
                                 alpha=0.4)
                ax1.legend([s_symbol, 'SMA'], loc='center left', bbox_to_anchor=(1, 0.5),
                           fancybox=True, shadow=True)
                ax1.set_ylabel('Adjusted Close')

                for dt in dls_sell[s_symbol]:
                    plt.axvline(x=dt, color='red')
                for dt in dls_buy[s_symbol]:
                    plt.axvline(x=dt, color='green')

                # Plot bollinger value chart
                ax2 = plt.subplot(212)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * .8, box.height])
                ax2.plot(ls_dates, df_bollinger[s_symbol])
                ax2.fill_between(ls_dates, -1, 1, facecolor='gray', alpha=0.4)
                ax2.set_ylabel('Bollinger value')

                for dt in dls_sell[s_symbol]:
                    plt.axvline(x=dt, color='red')
                for dt in dls_buy[s_symbol]:
                    plt.axvline(x=dt, color='green')

                plt.savefig(self.s_plot_out_file_prefix + '-' + s_symbol + '.pdf', format='pdf')

            logger.debug('Finished plotting bollinger bands')
