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

    def __init__(self, df_price: pd.DataFrame, s_bollinger_index_out_file: str, s_plot_out_file_prefix: str, s_orders_out_file: str, s_values_out_file: str, f_commission = 1.0):
        self.df_price = df_price
        self.s_bollinger_index_out_file = s_bollinger_index_out_file
        self.s_plot_out_file_prefix = s_plot_out_file_prefix
        self.s_orders_out_file = s_orders_out_file
        self.s_values_out_file = s_values_out_file
        self.f_commission = f_commission # percent of order

    def run(self, n_band_width, n_bar_to_look_back, f_initial_cash, f_amount_per_trade):
        method = 'ori'
        f_cash = f_initial_cash

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

        d_shares = {}
        for s_symbol in ls_symbols:
            d_shares[s_symbol] = 0
            dls_upper_intersection[s_symbol] = []
            dls_lower_intersection[s_symbol] = []

        logger.debug('Studying bollinger index over %d days to generate orders and values', len(ls_dates))
        with open(self.s_orders_out_file, 'w') as f_out_orders:
            order_writer = csv.writer(f_out_orders)

            with open(self.s_values_out_file, 'w') as f_out_values:
                value_writer = csv.writer(f_out_values)

                for idx in range(n_bar_to_look_back + 1, len(ls_dates)):
                    order_sell, order_buy = False, False
                    dt_date = ls_dates[idx]

                    for s_symbol in ls_symbols:
                        f_price_today = self.df_price[s_symbol][idx]

                        if df_bollinger[s_symbol][idx] >= 1 and df_bollinger[s_symbol][idx-1] < 1:
                            dls_upper_intersection[s_symbol].append(dt_date)
                        elif df_bollinger[s_symbol][idx] < 1 and df_bollinger[s_symbol][idx-1] >= 1:
                            dls_upper_intersection[s_symbol].append(dt_date)
                        elif df_bollinger[s_symbol][idx] <= -1 and df_bollinger[s_symbol][idx-1] > -1:
                            dls_lower_intersection[s_symbol].append(dt_date)
                        elif df_bollinger[s_symbol][idx] >= -1 and df_bollinger[s_symbol][idx-1] < -1:
                            dls_lower_intersection[s_symbol].append(dt_date)
                        if method == 'jlu':
                            if (df_bollinger[s_symbol][idx] >= 1 or df_bollinger[s_symbol][idx-1] >= 1) and \
                                    df_bollinger[s_symbol][idx] <= df_bollinger[s_symbol][idx-1]:
                                # Went over 1.0 and decreasing; sell all shares
                                order_sell = True
                            elif (df_bollinger[s_symbol][idx] <= -1 or df_bollinger[s_symbol][idx-1] <= -1) and \
                                    df_bollinger[s_symbol][idx] >= df_bollinger[s_symbol][idx-1]:
                                # Went below -1.0 and increasing; buy shares
                                order_buy = True
                        else:
                            if df_bollinger[s_symbol][idx] >= 1 and df_bollinger[s_symbol][idx-1] < 1:
                                # Went over 1.0; sell all shares
                                order_sell = True
                            elif df_bollinger[s_symbol][idx] >= -1 and df_bollinger[s_symbol][idx-1] < -1:
                                # Went below -1.0; buy shares
                                order_buy = True

                        if order_buy:
                            order_buy = False
                            f_quantity = math.ceil(min(f_amount_per_trade, f_cash) / f_price_today)
                            logger.debug('Buying %d shares of %s on %s at %f (%f,%f)',
                                         f_quantity, s_symbol, dt_date, f_price_today,
                                         df_bollinger[s_symbol][idx-1], df_bollinger[s_symbol][idx])
                            if f_quantity * f_price_today > 1500:
                                order_writer.writerow([dt_date.strftime('%Y-%m-%dT%H:%M:%S'), s_symbol, 'Buy', f_quantity])
                                d_shares[s_symbol] += f_quantity
                                f_cash -= f_price_today * f_quantity
                                f_cash -= self.f_commission * math.ceil(f_price_today * f_quantity / 100)
                        elif order_sell:
                            order_sell = False
                            f_quantity = d_shares[s_symbol]
                            if f_quantity > 0:
                                logger.debug('Selling %d shares of %s on %s at %f (%f,%f)',
                                             f_quantity, s_symbol, dt_date, f_price_today,
                                             df_bollinger[s_symbol][idx-1], df_bollinger[s_symbol][idx])
                                #order_writer.writerow([dt_date.year, dt_date.month, dt_date.day, s_symbol, 'Sell', f_quantity])
                                order_writer.writerow(
                                    [dt_date.strftime('%Y-%m-%dT%H:%M:%S'), s_symbol, 'Sell', f_quantity])
                                d_shares[s_symbol] -= f_quantity
                                f_cash += f_price_today * f_quantity
                                f_cash -= self.f_commission * math.ceil(f_price_today * f_quantity / 100)
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

        if len(ls_symbols) < 10:
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

                for dt in dls_upper_intersection[s_symbol]:
                    plt.axvline(x=dt, color='red')
                for dt in dls_lower_intersection[s_symbol]:
                    plt.axvline(x=dt, color='green')

                # Plot bollinger value chart
                ax2 = plt.subplot(212)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * .8, box.height])
                ax2.plot(ls_dates, df_bollinger[s_symbol])
                ax2.fill_between(ls_dates, -1, 1, facecolor='gray', alpha=0.4)
                ax2.set_ylabel('Bollinger value')

                for dt in dls_upper_intersection[s_symbol]:
                    plt.axvline(x=dt, color='red')
                for dt in dls_lower_intersection[s_symbol]:
                    plt.axvline(x=dt, color='green')

                plt.savefig(self.s_plot_out_file_prefix + '-' + s_symbol + '.pdf', format='pdf')

            logger.debug('Finished plotting bollinger bands')
