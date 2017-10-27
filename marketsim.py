import datetime as dt
import numpy as np
import pandas as pd
import sys
import csv
import math
import pdb

# types
from ibapi.common import *
from ibapi.order_condition import *
from ibapi.contract import *
from ibapi.order import *
from ibapi.order_state import *
from ibapi.execution import Execution
from ibapi.execution import ExecutionFilter
from ibapi.commission_report import CommissionReport
from ibapi.scanner import ScannerSubscription
from ibapi.ticktype import *

from orders import Orders
import contracts
import historicalData

class OrderWrapper:
    def __init__(self, date:dt.datetime, symbol:str, order:Order):
        self.date = date
        self.symbol = symbol
        self.order = order

class MarketSimulator:
    def __init__(self, initValue, ordersFile, valuesFile, commissionPrice=0, callback=None):
        self.initValue = initValue
        self.ordersFile = ordersFile
        self.valuesFile = valuesFile
        self.symbols_in_order = []
        self.data = []
        self.df_data = {}
        self.commissionPrice = commissionPrice
        self.callback = callback

    def run(self):
        self.orders = []
        self.symbols = []
        self.shares = {}
        with open (self.ordersFile, 'r') as fin:
            reader = csv.reader(fin)
            for row in reader:
                date = dt.datetime(int(row[0]), int(row[1]), int(row[2]), 16)
                symbol = row[3]
                action = row[4]
                quantity = float(row[5])
                order = OrderWrapper(date, symbol, Orders.MarketOrder(action, quantity))
                self.orders.append(order)

                if order.symbol not in self.symbols:
                    self.symbols.append(order.symbol)
                    self.shares[order.symbol] = 0

        self.orders = sorted(self.orders, key=lambda order: order.date)
        startDate = self.orders[0].date
        endDate = self.orders[-1].date

        print ("Needing", self.symbols)

        self.symbols_in_order = []
        def receive(symbol, df):
            self.onDataReceived(symbol, df)
        for symbol in self.symbols:
            historicalData.request(symbol, startDate, endDate, "ADJUSTED_LAST", "1 day", receive)

    def onDataReceived(self, symbol, df):
        print ("Got data for ", symbol)
        self.data.append(df)
        self.symbols_in_order.append(symbol)
        if len(self.data) == len(self.symbols):
            self.df_data = pd.concat(self.data, keys=self.symbols_in_order)
            self.df_data = self.df_data['close'].unstack().T # Convert into 2D df of closing price with date as index and symbol as column
            for column in self.df_data.columns.values:
                self.df_data[column] = self.df_data[column].fillna(method="ffill")
                self.df_data[column] = self.df_data[column].fillna(method="bfill")
                self.df_data[column] = self.df_data[column].fillna(1.0)
            print ("Got all the data we need. Simulating..")
            self.simulate()

    def simulate(self):
        cash = self.initValue
        n_days = len(self.df_data)
        n_orders = len(self.orders)
        i_order = 0

        with open(self.valuesFile, 'w') as fout:
            writer = csv.writer(fout)
            started = False
            for i in range(0, n_days):
                date = self.df_data.index[i].to_datetime() + dt.timedelta(hours=16)
                while i_order < n_orders and self.orders[i_order].date < date:
                    i_order += 1

                while i_order < n_orders and self.orders[i_order].date == date:
                    started = True
                    order = self.orders[i_order]
                    try:
                        price = self.df_data[order.symbol][i]
                    except:
                        print ("Bad data. ignoring")
                        i_order += 1
                        continue
                    quantity = order.order.totalQuantity if order.order.action.lower() == "buy" else -order.order.totalQuantity
                    amount = price * quantity
                    commission = self.commissionPrice * math.ceil(quantity / 100)
                    cash -= (amount + commission)
                    self.shares[order.symbol] += quantity
                    i_order += 1

                value = cash
                for symbol in self.df_data.columns.values:
                    value += self.shares[symbol] * self.df_data[symbol][i]

                if started:
                    writer.writerow([date.year, date.month, date.day, value])

        if self.callback != None:
            self.callback()
