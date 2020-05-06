import datetime as dt
import yaml


def execute():
    with open("config.yaml") as file:
        mycfg = yaml.full_load(file)
    symbol_file = mycfg["symbol_file"]
    bar_size = mycfg["bar_size"]
    data_type = mycfg["data_type"]
    date_analysis_start = dt.datetime.strptime(mycfg["date_analysis_start"], "%Y-%m-%dT%H:%M:%S")
    date_analysis_end = dt.datetime.strptime(mycfg["date_analysis_end"], "%Y-%m-%dT%H:%M:%S")
    n_bar_total_max = int(mycfg["n_bar_total_max"])
    symbol_file = mycfg["symbol_file"]
    reference_ind = mycfg["reference_ind"]
    starting_cash = int(mycfg["starting_cash"])
    f_amount_per_trade = int(mycfg["f_amount_per_trade"])
    n_band_width = float(mycfg["n_band_width"])
    n_bar_to_look_back = int(mycfg["n_bar_to_look_back"])

    # simulateMarket()
    loadContracts(symbol_file)
    # testSampleStrategy()

    testBollingerAnalysis(symbol_file, reference_ind, date_analysis_start, date_analysis_end,
                          starting_cash, f_amount_per_trade, n_band_width, n_bar_to_look_back, bar_size)


def testBollingerAnalysis(symbol_file, reference_ind, date_analysis_start, date_analysis_end,
                          starting_cash, f_amount_per_trade, n_band_width, n_bar_to_look_back, bar_size):
    from StrategyTest import testBollinger, sp500symbols
    #testBollinger(sp500symbols(symbol_file), 'SPY', dt.datetime(2017, 1, 1), dt.datetime(2017, 6, 1), 50000, 2, 20)
    testBollinger(sp500symbols(symbol_file), reference_ind, date_analysis_start, date_analysis_end,
                  starting_cash, f_amount_per_trade, n_band_width, n_bar_to_look_back, bar_size)
    # testBollinger(['MSFT', 'AMZN', 'GOOG', 'AAPL'], 'SPY', dt.datetime(2017, 1, 1), dt.datetime(2017, 6, 1), 10000, 1, 20)


def testSampleStrategy(symbol_file, reference_ind, date_analysis_start, date_analysis_end, starting_cash):
    from StrategyTest import test, sp500symbols
    test(sp500symbols(symbol_file), reference_ind, date_analysis_start, date_analysis_end, starting_cash)


def simulateMarket():
    from MarketSimulator import MarketSimulator
    MarketSimulator(1000000, "testdata/orders.csv", "testdata/values.csv", 1, callback=analyzePortfolio).run()


def analyzePortfolio(reference_ind):
    from Analyze import PortfolioAnalyzer
    PortfolioAnalyzer("testdata/values.csv", reference_ind).run()


def loadContracts(symbol_file):
    import SymbolsToContracts
    SymbolsToContracts.load(symbol_file)


