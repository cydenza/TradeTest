import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

#pd.set_printoptions(max_colwidth, 1000)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.width', 1000)

start_date = '2021-01-01'
end_date = '2021-09-30'
SRC_DATA_FILENAME = "samsung_data.pkl"

try:
    samsung_data = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    samsung_data = pdr.DataReader('005930.KS', 'yahoo', start_date, end_date)
    samsung_data.to_pickle(SRC_DATA_FILENAME)
    print(samsung_data)

samsung_data_signal = pd.DataFrame(index=samsung_data.index)
samsung_data_signal['price'] = samsung_data['Adj Close']

"""
samsung_data_signal['daily_difference'] = samsung_data_signal['price'].diff()

samsung_data_signal['signal'] = 0.0
samsung_data_signal['signal'] = np.where(samsung_data_signal['daily_difference'] > 0, 1.0, 0.0)

samsung_data_signal['positions'] = samsung_data_signal['signal'].diff()

print(samsung_data_signal.head(20))

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Samsung price in $')
samsung_data_signal['price'].plot(ax=ax1, color='r', lw=2.)

ax1.plot(samsung_data_signal.loc[samsung_data_signal.positions == 1.0].index,
         samsung_data_signal.price[samsung_data_signal.positions == 1.0],
         '^', markersize=5, color='m')

ax1.plot(samsung_data_signal.loc[samsung_data_signal.positions == -1.0].index,
         samsung_data_signal.price[samsung_data_signal.positions == -1.0],
         'v', markersize=5, color='k')
plt.show()


initial_capital = float(1000.0)

positions = pd.DataFrame(index=samsung_data_signal.index).fillna(0.0)
portfolio = pd.DataFrame(index=samsung_data_signal.index).fillna(0.0)

positions['SAMSUNG'] = samsung_data_signal['signal']
portfolio['positions'] = (positions.multiply(samsung_data_signal['price'], axis=0))

portfolio['cash'] = initial_capital - (positions.diff().multiply(samsung_data_signal['price'], axis=0)).cumsum()

portfolio['total'] = portfolio['positions'] + portfolio['cash']

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Samsung trade')
portfolio['total'].plot(ax=ax1, color='r', lw=2.)
#plt.bar(portfolio['total'], color='r', lw=2.)

plt.show()


lows = samsung_data['Low']
highs = samsung_data['High']

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Samsung trade')

highs.plot(ax=ax1, color='c', lw=2.)
lows.plot(ax=ax1, color='y', lw=2.)

plt.hlines(highs.head(50).max(), lows.index.values[0], lows.index.values[-1], linewidth=2, color='g')
plt.hlines(lows.head(50).min(), lows.index.values[0], lows.index.values[-1], linewidth=2, color='r')
plt.axvline(linewidth=2, color='b', x=lows.index.values[50], linestyle=':')
plt.show()
"""


def trading_support_resistance(data, bin_width=20):
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support = 0
    in_resistance = 0

    for x in  range((bin_width - 1) + bin_width, len(data)):
        data_section = data[x - bin_width:x+1]
        support_level = min(data_section['price'])
        resistance_level = max(data_section['price'])
        range_level = resistance_level-support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level
        data['sup_tolerance'][x] = support_level + 0.2 * range_level
        data['res_tolerance'][x] = resistance_level - 0.2 * range_level

        if data['price'][x] >= data['res_tolerance'][x] and \
            data['price'][x] <= data['res'][x]:
            in_resistance += 1
            print("in_resistance" + str(in_resistance))
            data['res_count'][x] = in_resistance
        elif data['price'][x] <= data['sup_tolerance'][x] and \
            data['price'][x] >= data['sup'][x]:
            in_support += 1
            print("in_support" + str(in_support))
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0

        if in_resistance > 2:
            data['signal'][x] = 1
            print(str(data['signal'][x]) + "1")
        elif in_support > 2:
            data['signal'][x] = 0
            print(str(data['signal'][x]) + "0")
        else:
            data['signal'][x] = data['signal'][x-1]
            print(str(data['signal'][x]) + "-")

    data['positions'] = data['signal'].diff()

    pos = []
    for x in data['signal']:
        if x == 1:
            pos.append(data['positions'])
    print(pos)

trading_support_resistance(samsung_data_signal)

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Samsung trade')
samsung_data_signal['sup'].plot(ax=ax1, color='g', lw=2.)
samsung_data_signal['res'].plot(ax=ax1, color='b', lw=2.)
samsung_data_signal['price'].plot(ax=ax1, color='r', lw=2.)
ax1.plot(samsung_data_signal.loc[samsung_data_signal.positions == 1.0].index,
         samsung_data_signal.price[samsung_data_signal.positions == 1.0],
         '^', markersize=7, color='k', label='buy')
ax1.plot(samsung_data_signal.loc[samsung_data_signal.positions == -1.0].index,
         samsung_data_signal.price[samsung_data_signal.positions == -1.0],
         'v', markersize=7, color='k', label='sell')
plt.legend()
plt.show()


"""
time_period = 20
history = []
sma_values = []

close = samsung_data['Adj Close']

for close_price in close:
    history.append(close_price)
    if len(history) > time_period:
        del(history[0])
    sma_values.append(stats.mean(history))

samsung_data = samsung_data.assign(ClosePrice=pd.Series(close, index=samsung_data.index))
samsung_data = samsung_data.assign(Simple20DayMovingAverage=pd.Series(sma_values,
                                    index=samsung_data.index))
close_price = samsung_data['ClosePrice']
sma = samsung_data['Simple20DayMovingAverage']

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Samsung')
close_price.plot(ax=ax1, color='g', lw=2., legend=True)
sma.plot(ax=ax1, color='r', lw=2., legend=True)
plt.show()



num_periods = 20
K = 2 / (num_periods + 1)
ema_p = 0
ema_values = []
close = samsung_data['Adj Close']

for close_price in close:
    if (ema_p == 0):
        ema_p = close_price
    else:
        ema_p = (close_price - ema_p) * K + ema_p
    ema_values.append(ema_p)

samsung_data = samsung_data.assign(ClosePrice=pd.Series(close, index=samsung_data.index))
samsung_data = samsung_data.assign(Exponential20DayMovingAverage=pd.Series(ema_values,
                                    index=samsung_data.index))
close_price = samsung_data['ClosePrice']
ema = samsung_data['Exponential20DayMovingAverage']

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Samsung')
close_price.plot(ax=ax1, color='g', lw=2., legend=True)
ema.plot(ax=ax1, color='r', lw=2., legend=True)
plt.savefig('ema.png')
plt.show()
"""

close = samsung_data['Adj Close']

num_periods_fast = 10
K_fast = 2 / (num_periods_fast + 1)
ema_fast = 0

num_periods_slow = 40
K_slow = 2 / (num_periods_slow + 1)
ema_slow = 0

ema_fast_values = []
ema_slow_values = []
apo_values = []

for close_price in close:
    if (ema_fast == 0):
        ema_fast = close_price
        ema_slow = close_price
    else:
        ema_fast = (close_price - ema_fast) * K_fast + ema_fast
        ema_slow = (close_price - ema_slow) * K_slow + ema_slow

    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)
    apo_values.append(ema_fast - ema_slow)

samsung_data = samsung_data.assign(ClosePrice=pd.Series(close, index=samsung_data.index))
samsung_data = samsung_data.assign(FastExponential10DayMovingAverage=pd.Series(ema_fast_values,
                                    index=samsung_data.index))
samsung_data = samsung_data.assign(SlowExponential40DayMovingAverage=pd.Series(ema_slow_values,
                                    index=samsung_data.index))
samsung_data = samsung_data.assign(AbsolutePriceOscillator=pd.Series(apo_values,
                                    index=samsung_data.index))
close_price = samsung_data['ClosePrice']
ema_f = samsung_data['FastExponential10DayMovingAverage']
ema_s = samsung_data['SlowExponential40DayMovingAverage']
apo = samsung_data['AbsolutePriceOscillator']

fig = plt.figure()
ax1 = fig.add_subplot(211, ylabel='Samsung')
close_price.plot(ax=ax1, color='g', lw=2., legend=True)
ema_f.plot(ax=ax1, color='b', lw=2., legend=True)
ema_s.plot(ax=ax1, color='r', lw=2., legend=True)
ax2 = fig.add_subplot(212, ylabel='APO')
apo.plot(ax=ax2, color='black', lw=2., legend=True)
#plt.savefig('ema.png')
plt.show()