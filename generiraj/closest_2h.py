import numpy as np
import pandas as pd
import datetime

train = pd.read_csv("../podatki/bicikelj_train.csv")
train["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in train["timestamp"].values]

train2 = pd.read_csv("../podatki/bicikelj_train.csv")
train2["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in train2["timestamp"].values]

test = pd.read_csv("../podatki/bicikelj_test.csv")
test["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in test["timestamp"].values]

test2 = pd.read_csv("../podatki/bicikelj_test.csv")
test2["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in test2["timestamp"].values]

times = train["timestamp"].values
times2 = train["timestamp"].values - pd.Timedelta(hours=2)
ptimes = test["timestamp"].values - pd.Timedelta(hours=2)
ptimes2 = test["timestamp"].values - pd.Timedelta(hours=2)

# delta = datetime.timedelta(hours=1)

# for i, t in enumerate(times):
#     closest = np.argmin(np.abs(times - (t - delta)))
#     train2.iloc[i, 1:] = train.iloc[closest, 1:]
#test['timestamp2'] = np.nan
for i, t in enumerate(ptimes):
    if i % 2 == 1:
        t = ptimes2[i]
    closest_times = np.abs(times[times < t] - t)
    if len(closest_times) == 0:
        test.iloc[i, 1:] = np.nan
        continue
    closest = np.argmin(closest_times)
    if abs(times[closest] - t) > pd.Timedelta(hours=2):
        # if the closest time is more than an hour away from t, assign only NaN values
        test.iloc[i, 1:] = np.nan
    else:
        # otherwise, assign values from train
        test.iloc[i, 1:] = train.iloc[closest, 1:]
        
#train2['timestamp2'] = np.nan
for i, t in enumerate(times2):
    closest_times = np.abs(times[times - t < pd.Timedelta(minutes=5)] - t)
    if len(closest_times) == 0:
        train2.iloc[i, 1:] = np.nan
        continue
    closest = np.argmin(closest_times)
    if abs(times[closest] - t) > pd.Timedelta(hours=2):
        # if the closest time is more than an hour away from t, assign only NaN values
        train2.iloc[i, 1:] = np.nan
    else:
        # otherwise, assign values from train
        train2.iloc[i, 1:] = train.iloc[closest, 1:]


train2.to_csv("closest_2h_train.csv", sep=",", index=False)
test.to_csv("closest_2h_test.csv", sep=",", index=False)