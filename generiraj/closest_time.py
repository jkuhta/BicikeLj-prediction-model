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
times2 = train["timestamp"].values
ptimes = test["timestamp"].values

# delta = datetime.timedelta(hours=1)

# for i, t in enumerate(times):
#     closest = np.argmin(np.abs(times - (t - delta)))
#     train2.iloc[i, 1:] = train.iloc[closest, 1:]

for i, t in enumerate(ptimes):
    closest = np.argmin(np.abs(times[times != t] - t))
    
    test.iloc[i, 1:] = train.iloc[closest, 1:]

for i, t in enumerate(times2):
    closest = np.argmin(np.abs(times - t))
    train2.iloc[i, 1:] = train.iloc[closest, 1:]


train2.to_csv("closest_time_train.csv", sep=",", index=False)
test.to_csv("closest_time_test.csv", sep=",", index=False)