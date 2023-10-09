
# In[104]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df_raw = pd.read_csv('./podatki/bicikelj_train.csv')
df_metadata = pd.read_csv('./podatki/bicikelj_metadata.csv', delimiter='\t')
#df_padavine = pd.read_csv('./podatki/padavine.csv')
#print(df_raw.shape)
#display(df_padavine.head(5))
#display(df_metadata.head(5))
def preprocess_metadata(df_metadata, col, df_raw):
    df = df_metadata[['postaja', col]].transpose()
    df = df.reset_index(drop=True)
    #df = df.set_axis(df.iloc[0], axis=1, inplace=False).iloc[1:]
    df.columns = df.iloc[0]
    # Exclude the first row
    df = df.iloc[1:]
    df = df.apply(lambda row: row.add_suffix(f'_{col}_h'), axis=1)
    #display(df)
    #display(merged_df) 
    # Repeat the rows in df2 to match the number of rows in df1
    df_repeated = pd.concat([df] * len(df_raw), ignore_index=True)

    # # Concatenate the DataFrames horizontally
    merged_df = pd.concat([df_raw, df_repeated], axis=1)

    return merged_df

df_total_space = preprocess_metadata(df_metadata, 'total_space', df_raw)

df = preprocess_metadata(df_metadata, 'total_space', df_raw)

#total_space_column = df.filter(like='total_space_h').columns[1]

#display(df_total_space.head(5))
#print(total_space_column)


# In[105]:


def add_weather_data(df):
    
    df_padavine = pd.read_csv('./podatki/padavine.csv')

    df_padavine = df_padavine.drop('station id', axis=1)
    df_padavine = df_padavine.drop(' station name', axis=1)

    df_padavine.rename(columns={' valid': 'timestamp'}, inplace=True)
    df_padavine.rename(columns={'količina padavin [mm]': 'prcp'}, inplace=True)
    df_padavine.rename(columns={'povp. T [°C]': 'avgT'}, inplace=True)

    df_padavine['timestamp'] = pd.to_datetime(df_padavine['timestamp'])

    df_padavine = df_padavine.dropna()
    

    merged_df = pd.merge_asof(df, df_padavine, on='timestamp', direction='nearest')
    #merged_df['is_rainy_hour'] = (merged_df['prcp'] > 0.5).astype(int)
    #display(merged_df.head(5))
    return merged_df


# In[106]:


# def add_total_space(df):
#     # Filter rows where 'postage' is equal to 'postaja1'
#     filtered_rows = df_metadata[df_metadata['postaja'] == column_name]
#     # Get the value of 'toatal_space' in the filtered row(s)
#     total_space_values = filtered_rows['total_space']
#     df_clip = np.clip(df_to_clip, None, total_space_values[0])
    
#     return df_clip


# In[107]:


def add_closest_times(df_ts, type, diff):
    df_closest_times = pd.read_csv(f'./generiraj/closest_{diff}h_{type}.csv')
    df_closest_times = df_closest_times.rename(columns={col: col+f'_closest_{diff}h' for col in df_closest_times.columns[1:]})

    # merge the two data frames on the "timestamp" column
    merged_df = pd.merge(df_ts, df_closest_times, on='timestamp', how='outer')
    return merged_df


# In[108]:


def add_is_rainy(df):
    df_precipitation = pd.read_csv('./podatki/export.csv')
    df_precipitation = df_precipitation.loc[:, ['date', 'prcp']] 
    # Convert the date column in precipitation_df to datetime type
    # Convert the date column in precipitation_df to datetime type
    df_precipitation['date'] = pd.to_datetime(df_precipitation['date'])

    # Group precipitation data by date and check if precipitation was more than 5
    rainy_dates = df_precipitation.groupby('date').sum()['prcp'] > 5

    # Map the is_rainy values to timestamp_df based on the corresponding date
    df['is_rainy'] = df['timestamp'].dt.date.map(rainy_dates).astype(int)
    #display(df.head(5))
    return df


# In[109]:


def manipulate_temperature(df):
    # Define the temperature categories
    #categories = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    categories = [0, 1, 2]

    # Categorize the temperature column
    df['Temperature Category'] = pd.cut(df['avgT'], bins=3, labels=categories)

    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(df['Temperature Category'], prefix='Temperature').astype(int)

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    df_encoded = pd.concat([df, one_hot_encoded], axis=1)

    # Remove the original temperature column and the temperature category column if needed
    df_encoded.drop(['avgT'], axis=1, inplace=True)
    #df_encoded.drop(['avgT', 'Temperature Category'], axis=1, inplace=True)
    
    return df_encoded


# In[110]:


df = df_raw.copy()
def preprocess_data(df, type, diff):
    #df = preprocess_metadata(df_metadata, 'total_space', df)
    # df = preprocess_metadata(df_metadata, 'geo-visina', df)
    # df = preprocess_metadata(df_metadata, 'geo-sirina', df)
    df = add_closest_times(df, type, 0 + diff)
    df = add_closest_times(df, type, 1 + diff)
    #df = add_closest_times(df, type, 3 + diff)
    #df = add_times_minus(df, "90", suff)
    # Convert 'timestamp' column to timestamptime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = add_weather_data(df)

    # Extract day, hour, minute, and second values
    df['is_august'] = (df['timestamp'].dt.month == 8).astype(int)
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    #df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    # df['sluzba'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int) #| ((df['hour'] >= 15) & (df['hour'] <= 17))
    df['is_weekend'] = ((df['dayofweek'] == 5) | (df['dayofweek'] == 6)).astype(int)
    #df['is_holiday'] = ((df['is_august'] == 1) | (df['day'] == 15)).astype(int)
    for hour in range(24):
        df[f'hour_{hour}_week'] = ((df['hour'] == hour) & (df['is_weekend'] == 0)).astype(int)
        df[f'hour_{hour}_weekend'] = ((df['hour'] == hour) & (df['is_weekend'] == 1)).astype(int)
    
    #one_hot_encoded = pd.get_dummies(df['dayofweek'], prefix='dayofweek').astype(int)
    #df = pd.concat([df, one_hot_encoded], axis=1)
    
    df = add_is_rainy(df)

    df = df.drop('day', axis=1)
    df = df.drop('hour', axis=1)
    df = df.drop('dayofweek', axis=1)
    df = df.drop('timestamp', axis=1)
    df = df.drop('avgT', axis=1)
    df = df.drop('prcp', axis=1)
    #df = manipulate_temperature(df)
    
    if type == "train":
        df = df.dropna()

    return df

    
df_1 = preprocess_data(df, "train", 1)
df_2 = preprocess_data(df, "train", 2)

dfs = [df_1, df_2]
#df_2.to_csv('./test/test.csv', index=False)


# In[111]:


# NORMALIZACIJA
from sklearn.preprocessing import Normalizer
    

def normalizacija_train(X):
    normalizer = Normalizer(norm='l2')
    normalizer.fit(X)
    normalized_data = normalizer.transform(X)
    X = pd.DataFrame(normalized_data, columns=X.columns)
    return X, normalizer

def normalizacija_test(X, normalizer):

    normalized_data = normalizer.transform(X)
    X = pd.DataFrame(normalized_data, columns=X.columns)
    return X


# In[112]:


# STANDARDIZACIJA
from sklearn.discriminant_analysis import StandardScaler



def standardizacija_train(X):
    scaler = StandardScaler()
    # fit the scaler to the data
    scaler.fit(X)
    #print(X.columns)
    # transform the data
    df_scaled = scaler.transform(X)
    # convert the scaled data back to a dataframe
    X = pd.DataFrame(df_scaled, columns=X.columns)
    return X, scaler

def standardizacija_test(X, scaler):
    
    # fit the scaler to the data
    #print(X.columns)
    # transform the data
    df_scaled = scaler.transform(X)
    # convert the scaled data back to a dataframe
    X = pd.DataFrame(df_scaled, columns=X.columns)
    return X


# In[113]:


def split_x_y(df):
    X = df.iloc[:, 83:]
    y = df.iloc[:, :83]
    
    return X, y

#X = standardizacija_train(X)
#X = normalizacija_train(X)
X_1, y_1 = split_x_y(df_1)
X_2, y_2 = split_x_y(df_2)

# display(X_1.tail(5))
# display(y_1.tail(5))


# In[114]:


from sklearn.model_selection import GridSearchCV

def get_best_model(model, X_train, y_train, param_grid):
    # Perform a grid search to find the best parameters
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Return the best model
    best_model = grid_search.best_estimator_
    return best_model


# In[115]:


from sklearn.model_selection import train_test_split

def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#X_train, X_test, y_train, y_test = split_train_test(X, y)


# In[116]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate
from scipy import stats

def cross_validation(model, X, y, n_folds=5):
# Create an instance of Leave-One-Out Cross-Validation
    # Perform cross-validation on your data
    scores = cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_absolute_error')

    # Compute the average cross-validation score for each output variable
    mean_scores = -np.mean(scores, axis=0).round(3)
    std_scores = np.std(scores, axis=0).round(3)

    #print("Cross-validation scores:", mean_scores)
    data = np.array([[mean_scores, std_scores]])
    scores_df = pd.DataFrame(data, columns=['MAE mean','MAE std'])
    #display(scores_df)
    
    return scores_df


# In[118]:


# Function to calculate the distance between two points using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface
    using the Haversine formula.
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert coordinates to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Calculate differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

def create_closest_station():
    # Load the data into a DataFrame
    data = pd.read_csv('./podatki/bicikelj_metadata.csv', delimiter='\t')

    # Create a new column to store the index of the closest station
    data['closest_station'] = np.nan

    # Iterate over each station
    for i in range(len(data)):
        lat1 = data.loc[i, 'geo-sirina']
        lon1 = data.loc[i, 'geo-visina']
        min_distance = np.inf
        closest_station_index = None
        
        # Compare the current station with all other stations
        for j in range(len(data)):
            if i != j:
                lat2 = data.loc[j, 'geo-sirina']
                lon2 = data.loc[j, 'geo-visina']
                
                # Calculate the distance between the stations
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                
                # Check if the current distance is smaller than the minimum distance
                if distance < min_distance:
                    min_distance = distance
                    closest_station_index = j
        
        # Assign the index of the closest station to the current station's row
        data.loc[i, f'closest_station'] = data.iloc[closest_station_index, 0]

    # Print the DataFrame with the closest station information
    return data

df_closest = create_closest_station()
#display(df_closest)

def add_closest_station(x, column, ix):
    closest_station = df_closest.loc[df_closest['postaja'] == column, 'closest_station'].values[0]
    total_space = df_total_space[f"{closest_station}_total_space_h"].iloc[0]
    #x[f'{column}_closest_station_is_empty'] = (x[f'{closest_station}_closest_{ix}h'] == 0).astype(int)
    x[f'{column}_closest_station_is_full'] = (x[f'{closest_station}_closest_{ix}h'] == total_space).astype(int)
    #print(closest_station)
    return x
    


# In[119]:


def add_empty_full(x, column, ix):
    total_space = df_total_space[f"{column}_total_space_h"].iloc[0]
    x['is_empty'] = (x[f'{column}_closest_{ix}h'] == 0).astype(int)
    x['is_full'] = (x[f'{column}_closest_{ix}h'] == total_space).astype(int)
   


# ### Train model

# In[131]:


from sklearn.model_selection import cross_val_score
import copy

from sklearn.preprocessing import PolynomialFeatures

def train_model(original_model, X1, y1, X2, y2):
    models = []
    # Create a multi-output regression model
    scalers = []
    score = pd.DataFrame(np.array([[0, 0]]), columns=['MAE mean','MAE std'])

    ix = 1

    for X, Y in [[X1, y1], [X2, y2]]:
        
        single_scalers = []
        single_models = []
        
        for i, column in enumerate(Y.columns):
            x = X.copy()
            y = Y[column]
            model = copy.deepcopy(original_model)
            #model = get_best_model(model, x, y, param_grid)
           # x = add_closest_station(x, column, ix)
            x = x[[col for col in x.columns if col.startswith(column) or not col.endswith('h')]]
            total_space = df_total_space[f"{column}_total_space_h"].iloc[0]
            x['is_empty'] = (x[f'{column}_closest_{ix}h'] == 0).astype(int)
            x['is_full'] = (x[f'{column}_closest_{ix}h'] == total_space).astype(int)
            #x['prctg'] = x[f'{column}_closest_{ix}h'] / total_space
            # display(x)
            # break
            x, scaler = standardizacija_train(x)
            #scaler = None
            # feature_names = scaler.get_feature_names_out()
            # print(feature_names)
            
            single_score = cross_validation(model, x, y)
            score += single_score
            model.fit(x,y)
            
            # coefficients = model.coef_

            # # Print the feature coefficients
            # for feature, coefficient in zip(x, coefficients):
            #     print(f"Feature: {feature}, Coefficient: {coefficient}")
            # if i == 2:
            
            
            #print(model.get_params())
            single_models.append(model)
            single_scalers.append(scaler)
            
        models.append(single_models)
        scalers.append(single_scalers)
        ix += 1
        
    score = (score / Y.shape[1]) / 2
    #display(score)

    # feature_names = scalers[0][0].get_feature_names_out()
    # print(123, feature_names)
    
    return models, score, scalers


# ### Linear regression

# In[121]:


from sklearn.linear_model import LinearRegression

# model = LinearRegression()

# models_lr, score_lr, scalers = train_model(model, X_1, y_1, X_2, y_2)
    


# ### Ridge

# In[132]:


from sklearn.linear_model import Ridge

model = Ridge(alpha=0.01)

models_ridge, score_ridge, scalers = train_model(model, X_1, y_1, X_2, y_2)

models = models_ridge


# ### Lasso

# In[123]:


from sklearn.linear_model import Lasso

# model = Lasso(alpha=0.01)

# models_lasso, score_lasso, scalers = train_model(model, X_1, y_1, X_2, y_2)


# ### Random Forest

# In[124]:


from sklearn.ensemble import RandomForestRegressor


# model = RandomForestRegressor()

# models_rf, score_rf, scalers = train_model(model, X_1, y_1, X_2, y_2)


# ### XGBoost

# In[125]:


import xgboost as xgb


# X_1_xgb = xgb.DMatrix(X_1, label=y_1)
# X_2_xgb = xgb.DMatrix(X_2, label=y_2)
# xgb_model = xgb.XGBRegressor()

# models_xgb, score_xgb, scalers = train_model(xgb_model, X_1, y_1, X_2, y_2)


# ### SVM

# In[126]:


from sklearn.svm import SVR

# model = SVR()

# models_svm, score_svm = train_model(model, X_1, y_1, X_2, y_2)


# ### KNN

# In[127]:


from sklearn.neighbors import KNeighborsRegressor


# model = KNeighborsRegressor()

# models_knn, score_knn = train_model(model, X_1, y_1, X_2, y_2)


# ### Neural Net

# In[128]:


from sklearn.neural_network import MLPRegressor


# model = MLPRegressor(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', alpha=0.001, learning_rate='constant', max_iter=200)
# models_nn, score_nn, scalers = train_model(model, X_1, y_1, X_2, y_2)

# Make predictions


# ### Choose model to predict on test set

# In[129]:


models = models_ridge


# ### Predict on final test set

# In[130]:


test_df = pd.read_csv('./podatki/bicikelj_test.csv')
X_raw = test_df.iloc[:, 0]

def predict(df, single_models, single_scalers, ix):
    
    X_test, y_pred = split_x_y(df)
    column_names = y_pred.columns
    
    #y_pred = pd.DataFrame(pred, index=df.index, columns=column_names)
    for i, model in enumerate(single_models):
        #display(X_test.head(40))
        scaler = single_scalers[i]
        # feature_names = scaler.get_feature_names_out()
        # print(123, feature_names)
        # break 
        column = y_pred.columns[i]
        x_test = X_test.copy()
        #x_test = add_closest_station(x_test, column, ix)

        x_test = x_test[[col for col in x_test.columns if col.startswith(y_pred.columns[i]) or not col.endswith('h')]]
        
        total_space = df_total_space[f"{y_pred.columns[i]}_total_space_h"].iloc[0]
        x_test['is_empty'] = (x_test[f'{column}_closest_{ix}h'] == 0).astype(int)
        x_test['is_full'] = (x_test[f'{column}_closest_{ix}h'] == total_space).astype(int)
        x_test = standardizacija_test(x_test, scaler)
            
        #print(total_space.iloc[0])
        
        pred = model.predict(x_test).clip(min=0, max=total_space).round(0)
        
        #pred = clip_by_total(pred, y_pred.columns[i])
        y_pred.iloc[:, i] = pred
        #pred =  np.round(pred).clip(min=0).astype(int)
        
    
    #display(y_pred)
    return y_pred

pred_dfs = []
ix = 1
for i, single_models in enumerate(models):
    df = test_df.copy()
    X_ts = X_raw.copy()
    df = preprocess_data(df, "test", i + 1)
    df = df.drop(index=df.index[(i + 1) % 2::2])
    X_ts = X_ts.drop(index=X_ts.index[(i + 1) % 2::2])
    y_pred_df = predict(df, single_models, scalers[i], ix)
    pred_df = pd.concat([X_ts, y_pred_df], axis=1)
    pred_df = pred_df.dropna()
    pred_dfs.append(pred_df)
    ix += 1

# Combine the two subsets into a new DataFrame
new_df = pd.concat([pred_dfs[0], pred_dfs[1]], ignore_index=True)
new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
final_df = new_df.sort_values(by='timestamp')
print(final_df.head(40))

final_df.to_csv('./output/bicikelj_test_oddaja.csv', index=False)

