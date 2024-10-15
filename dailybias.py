import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# fetch data from yahoo
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.set_index('Date', inplace=True)  # indexing by date
    return data

# swing points determination
def find_swing_points(data):
    swing_highs = []
    swing_lows = []
    for i in range(1, len(data)-1):
        if data['High'].iloc[i] > data['High'].iloc[i-1] and data['High'].iloc[i] > data['High'].iloc[i+1]:
            swing_highs.append((data.index[i], data['High'].iloc[i]))
        if data['Low'].iloc[i] < data['Low'].iloc[i-1] and data['Low'].iloc[i] < data['Low'].iloc[i+1]:
            swing_lows.append((data.index[i], data['Low'].iloc[i]))
    return swing_highs, swing_lows

# is swing point unliquidated?
def is_unliquidated_swing(swing_point, data, type='high'):
    swing_date = swing_point[0]
    try:
        current_index = data.index.get_loc(swing_date)
    except KeyError:
        print(f"Swing date {swing_date} not found in data index.")
        return False
    
    for i in range(current_index + 1, len(data)):
        if type == 'high' and data['High'].iloc[i] > swing_point[1]:
            return False
        if type == 'low' and data['Low'].iloc[i] < swing_point[1]:
            return False
    return True

# Add features to the dataset
def add_features(data, swing_highs, swing_lows):
    data['SwingHigh'] = 0
    data['SwingLow'] = 0
    for date, high in swing_highs:
        data.at[date, 'SwingHigh'] = 1
    for date, low in swing_lows:
        data.at[date, 'SwingLow'] = 1
    return data

# Function to determine if a swing point has been raided in the last 40 days
def has_been_raided(swing_point, data, current_index, window=40, type='high'):
    start = current_index + 1
    end = min(current_index + window + 1, len(data))
    if type == 'high':
        return any(data['High'].iloc[start:end] > swing_point[1])
    else:
        return any(data['Low'].iloc[start:end] < swing_point[1])

# add raid feature 
def add_raid_feature(data, window=40):
    data['SwingHigh_Raided'] = 0
    data['SwingLow_Raided'] = 0
    for i, (date, row) in enumerate(data.iterrows()):
        if row['SwingHigh'] == 1:
            swung = has_been_raided((date, row['High']), data, i, window, 'high')
            data.at[date, 'SwingHigh_Raided'] = 1 if swung else 0
        if row['SwingLow'] == 1:
            swung = has_been_raided((date, row['Low']), data, i, window, 'low')
            data.at[date, 'SwingLow_Raided'] = 1 if swung else 0
    return data

def prepare_features(data):
    # Generate target (1 for green day, 0 for red day)
    data['Target'] = np.where(data['Close'] > data['Open'], 1, 0)
    return data

# train model function
def train_model(train_data, features, target='Target'):
    X_train = train_data[features]
    y_train = train_data[target]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Main execution
if __name__ == "__main__":
    ticker = "NQ=F"
    
    # define parameters
    initial_train_days = 100
    prediction_days = 100  # Remaining days
    window = 40  # Swing points window
    
    # calculate date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=initial_train_days + prediction_days + window + 100)  # Buffer for swing points
    
    # fetch initial data
    data = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # find swing points
    swing_highs, swing_lows = find_swing_points(data)
    
    # filter out unliquidated swing points
    unliquidated_swing_highs = [high for high in swing_highs if is_unliquidated_swing(high, data, 'high')]
    unliquidated_swing_lows = [low for low in swing_lows if is_unliquidated_swing(low, data, 'low')]
    
    # add swing point features
    data = add_features(data, unliquidated_swing_highs, unliquidated_swing_lows)
    
    # add raid features
    data = add_raid_feature(data, window)
    
    # prepare features and target
    data = prepare_features(data)
    
    # ensure there are enough data points
    total_required_days = initial_train_days + prediction_days
    if len(data) < total_required_days:
        print(f"Not enough data fetched. Required: {total_required_days}, Fetched: {len(data)}")
        exit(1)
    
    # select the last 200 days
    data = data.tail(total_required_days)
    
    # initial 100 days for training then start prediction and keep retraining
    initial_train = data.iloc[:initial_train_days]
    prediction_period = data.iloc[initial_train_days:]
    
    # feature columns
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SwingHigh', 'SwingLow', 'SwingHigh_Raided', 'SwingLow_Raided']
    
    # results storage
    actuals = []
    predictions = []
    dates = []
    
    # satrt prediction and retraining loop
    for i in range(prediction_days):
        
        train_data = initial_train.copy()
        
        
        model = train_model(train_data, features)
        
        current_day = prediction_period.iloc[i]
        current_date = prediction_period.index[i]
        
        X_current = current_day[features].values.reshape(1, -1)
        
        pred = model.predict(X_current)[0]
        
        # store results
        dates.append(current_date)
        predictions.append(pred)
        actuals.append(current_day['Target'])
        
        # print results
        actual_label = "green" if current_day['Target'] == 1 else "red"
        predicted_label = "green" if pred == 1 else "red"
        print(f"Date: {current_date.strftime('%Y-%m-%d')}, Actual: {actual_label}, Predicted: {predicted_label}")
        
        # update training data
        initial_train = initial_train._append(current_day)
    
    # calculate overall accuracy
    accuracy = accuracy_score(actuals, predictions)
    print(f"\nOverall Accuracy over {prediction_days} days: {accuracy:.2%}")
    
    
    #  predict today
    latest_end_date = end_date
    latest_start_date = latest_end_date - timedelta(days=1)
    latest_data = fetch_data(ticker, latest_start_date.strftime('%Y-%m-%d'), latest_end_date.strftime('%Y-%m-%d'))
    
    if latest_data.empty:
        print("No new data available for today to make a prediction.")
    else:
        # incorporate the latest data into the dataset
        data = data._append(latest_data)
        
        # adjust swing points with the updated data
        swing_highs, swing_lows = find_swing_points(data)
        unliquidated_swing_highs = [high for high in swing_highs if is_unliquidated_swing(high, data, 'high')]
        unliquidated_swing_lows = [low for low in swing_lows if is_unliquidated_swing(low, data, 'low')]
        data = add_features(data, unliquidated_swing_highs, unliquidated_swing_lows)
        data = add_raid_feature(data, window)
        data = prepare_features(data)
        
        # train the final model with all available data
        final_train = data.tail(initial_train_days + prediction_days + 1)  # +1 for today
        final_model = train_model(final_train, features)
        
        # prepare features for today (the last row)
        today_data = final_train.iloc[-1]
        X_today = today_data[features].values.reshape(1, -1)
        
        # make prediction for today
        today_pred = final_model.predict(X_today)[0]
        today_label = "green" if today_pred == 1 else "red"
        print(f"\nToday's Prediction: {today_label}")
