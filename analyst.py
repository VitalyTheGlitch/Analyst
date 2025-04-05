import os
import sys
import ccxt
import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from colorama import Fore, Back, Style, init
from datetime import datetime

warnings.filterwarnings('ignore')
init(autoreset=True)


def read_proxies_from_file(filename='proxies.txt'):
    proxies = []

    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()

                    if line and not line.startswith('#'):
                        proxies.append(line)

            print(f'{Fore.GREEN}✅ Loaded {len(proxies)} proxies from {filename}{Style.RESET_ALL}')

        else:
            print(f'{Fore.YELLOW}⚠️  Proxy file {filename} not found{Style.RESET_ALL}')
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Error reading proxy file: {e}{Style.RESET_ALL}')

    return proxies

def get_user_input():
    while True:
        symbol = input(f'{Fore.GREEN}• Enter cryptocurrency pair (e.g., BTC/USDT): {Fore.YELLOW}')

        if symbol and '/' in symbol:
            break

        print(f'{Back.RED}{Fore.WHITE}🚫 Invalid symbol format! Use format like BTC/USDT{Style.RESET_ALL}')
    
    while True:
        start_date = input(f'{Fore.GREEN}• Enter start date (DD.MM.YYYY): {Fore.YELLOW}')
        start_date_dt = None

        try:
            start_date_dt = datetime.strptime(start_date, '%d.%m.%Y')

            if start_date_dt:
                break
        except ValueError:
            pass

        print(f'{Back.RED}{Fore.WHITE}🚫 Invalid date format! Use DD.MM.YYYY{Style.RESET_ALL}')
    
    while True:
        end_date = input(f'{Fore.GREEN}• Enter end date (DD.MM.YYYY): {Fore.YELLOW}')

        try:
            end_date_dt = datetime.strptime(end_date, '%d.%m.%Y')

            if end_date_dt:
                break
        except ValueError:
            pass

        if end_date_dt and end_date_dt > start_date_dt:
            break

        print(f'{Back.RED}{Fore.WHITE}🚫 Invalid date! Use DD.MM.YYYY and ensure it is after the start date{Style.RESET_ALL}')
    
    while True:
        interval = input(f'{Fore.GREEN}• Enter time interval (1m, 1h, 1d): {Fore.YELLOW}').lower()

        if interval in ['1m', '1h', '1d']:
            break

        print(f'{Back.RED}{Fore.WHITE}🚫 Invalid interval! Use 1m, 1h, or 1d{Style.RESET_ALL}')
    
    while True:
        simulation_days = input(f'{Fore.GREEN}• Enter simulation days: {Fore.YELLOW}')

        if simulation_days.isdigit() and int(simulation_days) > 0:
            simulation_days = int(simulation_days)

            break

        print(f'{Back.RED}{Fore.WHITE}🚫 Invalid number of simulation days!{Style.RESET_ALL}')
    
    while True:
        simulations_count = input(f'{Fore.GREEN}• Enter number of Monte Carlo simulations: {Fore.YELLOW}').strip()

        if simulations_count.isdigit() and int(simulations_count) > 0:
            simulations_count = int(simulations_count)

            break

        print(f'{Back.RED}{Fore.WHITE}🚫 Invalid number of simulation count!{Style.RESET_ALL}')
    
    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = end_date_dt.strftime('%Y-%m-%d')
    
    return symbol, start_date, end_date, interval, simulation_days, simulations_count

def get_exchange(
    proxies=None,
    exchange_name='kucoin',
    api_key=None,
    api_secret=None, 
    timeout=30,
    rate_limit=True,
    retries=3,
    verbose=False
):
    print(f'{Fore.MAGENTA}🔌 Initializing exchange connection...{Style.RESET_ALL}')
    
    try:
        supported_exchanges = {
            'kucoin': ccxt.kucoin,
            'binance': ccxt.binance,
            'bybit': ccxt.bybit,
            'okx': ccxt.okx
        }
        
        if exchange_name not in supported_exchanges:
            raise ValueError(f'Unsupported exchange: {exchange_name}. '
                           f'Supported exchanges: {list(supported_exchanges.keys())}')
            
        proxy_config = None

        if proxies:
            if isinstance(proxies, list):
                proxy = random.choice(proxies)

            else:
                proxy = proxies
                
            proxy_config = {
                'http': f'http://{proxy}',
                'https': f'http://{proxy}'
            }

            print(f'{Fore.YELLOW}   → Using proxy: {proxy}{Style.RESET_ALL}')
        
        config = {
            'timeout': timeout * 1000,
            'enableRateLimit': rate_limit,
            'proxies': proxy_config,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': timeout * 1000,
                'verbose': verbose
            }
        }
        
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret

            print(f'{Fore.YELLOW}   → API authentication enabled{Style.RESET_ALL}')
        
        exchange_class = supported_exchanges[exchange_name]
        exchange = exchange_class(config)
        
        print(f'{Fore.YELLOW}   → Testing connection...{Style.RESET_ALL}')
            
        for attempt in range(1, retries + 1):
            try:
                exchange.load_markets()

                print(f'{Fore.YELLOW}     → Successfully loaded {len(exchange.markets)} markets{Style.RESET_ALL}')

                break
            except Exception as e:
                if attempt == retries:
                    raise ConnectionError(f'Failed to connect after {retries} attempts: {str(e)}')

                print(f'{Fore.YELLOW}     → Attempt {attempt} failed, retrying...{Style.RESET_ALL}')

                time.sleep(1)
        
        print(f'{Fore.GREEN}✅ Successfully connected to {exchange_name.capitalize()}{Style.RESET_ALL}')

        return exchange
    except ValueError as ve:
        print(f'{Back.RED}{Fore.WHITE}🚫 Configuration error: {ve}{Style.RESET_ALL}')

        raise
    except ConnectionError as ce:
        print(f'{Back.RED}{Fore.WHITE}🚫 Connection error: {ce}{Style.RESET_ALL}')

        raise
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Unexpected error: {e}{Style.RESET_ALL}')

        raise

def load_data(symbol, start_date, end_date, interval, proxies=None):
    print(f'\n{Fore.MAGENTA}🔄 Loading data...{Style.RESET_ALL}')

    try:
        if interval in ['1m', '1h']:
            exchange = get_exchange(proxies)
            timeframe = f'{interval}' if interval == '1m' else f'{interval}'
            
            if timeframe not in exchange.timeframes:
                raise ValueError(f'Unsupported interval: {interval}. Supported: {exchange.timeframes}')
                
            since = exchange.parse8601(start_date + 'T00:00:00Z')
            until = exchange.parse8601(end_date + 'T00:00:00Z')
            all_ohlcv = []

            while since < until:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)

                if not ohlcv:
                    raise ValueError(f'No data returned for {symbol} from {start_date} to {end_date}')
                    
                since = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)

                if ohlcv[-1][0] > until:
                    break

            data = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)

        else:
            ticker = symbol.replace('/', '-')

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                group_by='column'
            )

            if data.empty:
                raise ValueError(f'No data found for {symbol} from {start_date} to {end_date}')
           
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.swaplevel(0, 1)
                data = data.stack(level=0).reset_index(level=1, drop=True)
                data.columns.name = None

            else:
                data.columns = [col.split('_')[0] for col in data.columns]

        print(f'{Fore.GREEN}✅ Data successfully loaded!{Style.RESET_ALL}')

        return data
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Data loading failed: {e}{Style.RESET_ALL}')
        
        sys.exit(1)

def handle_missing_data(data, method='linear', max_gap_days=3, fill_limit=None):
    print(f'{Fore.MAGENTA}🧹 Handling missing values...{Style.RESET_ALL}')
    
    try:
        missing_count = data.isnull().sum().sum()

        if missing_count == 0:
            print(f'{Fore.YELLOW}   → No missing values found{Style.RESET_ALL}')

            return data
            
        print(f'{Fore.YELLOW}   → Found {missing_count} missing values{Style.RESET_ALL}')
        
        processed_data = data.copy()
        
        if isinstance(data.index, pd.DatetimeIndex):
            time_diff = data.index.to_series().diff().dt.days
            max_gap = time_diff.max()

            print(f'{Fore.YELLOW}   → Maximum time gap: {max_gap} days{Style.RESET_ALL}')

        else:
            max_gap = None
            
        if method in ['linear', 'time', 'spline', 'nearest']:
            processed_data = processed_data.interpolate(
                method=method,
                limit=fill_limit,
                limit_direction='both'
            )
            
            if isinstance(data.index, pd.DatetimeIndex):
                processed_data = processed_data.interpolate(
                    method='time',
                    limit=fill_limit,
                    limit_direction='both'
                )
                
        elif method == 'pad':
            processed_data = processed_data.fillna(method='ffill', limit=fill_limit)
            
        else:
            raise ValueError(f'Unsupported interpolation method: {method}')
            
        if max_gap_days is not None and isinstance(data.index, pd.DatetimeIndex):
            mask = time_diff <= max_gap_days
            processed_data = processed_data[mask]
            removed_count = len(data) - len(processed_data)

            if removed_count:
                print(f'{Fore.YELLOW}   → Removed {removed_count} rows with gaps > {max_gap_days} days{Style.RESET_ALL}')
        
        remaining_missing = processed_data.isnull().sum().sum()

        if remaining_missing:
            print(f'{Fore.YELLOW}   → Warning: {remaining_missing} missing values remain{Style.RESET_ALL}')

        else:
            print(f'{Fore.GREEN}   → All missing values handled{Style.RESET_ALL}')
            
        return processed_data
        
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Error handling missing data: {e}{Style.RESET_ALL}')

        return data

def clean_data(data, threshold=1.5):
    print(f'{Fore.MAGENTA}📊 Cleaning data...{Style.RESET_ALL}')

    try:
        initial_count = len(data)
        clean_data = data[data['Close'] > 0].copy()
        removed_negative = initial_count - len(clean_data)

        if removed_negative:
            print(f'{Fore.RED}   → Removed {removed_negative} negative values ({removed_negative / initial_count * 100:.2f}% of data')

        else:
            print(f'{Fore.YELLOW}   → No negative values found{Style.RESET_ALL}')

        Q1 = data['Close'].quantile(0.25)
        Q3 = data['Close'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        final_data = clean_data[
            (clean_data['Close'] >= lower_bound) & 
            (clean_data['Close'] <= upper_bound)
        ].copy()

        removed_outliers  = len(clean_data) - len(final_data)

        if removed_outliers:
            print(f'{Fore.YELLOW}   → Removed {removed_outliers} outliers ({removed_outliers / initial_count * 100:.2f}% of data){Style.RESET_ALL}')

        else:
            print(f'{Fore.YELLOW}   → No outliers found{Style.RESET_ALL}')
            
        return final_data
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Data cleaning failed: {e}{Style.RESET_ALL}')

        sys.exit(1)

def normalize_data(data, method='minmax', feature_range=(0.0001, 1)):
    print(f'{Fore.MAGENTA}📈 Normalizing data...{Style.RESET_ALL}')

    try:
        columns = data.select_dtypes(include=['number']).columns.tolist()
        close_prices = data['Close'].copy()
        columns.remove('Close')
        
        if not columns:
            print(f'{Fore.YELLOW}   → No numeric columns found for normalization{Style.RESET_ALL}')

            return data if not return_scalers else (data, {})

        print(f'{Fore.YELLOW}   → Normalizing columns: {columns}{Style.RESET_ALL}')
        
        processed_data = data.copy()

        for col in columns:
            try:
                if processed_data[col].nunique() == 1:
                    print(f'{Fore.YELLOW}→ Skipping {col} (constant value){Style.RESET_ALL}')
                    
                    continue

                epsilon = 1e-8
                temp_series = processed_data[col] + epsilon

                if method == 'minmax':
                    scaler = MinMaxScaler(feature_range=feature_range)

                elif method == 'standard':
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()

                elif method == 'robust':
                    from sklearn.preprocessing import RobustScaler

                    scaler = RobustScaler()

                elif method == 'log':
                    from sklearn.preprocessing import FunctionTransformer

                    scaler = FunctionTransformer(np.log1p, validate=True)

                elif method == 'maxabs':
                    from sklearn.preprocessing import MaxAbsScaler

                    scaler = MaxAbsScaler()

                else:
                    raise ValueError(f'Unknown normalization method: {method}')
               
                processed_data[col] = scaler.fit_transform(temp_series.values.reshape(-1, 1))
                
                print(f'{Fore.YELLOW}     → {col}: {method} normalization applied{Style.RESET_ALL}')
                print(f'{Fore.YELLOW}       Min: {processed_data[col].min():.4f}, '
                      f'Max: {processed_data[col].max():.4f}, '
                      f'Mean: {processed_data[col].mean():.4f}{Style.RESET_ALL}')
            except Exception as col_error:
                print(f'{Back.RED}{Fore.WHITE}🚫 Error normalizing column {col}: {col_error}{Style.RESET_ALL}')
                
                processed_data[col] = data[col]
                
        print(f'{Fore.GREEN}✅ Data normalization completed!{Style.RESET_ALL}')
        
        data['Close'] = close_prices

        return processed_data
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Data normalization failed: {e}{Style.RESET_ALL}')

        sys.exit(1)

def calculate_moving_averages(data, short_window=20, long_window=50):
    try:
        if 'Close' not in data.columns:
            raise ValueError('Input data must contain "Close" column')
            
        data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window).mean()
        data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window).mean()
        data[f'EMA_{short_window}'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        
        print(f'{Fore.YELLOW}→ Calculated moving averages: '
              f'SMA({short_window}), SMA({long_window}), EMA({short_window}){Style.RESET_ALL}')
        
        return data
    except Exception as e:
        print(f'{Back.RED}Error in MA calculation: {e}{Style.RESET_ALL}')
        
        return data

def calculate_rsi(data, window=14):
    try:
        if 'Close' not in data.columns:
            raise ValueError('Input data must contain "Close" column')

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        print(f'{Fore.YELLOW}→ Calculated RSI({window}){Style.RESET_ALL}')

        return data
    except Exception as e:
        print(f'{Back.RED}Error in RSI calculation: {e}{Style.RESET_ALL}')

        sys.exit(1)

def calculate_macd(data, fast=12, slow=26, signal=9):
    try:
        if 'Close' not in data.columns:
            raise ValueError('Input data must contain "Close" column')

        data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        data['Histogram'] = data['MACD'] - data['Signal']
        
        print(f'{Fore.YELLOW}→ Calculated MACD({fast},{slow},{signal}){Style.RESET_ALL}')
        
        return data
    except Exception as e:
        print(f'{Back.RED}Error in MACD calculation: {e}{Style.RESET_ALL}')
        
        sys.exit(1)

def calculate_bollinger_bands(data, window=20, num_std=2):
    try:
        if 'Close' not in data.columns:
            raise ValueError('Input data must contain "Close" column')

        close_prices = data['Close'].copy()
        
        ma = close_prices.rolling(window).mean()
        std = close_prices.rolling(window).std()
        
        data['Bollinger_MA'] = ma
        data['Bollinger_Upper'] = ma + (std * num_std)
        data['Bollinger_Lower'] = ma - (std * num_std)
        
        print(f'{Fore.YELLOW}→ Calculated Bollinger Bands({window},{num_std}){Style.RESET_ALL}')
        
        return data
    except Exception as e:
        print(f'{Back.RED}Error in Bollinger Bands: {e}{Style.RESET_ALL}')
        
        sys.exit(1)

def calculate_atr(data, window=14):
    try:
        if 'Close' not in data.columns:
            raise ValueError('Input data must contain "Close" column')

        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, high_close, low_close)
        data['ATR'] = true_range.rolling(window=window).mean()
        
        print(f'{Fore.YELLOW}→ Calculated ATR({window}){Style.RESET_ALL}')

        return data
        
    except Exception as e:
        print(f'{Back.RED}Error in ATR calculation: {e}{Style.RESET_ALL}')

        sys.exit(1)

def calculate_obv(data):
    try:
        if 'Close' not in data.columns:
            raise ValueError('Input data must contain "Close" column')

        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        data['OBV'] = obv
        
        print(f'{Fore.YELLOW}→ Calculated On-Balance Volume{Style.RESET_ALL}')
        
        return data
        
    except Exception as e:
        print(f'{Back.RED}Error in OBV calculation: {e}{Style.RESET_ALL}')
        
        return data

def calculate_indicators(data):
    print(f'{Fore.MAGENTA}🔍 Calculating technical indicators:{Style.RESET_ALL}')

    try:
        data = calculate_moving_averages(data)
        data = calculate_rsi(data)
        data = calculate_macd(data)
        data = calculate_bollinger_bands(data)
        data = calculate_atr(data)
        data = calculate_obv(data)

        return data
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Indicator calculation failed: {e}{Style.RESET_ALL}')

        sys.exit(1)

def build_lstm_model(input_shape):
    try:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Model build failed: {e}{Style.RESET_ALL}')

        sys.exit(1)

def predict_with_lstm(data, future_days):
    print(f'\n{Fore.BLUE}{Style.BRIGHT}🧠 Running LSTM model...{Style.RESET_ALL}')

    try:
        if len(data) < 60:
            raise ValueError('Insufficient data for LSTM (need at least 60 data points)')
            
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        X, y = [], []

        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = build_lstm_model((X.shape[1], 1))

        print(f'{Fore.YELLOW}🔨 Training model...')

        model.fit(X, y, batch_size=1, epochs=1, verbose=0)

        inputs = data['Close'][-60:].values
        inputs = scaler.transform(inputs.reshape(-1, 1))
        future_prices = []

        for _ in range(future_days):
            X_test = np.array([inputs[-60:]])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_price = model.predict(X_test, verbose=0)
            future_prices.append(predicted_price[0][0])
            inputs = np.append(inputs, predicted_price)

        future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

        print(f'{Fore.GREEN}✅ Successfully generated forecast!{Style.RESET_ALL}')

        return future_prices.flatten()
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Prediction failed: {e}{Style.RESET_ALL}')

        sys.exit(1)

def monte_carlo_simulation(data, days=365, simulations_count=1000,  confidence_level=0.95, ):
    print(f'\n{Fore.MAGENTA}🎲 Running Monte Carlo simulations...{Style.RESET_ALL}')
    
    try:
        if 'Close' not in data.columns:
            raise ValueError('Input data must contain "Close" column')

        if data['Close'].isnull().any() or (data['Close'] <= 0).any():
            raise ValueError('Invalid Close prices (NaN, zero, or negative values)')
        
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        
        if returns.empty:
            raise ValueError('No valid returns data (all NaN after cleaning)')

        last_price = data['Close'].iloc[-1]
        mean_return = returns.mean()
        volatility = returns.std()
        
        mean_return = mean_return if not np.isinf(mean_return) and not np.isnan(mean_return) else 0.0
        volatility = volatility if not np.isinf(volatility) and not np.isnan(volatility) else 0.01
        volatility = np.clip(volatility, 0.001, 2.0)
        drift = mean_return - (0.5 * volatility**2)

        randn = np.random.randn(days, simulations_count)
        daily_returns = np.exp(np.clip(drift + volatility * randn, -1.0, 1.0))

        price_paths = np.zeros((days + 1, simulations_count))
        price_paths[0] = last_price

        for t in range(1, days + 1):
            price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

        final_prices = price_paths[-1]
        final_prices = final_prices[(final_prices > 0) & np.isfinite(final_prices)]

        mean_final_price = np.mean(final_prices)
        median_final_price = np.median(final_prices)
        min_final_price = np.min(final_prices)
        max_final_price = np.max(final_prices)
        positive_return_prob = np.mean(final_prices > last_price) * 100
        var_95 = np.percentile(final_prices, 5)
        var_99 = np.percentile(final_prices, 1)

        print(f'{Fore.YELLOW}   → Parameters:{Style.RESET_ALL}')
        print(f'{Fore.CYAN}     - Initial Price:{Style.RESET_ALL} {last_price:.2f}')
        print(f'{Fore.CYAN}     - Simulation Days:{Style.RESET_ALL} {days}')
        print(f'{Fore.CYAN}     - Total Simulations:{Style.RESET_ALL} {simulations_count}')
        print(f'{Fore.GREEN}     - Mean Final Price:{Style.RESET_ALL} {mean_final_price:.2f}')
        print(f'{Fore.GREEN}     - Median Final Price:{Style.RESET_ALL} {median_final_price:.2f}')
        print(f'{Fore.RED}     - Minimum Final Price:{Style.RESET_ALL} {min_final_price:.2f}')
        print(f'{Fore.GREEN}     - Maximum Final Price:{Style.RESET_ALL} {max_final_price:.2f}')
        print(f'{Fore.CYAN}     - Positive Return Probability:{Style.RESET_ALL} {positive_return_prob:.1f}%')
        print(f'{Fore.RED}     - 95% Value at Risk (VaR):{Style.RESET_ALL} {var_95:.2f}')
        print(f'{Fore.RED}     - 99% Value at Risk (VaR):{Style.RESET_ALL} {var_99:.2f}')
        print(f'{Fore.YELLOW}     - Annualized Volatility:{Style.RESET_ALL} {volatility * np.sqrt(252):.2%}')
        print(f'{Fore.YELLOW}     - Expected Daily Return:{Style.RESET_ALL} {mean_return:.2%}')
        print(f'{Fore.GREEN}✅ Simulations completed successfully!{Style.RESET_ALL}')

        return price_paths
    except Exception as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Simulation failed: {e}{Style.RESET_ALL}')
        
        return

def plot_simulation_results(data, simulations, future_prices, symbol):
    try:
        plt.figure(figsize=(14, 7))
        plt.gcf().canvas.manager.set_window_title('Analyst')
        plt.plot(data.index, data['Close'], label='Historical Price', color='blue')

        simulation_days = simulations.shape[0] - 1
        simulations_count = simulations.shape[1]
        last_date = data.index[-1]

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=simulation_days,
            freq='D'
        )

        for i in range(simulations_count):
            plt.plot(
                future_dates, 
                simulations[1:, i],
                color='gray', 
                alpha=0.1,
                linewidth=0.5
            )

        lstm_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(future_prices),
            freq='D'
        )
        
        plt.plot(
            lstm_dates,
            future_prices, 
            label='LSTM Forecast', 
            color='red',
            linewidth=2
        )
      
        lower_bound = np.percentile(simulations[1:], 5, axis=1)
        upper_bound = np.percentile(simulations[1:], 95, axis=1)
        median_values = np.median(simulations[1:], axis=1)

        plt.plot(
            future_dates,
            median_values,
            label='Median Simulation',
            color='blue',
            linestyle='--'
        )

        plt.fill_between(
            future_dates,
            lower_bound,
            upper_bound,
            label='95% Confidence Interval',
            color='gray', 
            alpha=0.2
        )

        plt.title(
            f'{symbol} Rate Simulation and Forecast', 
            fontsize=16,
            fontweight='bold',
            color='navy'
        )
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.legend()
        plt.grid(True)

        print(f'\n{Fore.CYAN}{Style.BRIGHT}📊 Displaying results...{Style.RESET_ALL}')

        plt.show()
    except KeyboardInterrupt as e:
        print(f'{Back.RED}{Fore.WHITE}🚫 Visualization failed: {e}{Style.RESET_ALL}')

        sys.exit(1)


try:
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f'{Style.BRIGHT}{Fore.RED}Analyst{Fore.YELLOW}{Fore.RESET}\n')

    proxies = read_proxies_from_file()
    symbol, start_date, end_date, interval, simulation_days, simulations_count = get_user_input()
    data = load_data(symbol, start_date, end_date, interval, proxies)
    data = handle_missing_data(data)
    data = clean_data(data)
    data = normalize_data(data)
    data = calculate_indicators(data)
    future_prices = predict_with_lstm(data, simulation_days)
    simulations = monte_carlo_simulation(data, simulation_days, simulations_count)
    plot_simulation_results(data, simulations, future_prices, symbol)

    print(f'{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}🔥 Analysis completed successfully!{Style.RESET_ALL}')
except KeyboardInterrupt:
    print(f'\n{Back.YELLOW}{Fore.BLACK}⚠️ Analysis interrupted by user!{Style.RESET_ALL}')

    sys.exit(1)
except KeyboardInterrupt as e:
    print(f'{Back.RED}{Fore.WHITE}{Style.BRIGHT}💀 Critical error: {e}{Style.RESET_ALL}')

    sys.exit(1)
