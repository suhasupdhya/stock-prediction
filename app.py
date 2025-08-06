import streamlit as st  # The "AI box" (Streamlit's UI) appears on top because Streamlit renders elements in the order they are called in the script, starting from the top. The import itself does not affect layout, but all Streamlit commands (like st.title, st.sidebar, etc.) are executed sequentially from the top of the file, so UI elements appear in that order.
from datetime import date
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Try to import XGBoost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost not available. Install with: `pip install xgboost`")

st.title('AI Stock Predictor - Multi-Model Comparison')

st.sidebar.header('Stock Prediction Settings')
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', value=date(2015, 1, 1))
end_date = st.sidebar.date_input('End Date', value=date.today())

# Model selection
st.sidebar.subheader('Model Selection')
use_random_forest = st.sidebar.checkbox('Random Forest', value=True)
use_xgboost = st.sidebar.checkbox('XGBoost', value=True, disabled=not XGBOOST_AVAILABLE)
use_linear_regression = st.sidebar.checkbox('Linear Regression', value=True)
use_ensemble = st.sidebar.checkbox('Ensemble (Average)', value=True)

if not XGBOOST_AVAILABLE and use_xgboost:
    st.error("❌ XGBoost is not available. Please install it or uncheck the XGBoost option.")
    st.stop()

# Sidebar controls for model parameters
st.sidebar.subheader('Model Parameters')
n_estimators = st.sidebar.slider('Number of Trees (RF/XGB)', min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider('Max Depth (RF/XGB)', min_value=3, max_value=20, value=10, step=1)
random_state = st.sidebar.slider('Random State', min_value=0, max_value=100, value=42, step=1)

# Sidebar controls for feature selection
st.sidebar.subheader('Features to Include')
use_volume = st.sidebar.checkbox('Volume', value=True)
use_sma = st.sidebar.checkbox('SMA (20)', value=True)
use_ema = st.sidebar.checkbox('EMA (20)', value=True)
use_rsi = st.sidebar.checkbox('RSI (14)', value=True)
use_macd = st.sidebar.checkbox('MACD', value=True)

if st.sidebar.button('Predict'):
    st.write('Fetching data and training models...')
    with st.spinner('Downloading stock data...'):
        data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error('No data found for the given ticker and date range.')
    else:
        st.subheader('Raw Stock Data')
        st.dataframe(data)

        # Feature Engineering: Add technical indicators
        st.subheader('Feature Engineering')
        df = data.copy()
        features = ['Close']
        if use_volume:
            features.append('Volume')
        if use_sma:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            features.append('SMA_20')
        if use_ema:
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            features.append('EMA_20')
        if use_rsi:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            features.append('RSI_14')
        if use_macd:
            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            features.append('MACD')
        df = df[features].dropna()
        st.write('Features with Technical Indicators:')
        st.dataframe(df)

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        
        # Create separate scaler for target variable (Close price)
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaled = close_scaler.fit_transform(df[['Close']])
        
        st.write('Scaled Features:')
        st.line_chart(scaled_data)

        # Create sequences for prediction (using last 60 days to predict next day)
        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i].flatten())  # Flatten for models
            y.append(close_scaled[i, 0])  # Predict 'Close' price
        X, y = np.array(X), np.array(y)
        st.write(f'Shape of X: {X.shape}, Shape of y: {y.shape}')

        # Display selected parameters and features
        st.subheader('Selected Model Parameters')
        st.write(f'Number of Trees: {n_estimators}, Max Depth: {max_depth}, Random State: {random_state}')
        st.write('Selected Features:', features)

        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Initialize models
        models = {}
        model_names = []
        
        if use_random_forest:
            models['Random Forest'] = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model_names.append('Random Forest')
            
        if use_xgboost and XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model_names.append('XGBoost')
            
        if use_linear_regression:
            models['Linear Regression'] = LinearRegression()
            model_names.append('Linear Regression')

        # Train models
        with st.spinner('Training models...'):
            for name, model in models.items():
                model.fit(X_train, y_train)

        # Predict and evaluate
        results = {}
        accuracy_df = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R²'])
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_rescaled = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test_rescaled = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
            r2 = r2_score(y_test_rescaled, y_pred_rescaled)
            
            results[name] = {
                'predictions': y_pred_rescaled,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            accuracy_df = pd.concat([accuracy_df, pd.DataFrame({
                'Model': [name],
                'RMSE': [rmse],
                'MAE': [mae],
                'R²': [r2]
            })], ignore_index=True)

        # Ensemble predictions
        if use_ensemble and len(models) > 1:
            ensemble_pred = np.mean([results[name]['predictions'] for name in models.keys()], axis=0)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test_rescaled, ensemble_pred))
            ensemble_mae = mean_absolute_error(y_test_rescaled, ensemble_pred)
            ensemble_r2 = r2_score(y_test_rescaled, ensemble_pred)
            
            results['Ensemble'] = {
                'predictions': ensemble_pred,
                'rmse': ensemble_rmse,
                'mae': ensemble_mae,
                'r2': ensemble_r2
            }
            
            accuracy_df = pd.concat([accuracy_df, pd.DataFrame({
                'Model': ['Ensemble'],
                'RMSE': [ensemble_rmse],
                'MAE': [ensemble_mae],
                'R²': [ensemble_r2]
            })], ignore_index=True)

        # Display accuracy comparison
        st.subheader('Model Accuracy Comparison')
        st.dataframe(accuracy_df.sort_values('R²', ascending=False))
        
        # Find best model
        best_model = accuracy_df.loc[accuracy_df['R²'].idxmax(), 'Model']
        st.success(f'🏆 Best Model: {best_model} (R² = {accuracy_df.loc[accuracy_df["R²"].idxmax(), "R²"]:.3f})')

        # Plot all model predictions
        st.subheader('Model Predictions Comparison')
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            y=y_test_rescaled, 
            mode='lines', 
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        # Add model predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (name, result) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                y=result['predictions'], 
                mode='lines', 
                name=f'{name} (R²={result["r2"]:.3f})',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            xaxis_title='Time', 
            yaxis_title='Price', 
            title=f'{ticker} - Model Predictions Comparison'
        )
        st.plotly_chart(fig)

        # Download results
        all_predictions_df = pd.DataFrame({
            'Actual': y_test_rescaled
        })
        for name, result in results.items():
            all_predictions_df[f'{name}_Predicted'] = result['predictions']
        
        csv = all_predictions_df.to_csv(index=False)
        st.download_button(
            label='Download All Model Predictions as CSV',
            data=csv,
            file_name=f'{ticker}_all_model_predictions.csv',
            mime='text/csv'
        )

        # Future Predictions using best model
        st.subheader(f'Future Predictions (Next 30 Days) - Using {best_model}')
        
        # Get the last sequence for future prediction
        last_sequence = scaled_data[-sequence_length:].flatten().reshape(1, -1)
        
        # Predict next 30 days using best model
        best_model_instance = models.get(best_model)
        if best_model_instance is None and best_model == 'Ensemble':
            # Use ensemble for future predictions
            future_predictions = []
            current_sequence = last_sequence.copy()
            
            for day in range(30):
                # Get predictions from all models
                model_preds = []
                for name, model in models.items():
                    pred = model.predict(current_sequence)[0]
                    model_preds.append(pred)
                
                # Average predictions
                next_pred = np.mean(model_preds)
                future_predictions.append(next_pred)
                
                # Update sequence
                new_row = scaled_data[-1].copy()
                new_row[0] = next_pred
                current_sequence = np.roll(current_sequence, -len(scaled_data[-1]))
                current_sequence[0, -len(scaled_data[-1]):] = new_row
        else:
            # Use single best model
            future_predictions = []
            current_sequence = last_sequence.copy()
            
            for day in range(30):
                next_pred = best_model_instance.predict(current_sequence)[0]
                future_predictions.append(next_pred)
                
                new_row = scaled_data[-1].copy()
                new_row[0] = next_pred
                current_sequence = np.roll(current_sequence, -len(scaled_data[-1]))
                current_sequence[0, -len(scaled_data[-1]):] = new_row
        
        # Rescale future predictions
        future_predictions_rescaled = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        
        # Display future predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions_rescaled
        })
        st.write('Future Predictions:')
        st.dataframe(future_df)
        
        # Download future predictions
        future_csv = future_df.to_csv(index=False)
        st.download_button(
            label='Download Future Predictions as CSV',
            data=future_csv,
            file_name=f'{ticker}_future_predictions_{best_model}.csv',
            mime='text/csv'
        )
        
        # Plot future predictions
        st.subheader('Future Price Predictions')
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions_rescaled, 
            mode='lines+markers', 
            name=f'Predicted Future Prices ({best_model})',
            line=dict(color='red', width=2)
        ))
        fig_future.update_layout(
            xaxis_title='Date', 
            yaxis_title='Predicted Price', 
            title=f'{ticker} - 30 Day Price Forecast ({best_model})'
        )
        st.plotly_chart(fig_future)

        # Feature importance (for tree-based models)
        if use_random_forest or (use_xgboost and XGBOOST_AVAILABLE):
            st.subheader('Feature Importance (Tree-based Models)')
            feature_names = [f'Day_{i}_{feat}' for i in range(sequence_length) for feat in features]
            
            for name in ['Random Forest', 'XGBoost']:
                if name in models:
                    importance = models[name].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(importance)],
                        'Importance': importance
                    }).sort_values('Importance', ascending=False).head(20)
                    
                    st.write(f'{name} Feature Importance:')
                    st.bar_chart(importance_df.set_index('Feature')['Importance'])

else:
    st.write('Enter stock ticker and date range, then click Predict.') 