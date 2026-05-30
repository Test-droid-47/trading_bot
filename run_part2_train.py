#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engine import FeatureEngine
from smart_money import SmartMoneyEngine
from alpha_factors import AlphaFactorEngine
from regime_detector import MarketRegimeDetector
from shap_selector import SHAPFeatureSelector
from optuna_tuner import OptunaTuner
from prediction_model import PredictionModel
from ensemble_model import EnsembleModel
from ppo_agent import PPOAgent
from trading_env import TradingEnvironment
from brain_manager import brain
from utils_math import correct_sharpe, max_drawdown

class TrainingPipeline:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.start_time = None
        self.stats = {
            'success': False,
            'duration_seconds': 0,
            'ohlcv_bars': 0,
            'features_count': 0,
            'selected_features': 0,
            'lstm_trained': False,
            'ensemble_trained': False,
            'ppo_trained': False,
            'regime_detector_fitted': False
        }

    def _load_config(self, config_path: str = None) -> dict:
        paths_to_try = [
            config_path,
            os.path.join(os.path.dirname(__file__), '..', 'config.json'),
            os.path.join(os.path.dirname(__file__), 'config.json'),
            'config.json'
        ]
        
        for path in paths_to_try:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    cfg = json.load(f)
                print(f"✅ Config loaded from {path}")
                return cfg
        
        print("⚠️ No config.json found. Using defaults.")
        return {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'window': 120,
            'train_split': 0.8,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lstm_units_1': 128,
            'lstm_units_2': 64,
            'attention_heads': 8,
            'attention_key_dim': 64,
            'dropout_rate': 0.2,
            'optuna_trials': 30,
            'shap_top_k': 150,
            'rl_n_episodes': 200
        }

    def load_data(self, data_dir: str = None) -> pd.DataFrame:
        if data_dir and os.path.isdir(data_dir):
            ohlcv_path = os.path.join(data_dir, 'ohlcv_data.csv')
            fg_path = os.path.join(data_dir, 'fear_greed_data.csv')
            
            if not os.path.exists(ohlcv_path):
                raise FileNotFoundError(f"OHLCV file not found: {ohlcv_path}")
            
            print(f"Loading OHLCV data from {ohlcv_path}")
            df = pd.read_csv(ohlcv_path)
            
            if os.path.exists(fg_path):
                print(f"Loading Fear & Greed data from {fg_path}")
                fg_df = pd.read_csv(fg_path)
                print(f"Loaded {len(fg_df)} Fear & Greed records")
        else:
            data_path = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ohlcv_data.csv')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}. Run Part 1 first.")
            print(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        print(f"Loaded {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        self.stats['ohlcv_bars'] = len(df)
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("STEP 1: Building Features")
        print("=" * 60)
        
        print("Adding trend, momentum, volatility indicators...")
        df = FeatureEngine.build_all(df)
        
        print("Adding Smart Money Concepts...")
        smc = SmartMoneyEngine(self.config)
        df = smc.build_all(df)
        
        print("Adding Alpha Factors (Hurst, Entropy, Efficiency)...")
        alpha = AlphaFactorEngine(self.config)
        df = alpha.build_all(df)
        
        print(f"Total features: {len(df.columns)}")
        self.stats['features_count'] = len(df.columns)
        
        return df

    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("STEP 2: Market Regime Detection")
        print("=" * 60)
        
        regime_detector = MarketRegimeDetector(self.config)
        regime_detector.fit(df)
        df = regime_detector.annotate(df)
        regime_detector.save_map()
        
        print(f"Regime distribution:")
        regime_counts = df['regime'].value_counts().sort_index()
        for regime, count in regime_counts.items():
            name = MarketRegimeDetector.REGIME_NAMES.get(regime, 'Unknown')
            print(f"  {name}: {count} bars ({count/len(df)*100:.1f}%)")
        
        self.stats['regime_detector_fitted'] = True
        return df

    def select_features(self, df: pd.DataFrame) -> list:
        print("\n" + "=" * 60)
        print("STEP 3: Boruta + SHAP Feature Selection")
        print("=" * 60)
        
        print("Running Boruta feature selection...")
        boruta_features = brain.run_boruta_flow(df)
        print(f"Boruta selected: {len(boruta_features)} features")
        if boruta_features:
            print(f"Boruta features: {boruta_features[:10]}...")
        
        # Run SHAP on Boruta results if available
        if boruta_features:
            print("\nRunning SHAP feature importance...")
            shap_selector = SHAPFeatureSelector(self.config)
            selected_features = shap_selector.fit_select(df[boruta_features])
        else:
            selected_features = []
        
        # Category-wise Top Performers (best features from each category)
        category_top_features = {
            'trend': ['ema_20', 'adx'],
            'momentum': ['rsi_14', 'macd'],
            'volatility': ['atr'],
            'volume': ['obv', 'vol_ratio'],
            'price': ['close', 'high', 'low'],
            'statistical': ['close_zscore_20', 'hurst_exp'],
            'lagged': ['ret_5', 'log_ret_10'],
            'smc': ['smc_score']
        }
        
        # Start with Boruta+SHAP selected features
        final_features = selected_features.copy() if selected_features else []
        
        # Add top performers from each category (if not already present)
        print("\nAdding top performers from each category:")
        for category, features in category_top_features.items():
            for f in features:
                if f in df.columns and f not in final_features:
                    final_features.append(f)
                    print(f"  ➕ Added {f} ({category})")
                    break  # Add only top 1 from each category
        
        # Ensure close is always present
        if 'close' not in final_features and 'close' in df.columns:
            final_features.append('close')
            print(f"  ➕ Added close (core feature)")
        
        # Remove duplicates while preserving order
        final_features = list(dict.fromkeys(final_features))
        
        print(f"\n✅ Final selected features: {len(final_features)}")
        print(f"Features: {final_features[:20]}...")
        
        self.stats['selected_features'] = len(final_features)
        return final_features

    def train_lstm(self, df: pd.DataFrame, feature_cols: list) -> PredictionModel:
        print("\n" + "=" * 60)
        print("STEP 4: Training LSTM + Transformer")
        print("=" * 60)
        
        model = PredictionModel(self.config)
        
        result = model.prepare_data(df, feature_cols=feature_cols)
        if result is None:
            raise RuntimeError("Failed to prepare data for LSTM")
        
        X_train, X_val, y_train, y_val, _, _ = result
        
        if X_train.shape[0] == 0:
            raise RuntimeError("No training data")
        
        if self.config.get('optuna_trials', 0) > 0:
            print("Running Optuna hyperparameter tuning...")
            tuner = OptunaTuner(self.config)
            tuned_cfg = tuner.tune(X_train, y_train, X_val, y_val)
            model.cfg = {**self.config, **tuned_cfg}
        
        model.build((X_train.shape[1], X_train.shape[2]))
        model.train(X_train, X_val, y_train, y_val)
        
        self.stats['lstm_trained'] = True
        return model

    def train_ensemble(self, df: pd.DataFrame) -> EnsembleModel:
        print("\n" + "=" * 60)
        print("STEP 5: Training Ensemble (XGBoost + LightGBM)")
        print("=" * 60)
        
        ensemble = EnsembleModel(self.config)
        ensemble.train(df)
        
        self.stats['ensemble_trained'] = True
        return ensemble

    def train_ppo(self, df: pd.DataFrame, pred_model: PredictionModel, feature_cols: list) -> PPOAgent:
        print("\n" + "=" * 60)
        print("STEP 6: Training PPO Agent")
        print("=" * 60)
        
        numeric_df = df.select_dtypes(include=[np.number])
        data = numeric_df[feature_cols].copy()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.ffill(inplace=True)
        data.fillna(0.0, inplace=True)
        
        scaled = pred_model.scaler.transform(data).astype(np.float32)
        close_idx = feature_cols.index('close') if 'close' in feature_cols else 0
        
        from trading_env import TradingEnvironment
        env = TradingEnvironment(df, scaled, self.config, close_idx)
        state_shape = (self.config['window'], scaled.shape[1])
        
        ppo = PPOAgent(self.config, state_shape=state_shape)
        ppo.train(env)
        
        self.stats['ppo_trained'] = True
        return ppo

    def save_models(self, pred_model: PredictionModel, ensemble: EnsembleModel, ppo: PPOAgent):
        print("\n" + "=" * 60)
        print("STEP 7: Saving Models")
        print("=" * 60)
        
        os.makedirs('models', exist_ok=True)
        
        pred_model.save('models/lstm_model.keras')
        ensemble.save('models/ensemble_model.pkl')
        ppo.save('models/ppo_agent')
        
        print("✅ All models saved to 'models/' directory")

    def run(self, data_dir: str = None) -> dict:
        self.start_time = time.time()
        
        print("=" * 70)
        print("PART 2: MODEL TRAINING PIPELINE")
        print("=" * 70)
        print(f"Symbol: {self.config.get('symbol', 'BTC/USDT')}")
        print(f"Timeframe: {self.config.get('timeframe', '1h')}")
        print(f"Window Size: {self.config.get('window', 120)} bars")
        print("=" * 70)
        
        try:
            df = self.load_data(data_dir)
            df = self.build_features(df)
            df = self.detect_regimes(df)
            
            feature_cols = self.select_features(df)
            
            pred_model = self.train_lstm(df, feature_cols)
            ensemble = self.train_ensemble(df)
            ppo = self.train_ppo(df, pred_model, feature_cols)
            
            self.save_models(pred_model, ensemble, ppo)
            
            self.stats['duration_seconds'] = round(time.time() - self.start_time, 2)
            self.stats['success'] = True
            
            print("\n" + "=" * 70)
            print("✅ PART 2 - TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Duration: {self.stats['duration_seconds']} seconds")
            print(f"OHLCV Bars: {self.stats['ohlcv_bars']:,}")
            print(f"Total Features: {self.stats['features_count']}")
            print(f"Selected Features: {self.stats['selected_features']}")
            print(f"LSTM: {'✅' if self.stats['lstm_trained'] else '❌'}")
            print(f"Ensemble: {'✅' if self.stats['ensemble_trained'] else '❌'}")
            print(f"PPO: {'✅' if self.stats['ppo_trained'] else '❌'}")
            print(f"Regime Detector: {'✅' if self.stats['regime_detector_fitted'] else '❌'}")
            print("=" * 70)
            
            return self.stats
            
        except Exception as e:
            self.stats['success'] = False
            self.stats['error'] = str(e)
            print(f"\n❌ TRAINING FAILED: {e}")
            import traceback
            traceback.print_exc()
            return self.stats

def main():
    parser = argparse.ArgumentParser(description='Part 2: Model Training Pipeline')
    parser.add_argument('--data', type=str, default=None, help='Data directory or CSV file path')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(config_path=args.config)
    result = pipeline.run(data_dir=args.data)
    
    return 0 if result['success'] else 1

if __name__ == '__main__':
    exit(main())