import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger('BrainManager')
logger.setLevel(logging.INFO)

class FeatureBrain:
    def __init__(self, feature_file="models/selected_features.json"):
        self.feature_file = feature_file
        self.selected_features = []
        self.last_update = None
        self.load_state()

    def load_state(self):
        if os.path.exists(self.feature_file):
            try:
                with open(self.feature_file, 'r') as f:
                    data = json.load(f)
                self.selected_features = data.get('features', [])
                last_update_str = data.get('last_update')
                if last_update_str:
                    self.last_update = datetime.fromisoformat(last_update_str)
                logger.info(f"Brain: Loaded {len(self.selected_features)} cached features")
            except Exception as e:
                logger.warning(f"Brain: Failed to load state: {e}")
                self.selected_features = []
                self.last_update = None

    def save_state(self):
        data = {
            'features': self.selected_features,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        try:
            # Create models folder if it doesn't exist to prevent Errno 2
            dir_name = os.path.dirname(self.feature_file)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                
            with open(self.feature_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.last_update = datetime.now(timezone.utc)
            logger.info(f"✅ Brain: Saved {len(self.selected_features)} features to {self.feature_file}")
        except Exception as e:
            logger.error(f"Brain: Failed to save state: {e}")

    def should_run_update(self):
        now = datetime.now(timezone.utc)
        if self.last_update is None:
            return True

        days_since = (now - self.last_update).days
        is_sunday = now.weekday() == 6
        return is_sunday and days_since >= 7

    def run_boruta_flow(self, df):
        logger.info("🚀 Brain: Starting Boruta Feature Selection...")

        if 'close' not in df.columns:
            logger.error("Brain: 'close' column missing. Cannot create target.")
            return self.selected_features

        # 1. Sort index to fix time sequence issues (VWAP, etc.)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # 2. Select numeric columns and drop completely empty ones
        X = df.select_dtypes(include=[np.number]).copy()
        X.dropna(axis=1, how='all', inplace=True)

        # 3. Memory Optimization (Downcast types to prevent Colab Crash)
        float_cols = X.select_dtypes(include=['float64']).columns
        X[float_cols] = X[float_cols].astype('float32')
        int_cols = X.select_dtypes(include=['int64']).columns
        X[int_cols] = X[int_cols].astype('int8')

        # 4. Handle NaNs smartly using forward and backward fill (Crucial for Boruta)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.ffill().bfill()  # Rolling windows and daily data gaps filled here

        # 5. Create Target
        y = (X['close'].shift(-1) > X['close']).astype(int)

        # Combine safely and drop only rows where target or features still have remaining NaNs
        data = X.assign(target=y).dropna()
        
        if data.empty or len(data) < 50:
            logger.error(f"Brain: Not enough data after dropna for Boruta. Rows left: {len(data)}")
            return self.selected_features

        y = data['target']
        # Drop non-feature columns that shouldn't feed into Boruta
        X_final = data.drop(columns=['target', 'close', 'open', 'high', 'low', 'volume'], errors='ignore')

        logger.info(f"Brain: Running Boruta on {X_final.shape[0]} rows and {X_final.shape[1]} features...")

        # Optimized RandomForest to save RAM inside Colab
        rf = RandomForestClassifier(
            n_jobs=-1,
            n_estimators=100,      # Reduced from 200 to 100 to save memory, plenty for Boruta
            max_depth=12,          # Depth restricted to prevent OOM/RAM crash
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42
        )
        
        feat_selector = BorutaPy(
            rf,
            n_estimators='auto',
            verbose=1,             # Set to 1 to see real-time filtering logs
            random_state=42,
            max_iter=30            # Iterations kept to 30 for high-speed computation
        )

        try:
            feat_selector.fit(X_final.values, y.values)
            confirmed = X_final.columns[feat_selector.support_].tolist()
            tentative = X_final.columns[feat_selector.support_weak_].tolist()

            if len(confirmed) < 5:
                self.selected_features = list(dict.fromkeys(confirmed + tentative))
                logger.warning(
                    f"Brain: Only {len(confirmed)} confirmed. Added tentative. Total: {len(self.selected_features)}"
                )
            else:
                self.selected_features = confirmed

            # Safety net: If Boruta still returns 0, pick top 15 features manually via correlation
            if len(self.selected_features) == 0:
                logger.warning("Brain: Boruta selected 0 features. Falling back to top correlated features.")
                correlations = X_final.corrwith(y).abs().sort_values(ascending=False)
                self.selected_features = correlations.head(15).index.tolist()

            self.save_state()

        except Exception as e:
            logger.error(f"❌ Brain: Boruta Error: {e}")

        return self.selected_features

    def auto_check_update(self, df):
        if self.should_run_update():
            logger.info("📅 Brain: Weekly update triggered")
            return self.run_boruta_flow(df)
        else:
            return self.selected_features

brain = FeatureBrain()
