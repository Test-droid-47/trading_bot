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
    def __init__(self, feature_file="selected_features.json"):
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

        X = df.select_dtypes(include=[np.number]).copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        y = (df['close'].shift(-1) > df['close']).astype(int)

        data = X.assign(target=y).dropna()
        if data.empty or len(data) < 50:
            logger.error("Brain: Not enough data after dropna for Boruta")
            return self.selected_features

        y = data['target']
        X = data.drop(columns=['target'])

        rf = RandomForestClassifier(
            n_jobs=-1,
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42
        )
        feat_selector = BorutaPy(
            rf,
            n_estimators='auto',
            verbose=0,
            random_state=42,
            max_iter=50
        )

        try:
            feat_selector.fit(X.values, y.values)
            confirmed = X.columns[feat_selector.support_].tolist()
            tentative = X.columns[feat_selector.support_weak_].tolist()

            if len(confirmed) < 5:
                self.selected_features = list(dict.fromkeys(confirmed + tentative))
                logger.warning(
                    f"Brain: Only {len(confirmed)} confirmed. Added tentative. Total: {len(self.selected_features)}"
                )
            else:
                self.selected_features = confirmed

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