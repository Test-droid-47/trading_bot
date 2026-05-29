import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.mixture import GaussianMixture

class MarketRegimeDetector:
    REGIME_NAMES = {0: 'Ranging', 1: 'StrongBull', 2: 'Bull', 3: 'Bear'}
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.n_components = self.cfg.get('gmm_components', 4)
        self.map_path = self.cfg.get('regime_map_path', 'regime_label_map.json')
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42,
            max_iter=500,
            n_init=5,
            tol=1e-4
        )
        self._fitted = False
        self._remap: Dict[int, int] = {}

    @staticmethod
    def _build_regime_features(df: pd.DataFrame) -> np.ndarray:
        feats = pd.DataFrame()
        
        if 'log_ret_20' in df.columns:
            feats['log_ret_20'] = df['log_ret_20']
        else:
            feats['log_ret_20'] = 0.0
        
        if 'log_ret_5' in df.columns:
            feats['log_ret_5'] = df['log_ret_5']
        else:
            log_ret_5 = np.log(df['close'] / (df['close'].shift(5) + 1e-10))
            feats['log_ret_5'] = log_ret_5
        
        if 'natr' in df.columns:
            feats['natr'] = df['natr']
        else:
            if 'atr' in df.columns and 'close' in df.columns:
                feats['natr'] = (df['atr'] / (df['close'] + 1e-10)) * 100
            else:
                feats['natr'] = 0.0
        
        if 'adx' in df.columns:
            feats['adx'] = df['adx']
        else:
            feats['adx'] = 25.0
        
        if 'hurst_exp' in df.columns:
            feats['hurst_exp'] = df['hurst_exp']
        else:
            feats['hurst_exp'] = 0.5
        
        if 'vol_ratio' in df.columns:
            feats['vol_ratio'] = df['vol_ratio']
        else:
            feats['vol_ratio'] = 1.0
        
        if 'rsi' in df.columns:
            feats['rsi'] = df['rsi']
        else:
            feats['rsi'] = 50.0
        
        feats.ffill(inplace=True)
        feats.fillna(0.0, inplace=True)
        
        return feats.values.astype(np.float32)

    def fit(self, df: pd.DataFrame) -> None:
        required_cols = ['close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing")
        
        X = self._build_regime_features(df)
        
        n_samples = len(X)
        if n_samples < self.n_components * 10:
            print(f"⚠️ Only {n_samples} samples, may not be enough")
        
        self.gmm.fit(X)
        self._fitted = True
        
        means = self.gmm.means_[:, 0]
        rank = np.argsort(means)
        
        if self.n_components == 4:
            self._remap = {
                int(rank[0]): 3,
                int(rank[1]): 0,
                int(rank[2]): 2,
                int(rank[3]): 1
            }
        elif self.n_components == 3:
            self._remap = {
                int(rank[0]): 3,
                int(rank[1]): 0,
                int(rank[2]): 1
            }
        else:
            for i, idx in enumerate(rank):
                self._remap[int(idx)] = i
        
        self.save_map()
        
        print(f"GMM fitted. Regime mapping: {self._remap}")
        for regime_id in range(self.n_components):
            print(f"  Component {regime_id}: mean_ret={means[regime_id]:.4f}")

    def save_map(self) -> None:
        data = {
            'remap': self._remap,
            'n_components': self.n_components,
            'fitted': self._fitted
        }
        try:
            with open(self.map_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Label map saved to {self.map_path}")
        except Exception as e:
            print(f"Failed to save map: {e}")

    def load_map(self) -> bool:
        if not os.path.exists(self.map_path):
            print(f"Map file not found: {self.map_path}")
            return False
        
        try:
            with open(self.map_path, 'r') as f:
                data = json.load(f)
            
            self._remap = {int(k): int(v) for k, v in data.get('remap', {}).items()}
            self.n_components = data.get('n_components', 4)
            self._fitted = data.get('fitted', False)
            
            print(f"Label map loaded: {self._remap}")
            return True
        except Exception as e:
            print(f"Failed to load map: {e}")
            return False

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("GMM not fitted. Call fit() or load_map() first.")
        
        X = self._build_regime_features(df)
        labels = self.gmm.predict(X)
        probs = self.gmm.predict_proba(X)
        
        mapped_labels = np.array([self._remap.get(int(l), 0) for l in labels], dtype=np.int8)
        
        df = df.copy()
        df['regime'] = mapped_labels
        df['regime_name'] = df['regime'].map(self.REGIME_NAMES).fillna('Unknown')
        
        for i in range(self.n_components):
            df[f'regime_p_{i}'] = probs[:, i].astype(np.float32)
        
        df['regime_confidence'] = np.max(probs, axis=1).astype(np.float32)
        df['regime_entropy'] = -np.sum(probs * np.log(probs + 1e-10), axis=1).astype(np.float32)
        
        regime_counts = df['regime'].value_counts().sort_index()
        for regime, count in regime_counts.items():
            print(f"  {self.REGIME_NAMES.get(regime, 'Unknown')}: {count} bars ({count/len(df)*100:.1f}%)")
        
        return df

    def predict_live(self, feature_row: np.ndarray) -> Dict[str, Any]:
        if not self._fitted:
            if self.load_map():
                if not self._fitted:
                    return {'regime': 0, 'regime_name': 'Unknown', 'probs': [], 'confidence': 0.0}
            else:
                return {'regime': 0, 'regime_name': 'Unknown', 'probs': [], 'confidence': 0.0}
        
        if feature_row.ndim == 1:
            feature_row = feature_row.reshape(1, -1)
        
        try:
            probs = self.gmm.predict_proba(feature_row)[0]
            raw = int(np.argmax(probs))
            confidence = float(np.max(probs))
            label = self._remap.get(raw, 0)
            
            return {
                'regime': label,
                'regime_name': self.REGIME_NAMES.get(label, 'Unknown'),
                'probs': probs.tolist(),
                'confidence': confidence,
                'raw_component': raw
            }
        except Exception as e:
            print(f"Live prediction failed: {e}")
            return {'regime': 0, 'regime_name': 'Unknown', 'probs': [], 'confidence': 0.0}