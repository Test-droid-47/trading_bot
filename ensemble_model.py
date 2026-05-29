import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

class EnsembleModel:
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = RobustScaler()
        self._feat_cols: List[str] = []
        self._weights = self.cfg.get('ensemble_weights', {'xgb': 0.5, 'lgb': 0.5})
        self._trained = False

    def _prepare_tabular(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        numeric_df = df.select_dtypes(include=[np.number])
        cols = [c for c in numeric_df.columns if c not in ['timestamp']]
        
        X = numeric_df[cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.ffill(inplace=True)
        X.fillna(0.0, inplace=True)
        
        for col in X.columns:
            if X[col].std() < 1e-10:
                X[col] = X[col] + np.random.normal(0, 1e-8, len(X))
        
        y = (df['close'].shift(-1) > df['close']).astype(int)
        
        valid_idx = ~y.isna()
        X = X[valid_idx].iloc[:-1]
        y = y[valid_idx].iloc[:-1]
        
        return X, y, cols

    def train(self, df: pd.DataFrame) -> None:
        print("Training XGBoost + LightGBM ensemble...")
        
        if 'close' not in df.columns:
            print("'close' column missing")
            return
        
        X, y, cols = self._prepare_tabular(df)
        self._feat_cols = cols
        
        if len(X) < 200:
            print(f"Only {len(X)} samples, may not be enough")
            return
        
        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]
        
        X_tr_sc = self.scaler.fit_transform(X_tr)
        X_val_sc = self.scaler.transform(X_val)
        
        print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Features: {len(cols)}")
        
        if XGB_AVAILABLE:
            try:
                self.xgb_model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=20,
                    verbosity=0
                )
                self.xgb_model.fit(
                    X_tr_sc, y_tr,
                    eval_set=[(X_val_sc, y_val)],
                    verbose=False
                )
                acc = (self.xgb_model.predict(X_val_sc) == y_val).mean()
                print(f"XGB Val Accuracy: {acc:.4f}")
            except Exception as e:
                print(f"XGB training failed: {e}")
                self.xgb_model = None
        
        if LGB_AVAILABLE:
            try:
                self.lgb_model = lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
                self.lgb_model.fit(
                    X_tr_sc, y_tr,
                    eval_set=[(X_val_sc, y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False)]
                )
                acc = (self.lgb_model.predict(X_val_sc) == y_val).mean()
                print(f"LGB Val Accuracy: {acc:.4f}")
            except Exception as e:
                print(f"LGB training failed: {e}")
                self.lgb_model = None
        
        self._trained = True
        self._update_weights(X_val_sc, y_val)

    def _update_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        predictions = []
        weights = []
        
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict_proba(X_val)[:, 1]
            xgb_acc = (self.xgb_model.predict(X_val) == y_val).mean()
            predictions.append(xgb_pred)
            weights.append(max(0.1, xgb_acc))
        
        if self.lgb_model is not None:
            lgb_pred = self.lgb_model.predict_proba(X_val)[:, 1]
            lgb_acc = (self.lgb_model.predict(X_val) == y_val).mean()
            predictions.append(lgb_pred)
            weights.append(max(0.1, lgb_acc))
        
        if predictions:
            total = sum(weights)
            if total > 0:
                self._weights['xgb'] = weights[0] / total if len(weights) > 0 else 0.5
                self._weights['lgb'] = weights[1] / total if len(weights) > 1 else 0.5
        
        print(f"Updated weights: {self._weights}")

    def predict_proba_bullish(self, row: pd.Series) -> float:
        if not self._trained or len(self._feat_cols) == 0:
            return 0.5
        
        available = [c for c in self._feat_cols if c in row.index]
        if len(available) == 0:
            return 0.5
        
        x_full = np.zeros((1, len(self._feat_cols)), dtype=np.float64)
        for i, c in enumerate(self._feat_cols):
            if c in row.index and pd.notna(row[c]):
                x_full[0, i] = float(row[c])
        
        try:
            x_sc = self.scaler.transform(x_full)
        except Exception:
            return 0.5
        
        probs = []
        weights_list = []
        
        if self.xgb_model is not None:
            try:
                probs.append(float(self.xgb_model.predict_proba(x_sc)[0, 1]))
                weights_list.append(self._weights.get('xgb', 0.5))
            except Exception:
                pass
        
        if self.lgb_model is not None:
            try:
                probs.append(float(self.lgb_model.predict_proba(x_sc)[0, 1]))
                weights_list.append(self._weights.get('lgb', 0.5))
            except Exception:
                pass
        
        if not probs:
            return 0.5
        
        weights_sum = sum(weights_list)
        if weights_sum > 0:
            weights_list = [w / weights_sum for w in weights_list]
            return float(np.average(probs, weights=weights_list))
        
        return float(np.mean(probs))

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained or len(self._feat_cols) == 0:
            return np.full(len(df), 0.5)
        
        X = df[self._feat_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.ffill(inplace=True)
        X.fillna(0.0, inplace=True)
        
        try:
            X_sc = self.scaler.transform(X)
        except Exception:
            return np.full(len(df), 0.5)
        
        predictions = []
        weights = []
        
        if self.xgb_model is not None:
            try:
                xgb_pred = self.xgb_model.predict_proba(X_sc)[:, 1]
                predictions.append(xgb_pred)
                weights.append(self._weights.get('xgb', 0.5))
            except Exception:
                pass
        
        if self.lgb_model is not None:
            try:
                lgb_pred = self.lgb_model.predict_proba(X_sc)[:, 1]
                predictions.append(lgb_pred)
                weights.append(self._weights.get('lgb', 0.5))
            except Exception:
                pass
        
        if not predictions:
            return np.full(len(df), 0.5)
        
        weights_sum = sum(weights)
        if weights_sum > 0:
            weights = [w / weights_sum for w in weights]
        
        ensemble = np.zeros(len(df))
        for pred, weight in zip(predictions, weights):
            ensemble += pred * weight
        
        return ensemble

    def get_feature_importance(self) -> pd.DataFrame:
        importance_data = []
        
        if self.xgb_model is not None:
            importance = self.xgb_model.feature_importances_
            for i, col in enumerate(self._feat_cols[:len(importance)]):
                importance_data.append({'feature': col, 'xgb_importance': float(importance[i]), 'lgb_importance': 0.0})
        
        if self.lgb_model is not None:
            importance = self.lgb_model.feature_importances_
            for i, col in enumerate(self._feat_cols[:len(importance)]):
                existing = next((d for d in importance_data if d['feature'] == col), None)
                if existing:
                    existing['lgb_importance'] = float(importance[i])
                else:
                    importance_data.append({'feature': col, 'xgb_importance': 0.0, 'lgb_importance': float(importance[i])})
        
        df_importance = pd.DataFrame(importance_data)
        if not df_importance.empty:
            df_importance['total_importance'] = df_importance['xgb_importance'] + df_importance['lgb_importance']
            df_importance = df_importance.sort_values('total_importance', ascending=False)
        
        return df_importance

    def save(self, path: str = "ensemble_model_v4.pkl"):
        try:
            joblib.dump({
                'xgb_model': self.xgb_model,
                'lgb_model': self.lgb_model,
                'scaler': self.scaler,
                'feat_cols': self._feat_cols,
                'weights': self._weights,
                'trained': self._trained
            }, path)
            print(f"Ensemble saved to {path}")
        except Exception as e:
            print(f"Save failed: {e}")

    def load(self, path: str = "ensemble_model_v4.pkl"):
        import os
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return False
        
        try:
            data = joblib.load(path)
            self.xgb_model = data.get('xgb_model')
            self.lgb_model = data.get('lgb_model')
            self.scaler = data.get('scaler', RobustScaler())
            self._feat_cols = data.get('feat_cols', [])
            self._weights = data.get('weights', {'xgb': 0.5, 'lgb': 0.5})
            self._trained = data.get('trained', True)
            print(f"Ensemble loaded from {path}")
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained and (self.xgb_model is not None or self.lgb_model is not None)