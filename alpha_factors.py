import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import hilbert

class AlphaFactorEngine:
    
    def __init__(self, cfg: Dict = None, window: int = 100):
        self.cfg = cfg or {}
        self.window = window

    @staticmethod
    def _hurst_rs(series: np.ndarray) -> float:
        n = len(series)
        if n < 20:
            return 0.5
        
        max_lag = max(3, n // 4)
        lags = list(range(2, max_lag))
        
        if len(lags) < 2:
            return 0.5
        
        rs_values = []
        lag_values = []
        
        for lag in lags:
            if lag >= n:
                continue
            
            chunks = [series[i:i+lag] for i in range(0, n - lag, lag)]
            chunks = [chunk for chunk in chunks if len(chunk) >= 2]
            
            if len(chunks) < 2:
                continue
            
            rs_list = []
            for chunk in chunks:
                mean_c = np.mean(chunk)
                devs = np.cumsum(chunk - mean_c)
                r = np.max(devs) - np.min(devs)
                s = np.std(chunk, ddof=1)
                if s > 1e-12:
                    rs_list.append(r / s)
            
            if rs_list:
                rs_values.append(np.mean(rs_list))
                lag_values.append(lag)
        
        if len(rs_values) < 2:
            return 0.5
        
        try:
            log_lags = np.log(np.array(lag_values, dtype=np.float64))
            log_rs = np.log(np.array(rs_values, dtype=np.float64))
            log_rs = np.maximum(log_rs, -20)
            h, _, _, _, _ = stats.linregress(log_lags, log_rs)
            return float(np.clip(h, 0.0, 1.0))
        except Exception:
            return 0.5

    def add_hurst_fractal(self, df: pd.DataFrame) -> pd.DataFrame:
        close_arr = df['close'].values.astype(float)
        n = len(close_arr)
        
        hurst = np.full(n, 0.5, dtype=np.float32)
        min_window = min(self.window, n)
        
        for i in range(min_window, n):
            try:
                window_data = close_arr[i-self.window:i]
                hurst[i] = self._hurst_rs(window_data)
            except Exception:
                hurst[i] = 0.5
        
        df['hurst_exp'] = hurst
        df['fractal_dim'] = (2.0 - hurst).astype(np.float32)
        df['market_memory'] = (hurst - 0.5).astype(np.float32)
        
        h_series = pd.Series(hurst)
        df['hurst_mean_20'] = h_series.rolling(20).mean().values.astype(np.float32)
        df['hurst_std_20'] = h_series.rolling(20).std().values.astype(np.float32)
        df['hurst_mean_50'] = h_series.rolling(50).mean().values.astype(np.float32)
        df['hurst_std_50'] = h_series.rolling(50).std().values.astype(np.float32)
        df['trending_mkt'] = (hurst > 0.6).astype(np.int8)
        df['reverting_mkt'] = (hurst < 0.4).astype(np.int8)
        df['random_walk'] = ((hurst >= 0.45) & (hurst <= 0.55)).astype(np.int8)
        
        return df

    def add_entropy_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        close_arr = df['close'].values
        n = len(close_arr)
        
        approx_entropy = np.full(n, np.nan, dtype=np.float32)
        sample_entropy = np.full(n, np.nan, dtype=np.float32)
        
        for i in range(window * 2, n):
            segment = close_arr[i-window:i]
            
            try:
                returns = np.diff(segment) / (segment[:-1] + 1e-10)
                hist, _ = np.histogram(returns, bins=20, density=True)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    approx_entropy[i] = -np.sum(hist * np.log(hist))
                else:
                    approx_entropy[i] = 0.0
            except Exception:
                approx_entropy[i] = 0.0
            
            try:
                m = 2
                r = 0.2 * np.std(segment)
                if r < 1e-10:
                    sample_entropy[i] = 0.0
                else:
                    n_m = 0
                    n_m1 = 0
                    for j in range(len(segment) - m):
                        for k in range(j + 1, len(segment) - m):
                            dj = np.max(np.abs(segment[j:j+m] - segment[k:k+m]))
                            if dj <= r:
                                n_m += 1
                            dj1 = np.max(np.abs(segment[j:j+m+1] - segment[k:k+m+1]))
                            if dj1 <= r:
                                n_m1 += 1
                    if n_m > 0 and n_m1 > 0:
                        sample_entropy[i] = -np.log(n_m1 / n_m)
                    else:
                        sample_entropy[i] = 0.0
            except Exception:
                sample_entropy[i] = 0.0
        
        df['approx_entropy'] = approx_entropy
        df['sample_entropy'] = sample_entropy
        df['entropy_ratio'] = (df['approx_entropy'] / (df['sample_entropy'] + 0.01)).astype(np.float32)
        
        return df

    def add_efficiency_ratio(self, df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        close_arr = df['close'].values
        
        for window in windows:
            er = np.zeros(len(df), dtype=np.float32)
            for i in range(window, len(df)):
                direction = abs(close_arr[i] - close_arr[i-window])
                volatility = np.sum(np.abs(np.diff(close_arr[i-window:i+1])))
                if volatility > 1e-10:
                    er[i] = direction / volatility
                else:
                    er[i] = 0.0
            df[f'efficiency_ratio_{window}'] = er
        
        df['er_smooth_20'] = df['efficiency_ratio_20'].rolling(5).mean()
        df['trend_efficiency'] = (df['efficiency_ratio_20'] > 0.5).astype(np.int8)
        df['choppy_efficiency'] = (df['efficiency_ratio_20'] < 0.3).astype(np.int8)
        
        return df

    def add_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'realized_vol_20' not in df.columns:
            log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
            df['realized_vol_20'] = log_ret.rolling(20).std() * np.sqrt(20)
        
        df['vol_percentile_20'] = df['realized_vol_20'].rolling(100).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0.5, raw=False
        ).astype(np.float32)
        
        df['vol_regime_low'] = (df['vol_percentile_20'] < 0.3).astype(np.int8)
        df['vol_regime_medium'] = ((df['vol_percentile_20'] >= 0.3) & (df['vol_percentile_20'] < 0.7)).astype(np.int8)
        df['vol_regime_high'] = (df['vol_percentile_20'] >= 0.7).astype(np.int8)
        df['vol_regime_score'] = df['vol_percentile_20'] * 2 - 1
        
        return df

    def add_skew_kurtosis_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        returns = df['close'].pct_change()
        
        df['ret_skew_20'] = returns.rolling(20).skew().astype(np.float32)
        df['ret_kurt_20'] = returns.rolling(20).kurt().astype(np.float32)
        df['ret_skew_50'] = returns.rolling(50).skew().astype(np.float32)
        df['ret_kurt_50'] = returns.rolling(50).kurt().astype(np.float32)
        
        df['positive_skew'] = (df['ret_skew_20'] > 0.5).astype(np.int8)
        df['negative_skew'] = (df['ret_skew_20'] < -0.5).astype(np.int8)
        df['fat_tails'] = (df['ret_kurt_20'] > 3).astype(np.int8)
        
        return df

    def add_ln_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        for period in periods:
            df[f'ln_ret_{period}'] = np.log(df['close'] / (df['close'].shift(period) + 1e-10)).astype(np.float32)
        
        df['ln_ret_accum_20'] = df['ln_ret_1'].rolling(20).sum().astype(np.float32)
        df['ln_ret_accum_50'] = df['ln_ret_1'].rolling(50).sum().astype(np.float32)
        
        return df

    def add_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volume' in df.columns:
            df['price_vol_corr_20'] = df['close'].rolling(20).corr(df['volume']).fillna(0).astype(np.float32)
        
        if 'rsi' in df.columns and 'macd' in df.columns:
            df['rsi_macd_corr_20'] = df['rsi'].rolling(20).corr(df['macd']).fillna(0).astype(np.float32)
        
        if 'atr' in df.columns:
            df['price_atr_corr_20'] = df['close'].rolling(20).corr(df['atr']).fillna(0).astype(np.float32)
        
        return df

    def add_zscore_features(self, df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        for window in windows:
            rolling_mean = df['close'].rolling(window).mean()
            rolling_std = df['close'].rolling(window).std()
            df[f'zscore_{window}'] = ((df['close'] - rolling_mean) / (rolling_std + 1e-10)).astype(np.float32)
        
        df['zscore_divergence'] = (df['zscore_10'] - df['zscore_50']).astype(np.float32)
        df['zscore_extreme'] = (np.abs(df['zscore_20']) > 2).astype(np.int8)
        
        return df

    def add_cycle_hilbert(self, df: pd.DataFrame, window: int = 40) -> pd.DataFrame:
        close = df['close'].values.astype(np.float64)
        n = len(close)

        amplitude = np.zeros(n, np.float32)
        phase = np.zeros(n, np.float32)
        phase_diff = np.zeros(n, np.float32)
        quad = np.zeros(n, np.float32)
        inphase = np.zeros(n, np.float32)
        dom_period = np.full(n, 20.0, np.float32)
        mesa_sine = np.zeros(n, np.float32)
        mesa_lead = np.zeros(n, np.float32)

        for i in range(window, n):
            seg = close[i - window: i]
            seg = seg - np.mean(seg)
            analy = hilbert(seg)
            amp = float(np.abs(analy[-1]))
            ph = float(np.angle(analy[-1]))
            amplitude[i] = amp
            phase[i] = ph
            phase_diff[i] = float(ph - np.angle(analy[-2])) if i > window else 0.0
            quad[i] = float(np.imag(analy[-1]))
            inphase[i] = float(np.real(analy[-1]))
            inst_freq = abs(ph - np.angle(analy[-2])) / (2 * np.pi + 1e-10)
            dom_period[i] = float(np.clip(1.0 / (inst_freq + 1e-10), 2, 200))
            mesa_sine[i] = float(np.sin(ph))
            mesa_lead[i] = float(np.sin(ph + np.pi / 4))

        df['hilbert_amplitude'] = amplitude
        df['hilbert_phase'] = phase
        df['hilbert_phase_diff'] = phase_diff
        df['hilbert_quad'] = quad
        df['hilbert_inphase'] = inphase
        df['dominant_cycle_period'] = dom_period
        df['mesa_sine'] = mesa_sine
        df['mesa_lead_sine'] = mesa_lead
        return df

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_hurst_fractal(df)
        df = self.add_entropy_features(df)
        df = self.add_efficiency_ratio(df)
        df = self.add_volatility_regimes(df)
        df = self.add_skew_kurtosis_features(df)
        df = self.add_ln_returns(df)
        df = self.add_correlation_features(df)
        df = self.add_zscore_features(df)
        df = self.add_cycle_hilbert(df)
        return df