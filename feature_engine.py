import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    print("⚠️ pandas-ta not installed. Install with: pip install pandas-ta")
    TA_AVAILABLE = False

class FeatureEngine:

    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not TA_AVAILABLE:
            return df
        
        for length in [9, 20, 50, 100, 200]:
            df[f'ema_{length}'] = ta.ema(df['close'], length=length)
            df[f'sma_{length}'] = ta.sma(df['close'], length=length)
        
        ich_df, _ = ta.ichimoku(df['high'], df['low'], df['close'])
        if ich_df is not None:
            df['ich_conversion'] = ich_df.get('ITS_9_26_52', np.nan)
            df['ich_base'] = ich_df.get('IKS_9_26_52', np.nan)
            df['ich_span_a'] = ich_df.get('ISA_9', np.nan)
            df['ich_span_b'] = ich_df.get('ISB_26', np.nan)
        df['ich_lagging'] = df['close'].shift(26)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx_df['ADX_14']
        df['adx_p'] = adx_df['DMP_14']
        df['adx_n'] = adx_df['DMN_14']
        
        vx = ta.vortex(df['high'], df['low'], df['close'])
        df['vortex_p'] = vx['VTXP_14']
        df['vortex_n'] = vx['VTXM_14']
        
        ar = ta.aroon(df['high'], df['low'])
        df['aroon_up'] = ar['AROONU_14']
        df['aroon_down'] = ar['AROOND_14']
        
        for length in [20, 50, 200]:
            df[f'close_vs_ema_{length}'] = (df['close'] - df[f'ema_{length}']) / (df[f'ema_{length}'] + 1e-10)
        
        df['golden_cross'] = (df['ema_50'] > df['ema_200']).astype(np.int8)
        df['death_cross'] = (df['ema_50'] < df['ema_200']).astype(np.int8)
        
        for length in [20, 50]:
            df[f'ema_{length}_slope'] = df[f'ema_{length}'].diff(3) / (df[f'ema_{length}'].shift(3) + 1e-10)
        
        return df

    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not TA_AVAILABLE:
            return df
        
        for length in [6, 9, 14, 21, 28]:
            df[f'rsi_{length}'] = ta.rsi(df['close'], length=length)
        df['rsi'] = df['rsi_14']
        
        macd_df = ta.macd(df['close'])
        df['macd'] = macd_df['MACD_12_26_9']
        df['macd_signal'] = macd_df['MACDs_12_26_9']
        df['macd_hist'] = macd_df['MACDh_12_26_9']
        df['macd_cross'] = np.sign(df['macd_hist']).diff().fillna(0).astype(np.int8)
        
        stoch_df = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_df['STOCHk_14_3_3']
        df['stoch_d'] = stoch_df['STOCHd_14_3_3']
        
        stochrsi_df = ta.stochrsi(df['close'])
        if stochrsi_df is not None:
            df['stochrsi_k'] = stochrsi_df.iloc[:, 0]
            df['stochrsi_d'] = stochrsi_df.iloc[:, 1]
        
        df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
        df['roc_5'] = ta.roc(df['close'], length=5)
        df['roc_10'] = ta.roc(df['close'], length=10)
        df['roc_20'] = ta.roc(df['close'], length=20)
        df['momentum_10'] = ta.mom(df['close'], length=10)
        df['momentum_20'] = ta.mom(df['close'], length=20)
        df['trix'] = ta.trix(df['close'], length=15)['TRIX_15_9']
        df['awesome_osc'] = ta.ao(df['high'], df['low'])
        df['dpo'] = ta.dpo(df['close'])
        df['ppo'] = ta.ppo(df['close'])['PPO_12_26_9']
        df['rsi_overbought'] = (df['rsi'] > 70).astype(np.int8)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(np.int8)
        df['rsi_midline'] = (df['rsi'] > 50).astype(np.int8)
        
        return df

    @staticmethod
    def add_volatility_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not TA_AVAILABLE:
            return df
        
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_7'] = ta.atr(df['high'], df['low'], df['close'], length=7)
        df['atr_21'] = ta.atr(df['high'], df['low'], df['close'], length=21)
        df['natr'] = ta.natr(df['high'], df['low'], df['close'], length=14)
        df['true_range'] = ta.true_range(df['high'], df['low'], df['close'])
        df['atr_vs_mean'] = df['atr'] / (df['atr'].rolling(50).mean() + 1e-10)
        df['high_vol_regime'] = (df['atr_vs_mean'] > 1.5).astype(np.int8)
        df['low_vol_regime'] = (df['atr_vs_mean'] < 0.7).astype(np.int8)
        
        for std in [1.5, 2.0, 2.5]:
            try:
                bb = ta.bbands(df['close'], length=20, std=std)
                s = str(std).replace('.', '_')
                df[f'bb_upper_{s}'] = bb.iloc[:, 0]
                df[f'bb_lower_{s}'] = bb.iloc[:, 1]
                df[f'bb_width_{s}'] = (bb.iloc[:, 0] - bb.iloc[:, 1]) / (bb.iloc[:, 2] + 1e-10)
            except Exception as e:
                print(f"Warning: Bollinger Bands for std={std} failed: {e}")
                df[f'bb_upper_{s}'] = 0
                df[f'bb_lower_{s}'] = 0
                df[f'bb_width_{s}'] = 0
        
        df['bb_upper'] = df['bb_upper_2_0']
        df['bb_lower'] = df['bb_lower_2_0']
        df['bb_width'] = df['bb_width_2_0']
        
        try:
            bb_simple = ta.bbands(df['close'])
            df['bb_middle'] = bb_simple.iloc[:, 2]
        except:
            df['bb_middle'] = df['close'].rolling(20).mean()
        
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(np.int8)
        
        try:
            kc = ta.kc(df['high'], df['low'], df['close'])
            df['kc_upper'] = kc.iloc[:, 0]
            df['kc_middle'] = kc.iloc[:, 1]
            df['kc_lower'] = kc.iloc[:, 2]
        except Exception as e:
            print(f"Warning: Keltner Channels failed: {e}")
            df['kc_upper'] = 0
            df['kc_middle'] = 0
            df['kc_lower'] = 0
        
        try:
            dc = ta.donchian(df['high'], df['low'])
            df['dc_upper'] = dc.iloc[:, 0]
            df['dc_lower'] = dc.iloc[:, 1]
            df['dc_middle'] = (dc.iloc[:, 0] + dc.iloc[:, 1]) / 2.0
            df['dc_width'] = (dc.iloc[:, 0] - dc.iloc[:, 1]) / (df['dc_middle'] + 1e-10)
        except Exception as e:
            print(f"Warning: Donchian Channels failed: {e}")
            df['dc_upper'] = 0
            df['dc_lower'] = 0
            df['dc_middle'] = 0
            df['dc_width'] = 0
        
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['obv_ema'] = ta.ema(df['obv'], length=20)
        df['obv_divergence'] = (df['obv'] - df['obv_ema']) / (df['obv_ema'].abs() + 1e-10)
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        df['mfi_overbought'] = (df['mfi'] > 80).astype(np.int8)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(np.int8)
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['close_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
        df['ad_line'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        df['adosc'] = ta.adosc(df['high'], df['low'], df['close'], df['volume'])
        df['vol_ema_20'] = ta.ema(df['volume'], length=20)
        df['vol_ratio'] = df['volume'] / (df['vol_ema_20'] + 1e-10)
        df['vol_spike'] = (df['vol_ratio'] > 2.0).astype(np.int8)
        df['vol_dry'] = (df['vol_ratio'] < 0.5).astype(np.int8)
        df['candle_delta'] = np.where(df['close'] >= df['open'], df['volume'], -df['volume'])
        df['cvd'] = df['candle_delta'].cumsum()
        df['cvd_ema'] = ta.ema(pd.Series(df['cvd'].values), length=20)
        df['cvd_trend'] = (df['cvd'] > df['cvd_ema']).astype(np.int8)
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        
        return df

    @staticmethod
    def add_pivot_fibonacci(df: pd.DataFrame) -> pd.DataFrame:
        ph = df['high'].shift(1).rolling(20).max()
        pl = df['low'].shift(1).rolling(20).min()
        rng = ph - pl
        
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fib_names = ['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100']
        
        for name, ratio in zip(fib_names, fib_levels):
            df[name] = ph - rng * ratio
            df[f'{name}_dist'] = (df['close'] - df[name]) / (df['close'] + 1e-10)
        
        prev_h, prev_l, prev_c = df['high'].shift(1), df['low'].shift(1), df['close'].shift(1)
        df['pvt_std'] = (prev_h + prev_l + prev_c) / 3
        df['pvt_r1'] = 2 * df['pvt_std'] - prev_l
        df['pvt_s1'] = 2 * df['pvt_std'] - prev_h
        df['pvt_r2'] = df['pvt_std'] + (prev_h - prev_l)
        df['pvt_s2'] = df['pvt_std'] - (prev_h - prev_l)
        df['cam_r3'] = prev_c + (prev_h - prev_l) * 1.1 / 4
        df['cam_s3'] = prev_c - (prev_h - prev_l) * 1.1 / 4
        df['dist_to_r1'] = (df['pvt_r1'] - df['close']) / (df['close'] + 1e-10)
        df['dist_to_s1'] = (df['close'] - df['pvt_s1']) / (df['close'] + 1e-10)
        
        return df

    @staticmethod
    def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        c = df['close'].values.astype(float)
        body = np.abs(c - o)
        rng = h - l + 1e-10
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l
        
        df['cdl_doji'] = (body <= rng * 0.05).astype(np.int8)
        df['cdl_dragonfly'] = ((body <= rng * 0.05) & (lower > 2*body)).astype(np.int8)
        df['cdl_gravestone'] = ((body <= rng * 0.05) & (upper > 2*body)).astype(np.int8)
        df['cdl_hammer'] = ((lower >= body*2) & (upper <= body*0.3) & (c > o)).astype(np.int8)
        df['cdl_hanging_man'] = ((lower >= body*2) & (upper <= body*0.3) & (c < o)).astype(np.int8)
        df['cdl_marubozu_bull'] = ((body >= rng*0.95) & (c > o)).astype(np.int8)
        df['cdl_marubozu_bear'] = ((body >= rng*0.95) & (c < o)).astype(np.int8)
        df['cdl_candle_bull'] = (c > o).astype(np.int8)
        df['cdl_candle_bear'] = (c < o).astype(np.int8)
        df['cdl_body_size'] = (body / rng).astype(np.float32)
        df['cdl_upper_wick'] = (upper / rng).astype(np.float32)
        df['cdl_lower_wick'] = (lower / rng).astype(np.float32)
        
        prev_c = np.roll(c, 1)
        prev_o = np.roll(o, 1)
        prev_c[:1] = c[:1]
        prev_o[:1] = o[:1]
        df['cdl_bull_engulf'] = ((c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)).astype(np.int8)
        df['cdl_bear_engulf'] = ((c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)).astype(np.int8)
        
        return df

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        ts = df['timestamp']
        df['hour'] = ts.dt.hour.astype(np.int8)
        df['day_of_week'] = ts.dt.dayofweek.astype(np.int8)
        df['month'] = ts.dt.month.astype(np.int8)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
        
        return df

    @staticmethod
    def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        for w in [10, 20, 50]:
            roll = df['close'].rolling(window=w)
            df[f'close_mean_{w}'] = roll.mean()
            df[f'close_std_{w}'] = roll.std()
            df[f'close_skew_{w}'] = roll.skew()
            df[f'close_kurt_{w}'] = roll.kurt()
            df[f'close_range_{w}'] = roll.max() - roll.min()
            df[f'close_median_{w}'] = roll.median()
            df[f'close_zscore_{w}'] = ((df['close'] - df[f'close_mean_{w}']) / (df[f'close_std_{w}'] + 1e-10)).astype(np.float32)
        
        for w in [5, 10, 20]:
            log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
            df[f'realized_vol_{w}'] = log_ret.rolling(w).std() * np.sqrt(w)
        
        df['autocorr_5'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
        df['autocorr_10'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=10), raw=False)
        
        return df

    @staticmethod
    def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
        for lag in range(1, 13):
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            if 'rsi' in df.columns:
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            if 'macd' in df.columns:
                df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
        
        for period in [1, 3, 5, 10, 20, 50]:
            df[f'ret_{period}'] = df['close'].pct_change(period).astype(np.float32)
            df[f'log_ret_{period}'] = np.log(df['close'] / (df['close'].shift(period) + 1e-10)).astype(np.float32)
        
        df['ret_accel'] = df['ret_1'].diff()
        
        return df

    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        if 'rsi' in df.columns and 'macd' in df.columns:
            df['rsi_macd_x'] = (df['rsi'] * df['macd']).astype(np.float32)
        if 'volume' in df.columns and 'rsi' in df.columns:
            df['vol_rsi_x'] = (df['volume'] * df['rsi']).astype(np.float32)
        if 'atr' in df.columns and 'ema_20' in df.columns:
            df['atr_ema_x'] = (df['atr'] * df['ema_20']).astype(np.float32)
        if 'bb_width' in df.columns and 'rsi' in df.columns:
            df['bb_rsi_x'] = (df['bb_width'] * df['rsi']).astype(np.float32)
        if 'adx' in df.columns and 'rsi' in df.columns:
            df['adx_rsi_x'] = (df['adx'] * df['rsi']).astype(np.float32)
        if 'obv' in df.columns and 'volume' in df.columns:
            df['obv_vol_x'] = (df['obv'] * df['volume']).astype(np.float32)
        if 'cmf' in df.columns and 'mfi' in df.columns:
            df['cmf_mfi_x'] = (df['cmf'] * df['mfi']).astype(np.float32)
        if 'adx' in df.columns and 'atr' in df.columns:
            df['adx_atr_x'] = (df['adx'] * df['atr']).astype(np.float32)
        if 'volume' in df.columns and 'atr' in df.columns:
            df['vol_atr_x'] = (df['volume'] * df['atr']).astype(np.float32)
        if 'rsi' in df.columns and 'vol_ratio' in df.columns:
            df['rsi_vol_x'] = (df['rsi'] / (df['vol_ratio'] + 1e-10)).astype(np.float32)
        if 'momentum_10' in df.columns and 'vol_ratio' in df.columns:
            df['momentum_vol_x'] = (df['momentum_10'] * df['vol_ratio']).astype(np.float32)
        
        return df

    @classmethod
    def build_all(cls, df: pd.DataFrame, mtf_data: Optional[Dict] = None) -> pd.DataFrame:
        df = cls.add_trend_indicators(df)
        df = cls.add_momentum_indicators(df)
        df = cls.add_volatility_volume_indicators(df)
        df = cls.add_pivot_fibonacci(df)
        df = cls.add_candlestick_patterns(df)
        df = cls.add_time_features(df)
        df = cls.add_statistical_features(df)
        df = cls.add_lagged_returns(df)
        df = cls.add_interaction_features(df)
        if mtf_data:
            df = cls.add_mtf_confluence(df, mtf_data)
        return df