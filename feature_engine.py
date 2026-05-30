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
    def add_trend_indicators(df: pd.DataFrame, features: dict) -> None:
        if not TA_AVAILABLE:
            return
        
        # Ensure index order for safe sequential calculations like VWAP
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        for length in [9, 20, 50, 100, 200]:
            features[f'ema_{length}'] = ta.ema(df['close'], length=length).astype(np.float32)
            features[f'sma_{length}'] = ta.sma(df['close'], length=length).astype(np.float32)
        
                ich_df, _ = ta.ichimoku(df['high'], df['low'], df['close'])
        if ich_df is not None and 'ITS_9_26_52' in ich_df.columns:
            features['ich_conversion'] = ich_df['ITS_9_26_52'].astype(np.float32).values
            features['ich_base'] = ich_df['IKS_9_26_52'].astype(np.float32).values
            features['ich_span_a'] = ich_df['ISA_9'].astype(np.float32).values
            features['ich_span_b'] = ich_df['ISB_26'].astype(np.float32).values
        else:
            # Agar indicator fail ho jaye ya column na mile, toh pure array assign karein
            empty_array = np.full(len(df), np.nan, dtype=np.float32)
            features['ich_conversion'] = empty_array
            features['ich_base'] = empty_array
            features['ich_span_a'] = empty_array
            features['ich_span_b'] = empty_array

        
        adx_df = ta.adx(df['high'], df['low'], df['close'])
        features['adx'] = adx_df['ADX_14'].astype(np.float32)
        features['adx_p'] = adx_df['DMP_14'].astype(np.float32)
        features['adx_n'] = adx_df['DMN_14'].astype(np.float32)
        
        vx = ta.vortex(df['high'], df['low'], df['close'])
        features['vortex_p'] = vx['VTXP_14'].astype(np.float32)
        features['vortex_n'] = vx['VTXM_14'].astype(np.float32)
        
        ar = ta.aroon(df['high'], df['low'])
        features['aroon_up'] = ar['AROONU_14'].astype(np.float32)
        features['aroon_down'] = ar['AROOND_14'].astype(np.float32)
        
        for length in [20, 50, 200]:
            features[f'close_vs_ema_{length}'] = ((df['close'] - features[f'ema_{length}']) / (features[f'ema_{length}'] + 1e-10)).astype(np.float32)
        
        features['golden_cross'] = (features['ema_50'] > features['ema_200']).astype(np.int8)
        features['death_cross'] = (features['ema_50'] < features['ema_200']).astype(np.int8)
        
        for length in [20, 50]:
            features[f'ema_{length}_slope'] = (pd.Series(features[f'ema_{length}'], index=df.index).diff(3) / (pd.Series(features[f'ema_{length}'], index=df.index).shift(3) + 1e-10)).astype(np.float32)

    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame, features: dict) -> None:
        if not TA_AVAILABLE:
            return
        
        for length in [6, 9, 14, 21, 28]:
            features[f'rsi_{length}'] = ta.rsi(df['close'], length=length).astype(np.float32)
        features['rsi'] = features['rsi_14']
        
        macd_df = ta.macd(df['close'])
        features['macd'] = macd_df['MACD_12_26_9'].astype(np.float32)
        features['macd_signal'] = macd_df['MACDs_12_26_9'].astype(np.float32)
        features['macd_hist'] = macd_df['MACDh_12_26_9'].astype(np.float32)
        features['macd_cross'] = np.sign(features['macd_hist']).diff().fillna(0).astype(np.int8)
        
        stoch_df = ta.stoch(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch_df['STOCHk_14_3_3'].astype(np.float32)
        features['stoch_d'] = stoch_df['STOCHd_14_3_3'].astype(np.float32)
        
        stochrsi_df = ta.stochrsi(df['close'])
        if stochrsi_df is not None:
            features['stochrsi_k'] = stochrsi_df.iloc[:, 0].astype(np.float32)
            features['stochrsi_d'] = stochrsi_df.iloc[:, 1].astype(np.float32)
        
        features['cci'] = ta.cci(df['high'], df['low'], df['close']).astype(np.float32)
        features['williams_r'] = ta.willr(df['high'], df['low'], df['close']).astype(np.float32)
        features['roc_5'] = ta.roc(df['close'], length=5).astype(np.float32)
        features['roc_10'] = ta.roc(df['close'], length=10).astype(np.float32)
        features['roc_20'] = ta.roc(df['close'], length=20).astype(np.float32)
        features['momentum_10'] = ta.mom(df['close'], length=10).astype(np.float32)
        features['momentum_20'] = ta.mom(df['close'], length=20).astype(np.float32)
        features['trix'] = ta.trix(df['close'], length=15)['TRIX_15_9'].astype(np.float32)
        features['awesome_osc'] = ta.ao(df['high'], df['low']).astype(np.float32)
        features['dpo'] = ta.dpo(df['close']).astype(np.float32)
        features['ppo'] = ta.ppo(df['close'])['PPO_12_26_9'].astype(np.float32)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(np.int8)
        features['rsi_oversold'] = (features['rsi'] < 30).astype(np.int8)
        features['rsi_midline'] = (features['rsi'] > 50).astype(np.int8)

    @staticmethod
    def add_volatility_volume_indicators(df: pd.DataFrame, features: dict) -> None:
        if not TA_AVAILABLE:
            return
        
        features['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).astype(np.float32)
        features['atr_7'] = ta.atr(df['high'], df['low'], df['close'], length=7).astype(np.float32)
        features['atr_21'] = ta.atr(df['high'], df['low'], df['close'], length=21).astype(np.float32)
        features['natr'] = ta.natr(df['high'], df['low'], df['close'], length=14).astype(np.float32)
        features['true_range'] = ta.true_range(df['high'], df['low'], df['close']).astype(np.float32)
        features['atr_vs_mean'] = (features['atr'] / (pd.Series(features['atr']).rolling(50).mean().values + 1e-10)).astype(np.float32)
        features['high_vol_regime'] = (features['atr_vs_mean'] > 1.5).astype(np.int8)
        features['low_vol_regime'] = (features['atr_vs_mean'] < 0.7).astype(np.int8)
        
        for std in [1.5, 2.0, 2.5]:
            try:
                bb = ta.bbands(df['close'], length=20, std=std)
                s = str(std).replace('.', '_')
                features[f'bb_upper_{s}'] = bb.iloc[:, 0].astype(np.float32)
                features[f'bb_lower_{s}'] = bb.iloc[:, 1].astype(np.float32)
                features[f'bb_width_{s}'] = ((bb.iloc[:, 0] - bb.iloc[:, 1]) / (bb.iloc[:, 2] + 1e-10)).astype(np.float32)
            except Exception as e:
                features[f'bb_upper_{s}'] = np.zeros(len(df), dtype=np.float32)
                features[f'bb_lower_{s}'] = np.zeros(len(df), dtype=np.float32)
                features[f'bb_width_{s}'] = np.zeros(len(df), dtype=np.float32)
        
        features['bb_upper'] = features['bb_upper_2_0']
        features['bb_lower'] = features['bb_lower_2_0']
        features['bb_width'] = features['bb_width_2_0']
        
        try:
            bb_simple = ta.bbands(df['close'])
            features['bb_middle'] = bb_simple.iloc[:, 2].astype(np.float32)
        except:
            features['bb_middle'] = df['close'].rolling(20).mean().astype(np.float32).values
        
        features['bb_pct'] = ((df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)).astype(np.float32)
        features['bb_squeeze'] = (features['bb_width'] < pd.Series(features['bb_width']).rolling(20).quantile(0.2).values).astype(np.int8)
        
        try:
            kc = ta.kc(df['high'], df['low'], df['close'])
            features['kc_upper'] = kc.iloc[:, 0].astype(np.float32)
            features['kc_middle'] = kc.iloc[:, 1].astype(np.float32)
            features['kc_lower'] = kc.iloc[:, 2].astype(np.float32)
        except Exception as e:
            features['kc_upper'] = np.zeros(len(df), dtype=np.float32)
            features['kc_middle'] = np.zeros(len(df), dtype=np.float32)
            features['kc_lower'] = np.zeros(len(df), dtype=np.float32)
        
        try:
            dc = ta.donchian(df['high'], df['low'])
            features['dc_upper'] = dc.iloc[:, 0].astype(np.float32)
            features['dc_lower'] = dc.iloc[:, 1].astype(np.float32)
            features['dc_middle'] = ((dc.iloc[:, 0] + dc.iloc[:, 1]) / 2.0).astype(np.float32)
            features['dc_width'] = ((dc.iloc[:, 0] - dc.iloc[:, 1]) / (features['dc_middle'] + 1e-10)).astype(np.float32)
        except Exception as e:
            features['dc_upper'] = np.zeros(len(df), dtype=np.float32)
            features['dc_lower'] = np.zeros(len(df), dtype=np.float32)
            features['dc_middle'] = np.zeros(len(df), dtype=np.float32)
            features['dc_width'] = np.zeros(len(df), dtype=np.float32)
        
        features['obv'] = ta.obv(df['close'], df['volume']).astype(np.float32)
        features['obv_ema'] = ta.ema(pd.Series(features['obv']), length=20).astype(np.float32).values
        features['obv_divergence'] = ((features['obv'] - features['obv_ema']) / (np.abs(features['obv_ema']) + 1e-10)).astype(np.float32)
        features['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume']).astype(np.float32)
        features['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume']).astype(np.float32)
        features['mfi_overbought'] = (features['mfi'] > 80).astype(np.int8)
        features['mfi_oversold'] = (features['mfi'] < 20).astype(np.int8)
        
        features['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume']).astype(np.float32)
        features['close_vs_vwap'] = ((df['close'] - features['vwap']) / (features['vwap'] + 1e-10)).astype(np.float32)
        features['ad_line'] = ta.ad(df['high'], df['low'], df['close'], df['volume']).astype(np.float32)
        features['adosc'] = ta.adosc(df['high'], df['low'], df['close'], df['volume']).astype(np.float32)
        features['vol_ema_20'] = ta.ema(df['volume'], length=20).astype(np.float32)
        features['vol_ratio'] = (df['volume'] / (features['vol_ema_20'] + 1e-10)).astype(np.float32)
        features['vol_spike'] = (features['vol_ratio'] > 2.0).astype(np.int8)
        features['vol_dry'] = (features['vol_ratio'] < 0.5).astype(np.int8)
        features['candle_delta'] = np.where(df['close'] >= df['open'], df['volume'], -df['volume']).astype(np.float32)
        features['cvd'] = features['candle_delta'].cumsum().astype(np.float32)
        features['cvd_ema'] = ta.ema(pd.Series(features['cvd']), length=20).astype(np.float32).values
        features['cvd_trend'] = (features['cvd'] > features['cvd_ema']).astype(np.int8)
        features['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)).astype(np.float32)
        features['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)).astype(np.float32)

    @staticmethod
    def add_pivot_fibonacci(df: pd.DataFrame, features: dict) -> None:
        ph = df['high'].shift(1).rolling(20).max()
        pl = df['low'].shift(1).rolling(20).min()
        rng = ph - pl
        
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fib_names = ['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100']
        
        for name, ratio in zip(fib_names, fib_levels):
            features[name] = (ph - rng * ratio).astype(np.float32)
            features[f'{name}_dist'] = ((df['close'] - features[name]) / (df['close'] + 1e-10)).astype(np.float32)
        
        prev_h, prev_l, prev_c = df['high'].shift(1), df['low'].shift(1), df['close'].shift(1)
        features['pvt_std'] = ((prev_h + prev_l + prev_c) / 3).astype(np.float32)
        features['pvt_r1'] = (2 * features['pvt_std'] - prev_l).astype(np.float32)
        features['pvt_s1'] = (2 * features['pvt_std'] - prev_h).astype(np.float32)
        features['pvt_r2'] = (features['pvt_std'] + (prev_h - prev_l)).astype(np.float32)
        features['pvt_s2'] = (features['pvt_std'] - (prev_h - prev_l)).astype(np.float32)
        features['cam_r3'] = (prev_c + (prev_h - prev_l) * 1.1 / 4).astype(np.float32)
        features['cam_s3'] = (prev_c - (prev_h - prev_l) * 1.1 / 4).astype(np.float32)
        features['dist_to_r1'] = ((features['pvt_r1'] - df['close']) / (df['close'] + 1e-10)).astype(np.float32)
        features['dist_to_s1'] = ((df['close'] - features['pvt_s1']) / (df['close'] + 1e-10)).astype(np.float32)

    @staticmethod
    def add_candlestick_patterns(df: pd.DataFrame, features: dict) -> None:
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        c = df['close'].values.astype(float)
        body = np.abs(c - o)
        rng = h - l + 1e-10
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l
        
        features['cdl_doji'] = (body <= rng * 0.05).astype(np.int8)
        features['cdl_dragonfly'] = ((body <= rng * 0.05) & (lower > 2*body)).astype(np.int8)
        features['cdl_gravestone'] = ((body <= rng * 0.05) & (upper > 2*body)).astype(np.int8)
        features['cdl_hammer'] = ((lower >= body*2) & (upper <= body*0.3) & (c > o)).astype(np.int8)
        features['cdl_hanging_man'] = ((lower >= body*2) & (upper <= body*0.3) & (c < o)).astype(np.int8)
        features['cdl_marubozu_bull'] = ((body >= rng*0.95) & (c > o)).astype(np.int8)
        features['cdl_marubozu_bear'] = ((body >= rng*0.95) & (c < o)).astype(np.int8)
        features['cdl_candle_bull'] = (c > o).astype(np.int8)
        features['cdl_candle_bear'] = (c < o).astype(np.int8)
        features['cdl_body_size'] = (body / rng).astype(np.float32)
        features['cdl_upper_wick'] = (upper / rng).astype(np.float32)
        features['cdl_lower_wick'] = (lower / rng).astype(np.float32)
        
        prev_c = np.roll(c, 1)
        prev_o = np.roll(o, 1)
        prev_c[:1] = c[:1]
        prev_o[:1] = o[:1]
        features['cdl_bull_engulf'] = ((c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)).astype(np.int8)
        features['cdl_bear_engulf'] = ((c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)).astype(np.int8)

    @staticmethod
    def add_time_features(df: pd.DataFrame, features: dict) -> None:
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
        else:
            ts = pd.to_datetime(df.index)
            
        hour = ts.dt.hour.astype(np.int8).values
        day_of_week = ts.dt.dayofweek.astype(np.int8).values
        
        features['hour'] = hour
        features['day_of_week'] = day_of_week
        features['month'] = ts.dt.month.astype(np.int8).values
        features['is_weekend'] = (day_of_week >= 5).astype(np.int8)
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7).astype(np.float32)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7).astype(np.float32)

    @staticmethod
    def add_statistical_features(df: pd.DataFrame, features: dict) -> None:
        for w in [10, 20, 50]:
            roll = df['close'].rolling(window=w)
            features[f'close_mean_{w}'] = roll.mean().astype(np.float32).values
            features[f'close_std_{w}'] = roll.std().astype(np.float32).values
            features[f'close_skew_{w}'] = roll.skew().astype(np.float32).values
            features[f'close_kurt_{w}'] = roll.kurt().astype(np.float32).values
            features[f'close_range_{w}'] = (roll.max() - roll.min()).astype(np.float32).values
            features[f'close_median_{w}'] = roll.median().astype(np.float32).values
            features[f'close_zscore_{w}'] = ((df['close'] - features[f'close_mean_{w}']) / (features[f'close_std_{w}'] + 1e-10)).astype(np.float32).values
        
        for w in [5, 10, 20]:
            log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
            features[f'realized_vol_{w}'] = (log_ret.rolling(w).std() * np.sqrt(w)).astype(np.float32).values
        
        features['autocorr_5'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False).astype(np.float32).values
        features['autocorr_10'] = df['close'].rolling(20).apply(lambda x: x.autocorr(lag=10), raw=False).astype(np.float32).values

    @staticmethod
    def add_lagged_returns(df: pd.DataFrame, features: dict) -> None:
        for lag in range(1, 13):
            features[f'close_lag_{lag}'] = df['close'].shift(lag).astype(np.float32).values
            features[f'volume_lag_{lag}'] = df['volume'].shift(lag).astype(np.float32).values
            if 'rsi' in features:
                features[f'rsi_lag_{lag}'] = pd.Series(features['rsi']).shift(lag).astype(np.float32).values
            if 'macd' in features:
                features[f'macd_lag_{lag}'] = pd.Series(features['macd']).shift(lag).astype(np.float32).values
        
        for period in [1, 3, 5, 10, 20, 50]:
            features[f'ret_{period}'] = df['close'].pct_change(period).astype(np.float32).values
            features[f'log_ret_{period}'] = np.log(df['close'] / (df['close'].shift(period) + 1e-10)).astype(np.float32).values
        
        features['ret_accel'] = pd.Series(features['ret_1']).diff().astype(np.float32).values

    @staticmethod
    def add_interaction_features(df: pd.DataFrame, features: dict) -> None:
        if 'rsi' in features and 'macd' in features:
            features['rsi_macd_x'] = (features['rsi'] * features['macd']).astype(np.float32)
        if 'rsi' in features:
            features['vol_rsi_x'] = (df['volume'].values * features['rsi']).astype(np.float32)
        if 'atr' in features and 'ema_20' in features:
            features['atr_ema_x'] = (features['atr'] * features['ema_20']).astype(np.float32)
        if 'bb_width' in features and 'rsi' in features:
            features['bb_rsi_x'] = (features['bb_width'] * features['rsi']).astype(np.float32)
        if 'adx' in features and 'rsi' in features:
            features['adx_rsi_x'] = (features['adx'] * features['rsi']).astype(np.float32)
        if 'obv' in features:
            features['obv_vol_x'] = (features['obv'] * df['volume'].values).astype(np.float32)
        if 'cmf' in features and 'mfi' in features:
            features['cmf_mfi_x'] = (features['cmf'] * features['mfi']).astype(np.float32)
        if 'adx' in features and 'atr' in features:
            features['adx_atr_x'] = (features['adx'] * features['atr']).astype(np.float32)
        if 'atr' in features:
            features['vol_atr_x'] = (df['volume'].values * features['atr']).astype(np.float32)
        if 'rsi' in features and 'vol_ratio' in features:
            features['rsi_vol_x'] = (features['rsi'] / (features['vol_ratio'] + 1e-10)).astype(np.float32)
        if 'momentum_10' in features and 'vol_ratio' in features:
            features['momentum_vol_x'] = (features['momentum_10'] * features['vol_ratio']).astype(np.float32)

    @classmethod
    def build_all(cls, df: pd.DataFrame, mtf_data: Optional[Dict] = None) -> pd.DataFrame:
        # Dictionary to accumulate features to prevent memory fragmentation
        features_dict = {}
        
        cls.add_trend_indicators(df, features_dict)
        cls.add_momentum_indicators(df, features_dict)
        cls.add_volatility_volume_indicators(df, features_dict)
        cls.add_pivot_fibonacci(df, features_dict)
        cls.add_candlestick_patterns(df, features_dict)
        cls.add_time_features(df, features_dict)
        cls.add_statistical_features(df, features_dict)
        cls.add_lagged_returns(df, features_dict)
        cls.add_interaction_features(df, features_dict)
        
        # Performance Booster: Convert accumulated dict into a single DataFrame and concatenate ONCE
        features_df = pd.DataFrame(features_dict, index=df.index)
        
        # Merge with base dataframe
        out_df = pd.concat([df, features_df], axis=1)
        
        # De-fragment internal memory structures explicitly
        out_df = out_df.copy()
        
        return out_df
