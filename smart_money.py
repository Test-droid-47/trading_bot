import numpy as np
import pandas as pd
from typing import Dict, Tuple

class SmartMoneyEngine:
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.ob_lookback = self.cfg.get('ob_lookback', 20)
        self.fvg_min_gap = self.cfg.get('fvg_min_gap_pct', 0.001)

    def detect_bos_choch(self, df: pd.DataFrame) -> pd.DataFrame:
        highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        bos_bull = np.zeros(n, np.int8)
        bos_bear = np.zeros(n, np.int8)
        choch_bull = np.zeros(n, np.int8)
        choch_bear = np.zeros(n, np.int8)
        window = self.ob_lookback
        last_dir = 0
        
        for i in range(window, n):
            sh = np.max(highs[i-window:i])
            sl = np.min(lows[i-window:i])
            
            if closes[i] > sh:
                bos_bull[i] = 1
                if last_dir == -1:
                    choch_bull[i] = 1
                last_dir = 1
            elif closes[i] < sl:
                bos_bear[i] = 1
                if last_dir == 1:
                    choch_bear[i] = 1
                last_dir = -1
        
        df['smc_bos_bull'] = bos_bull
        df['smc_bos_bear'] = bos_bear
        df['smc_choch_bull'] = choch_bull
        df['smc_choch_bear'] = choch_bear
        return df

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        opens, closes, volumes = df['open'].values, df['close'].values, df['volume'].values
        n = len(df)
        ob_bull_mid = np.zeros(n, np.float32)
        ob_bear_mid = np.zeros(n, np.float32)
        ob_bull_str = np.zeros(n, np.float32)
        ob_bear_str = np.zeros(n, np.float32)
        
        if 'smc_bos_bull' in df.columns:
            bos_bull = df['smc_bos_bull'].values
            bos_bear = df['smc_bos_bear'].values
        else:
            bos_bull = np.zeros(n)
            bos_bear = np.zeros(n)
        
        for i in range(self.ob_lookback, n):
            if bos_bull[i]:
                for j in range(i-1, max(i-self.ob_lookback, 0), -1):
                    if closes[j] < opens[j]:
                        ob_bull_mid[i] = (opens[j] + closes[j]) / 2
                        ob_bull_str[i] = volumes[j]
                        break
            
            if bos_bear[i]:
                for j in range(i-1, max(i-self.ob_lookback, 0), -1):
                    if closes[j] > opens[j]:
                        ob_bear_mid[i] = (opens[j] + closes[j]) / 2
                        ob_bear_str[i] = volumes[j]
                        break
        
        df['smc_ob_bull_mid'] = ob_bull_mid
        df['smc_ob_bear_mid'] = ob_bear_mid
        df['smc_ob_bull_strength'] = ob_bull_str
        df['smc_ob_bear_strength'] = ob_bear_str
        return df

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        fvg_bull = np.zeros(n, np.int8)
        fvg_bear = np.zeros(n, np.int8)
        fvg_bull_sz = np.zeros(n, np.float32)
        fvg_bear_sz = np.zeros(n, np.float32)
        
        for i in range(2, n):
            min_gap = closes[i] * self.fvg_min_gap
            gap = lows[i] - highs[i-2]
            if gap > min_gap:
                fvg_bull[i] = 1
                fvg_bull_sz[i] = gap
            
            gap = lows[i-2] - highs[i]
            if gap > min_gap:
                fvg_bear[i] = 1
                fvg_bear_sz[i] = gap
        
        df['smc_fvg_bull'] = fvg_bull
        df['smc_fvg_bear'] = fvg_bear
        df['smc_fvg_bull_size'] = fvg_bull_sz
        df['smc_fvg_bear_size'] = fvg_bear_sz
        return df

    def detect_premium_discount(self, df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        roll_h = df['high'].rolling(period).max()
        roll_l = df['low'].rolling(period).min()
        eq = (roll_h + roll_l) / 2.0
        
        df['smc_premium'] = (df['close'] > eq).astype(np.int8)
        df['smc_discount'] = (df['close'] < eq).astype(np.int8)
        df['smc_eq_level'] = eq.astype(np.float32)
        df['smc_eq_dist'] = ((df['close'] - eq) / (eq + 1e-10)).astype(np.float32)
        return df

    def detect_liquidity_sweeps(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)
        
        liq_sweep_bull = np.zeros(n, np.int8)
        liq_sweep_bear = np.zeros(n, np.int8)
        
        for i in range(lookback, n):
            recent_high = np.max(highs[i-lookback:i])
            recent_low = np.min(lows[i-lookback:i])
            
            if closes[i] > recent_high:
                liq_sweep_bull[i] = 1
            elif closes[i] < recent_low:
                liq_sweep_bear[i] = 1
        
        df['smc_liq_sweep_bull'] = liq_sweep_bull
        df['smc_liq_sweep_bear'] = liq_sweep_bear
        return df

    def detect_breaker_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'smc_choch_bull' not in df.columns or 'smc_choch_bear' not in df.columns:
            df = self.detect_bos_choch(df)
        
        n = len(df)
        bb_bull = np.zeros(n, np.float32)
        bb_bear = np.zeros(n, np.float32)
        
        for i in range(1, n):
            if df['smc_choch_bull'].iloc[i] == 1:
                for j in range(max(0, i-10), i):
                    if df['smc_bos_bear'].iloc[j] == 1:
                        bb_bull[i] = df['low'].iloc[j]
                        break
            
            if df['smc_choch_bear'].iloc[i] == 1:
                for j in range(max(0, i-10), i):
                    if df['smc_bos_bull'].iloc[j] == 1:
                        bb_bear[i] = df['high'].iloc[j]
                        break
        
        df['smc_breaker_bull'] = bb_bull
        df['smc_breaker_bear'] = bb_bear
        return df

    def calculate_smc_score(self, df: pd.DataFrame) -> pd.DataFrame:
        score = np.zeros(len(df), np.float32)
        
        if 'smc_bos_bull' in df.columns:
            score += df['smc_bos_bull'].values * 10
            score -= df['smc_bos_bear'].values * 10
        
        if 'smc_choch_bull' in df.columns:
            score += df['smc_choch_bull'].values * 15
            score -= df['smc_choch_bear'].values * 15
        
        if 'smc_fvg_bull' in df.columns:
            score += df['smc_fvg_bull'].values * 8
            score -= df['smc_fvg_bear'].values * 8
        
        if 'smc_premium' in df.columns:
            score -= df['smc_premium'].values * 5
            score += df['smc_discount'].values * 5
        
        if 'smc_liq_sweep_bull' in df.columns:
            score += df['smc_liq_sweep_bull'].values * 12
            score -= df['smc_liq_sweep_bear'].values * 12
        
        df['smc_score'] = score.astype(np.float32)
        df['smc_signal_bull'] = (df['smc_score'] > 20).astype(np.int8)
        df['smc_signal_bear'] = (df['smc_score'] < -20).astype(np.int8)
        
        return df

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.detect_bos_choch(df)
        df = self.detect_order_blocks(df)
        df = self.detect_fvg(df)
        df = self.detect_premium_discount(df)
        df = self.detect_liquidity_sweeps(df)
        df = self.detect_breaker_blocks(df)
        df = self.calculate_smc_score(df)
        return df