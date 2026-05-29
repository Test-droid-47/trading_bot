import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class TradingEnvironment:
    
    def __init__(self, df: pd.DataFrame, scaled_features: np.ndarray, cfg: Dict, close_idx: int):
        self.df = df.reset_index(drop=True)
        self.features = scaled_features
        self.cfg = cfg
        self.close_idx = close_idx
        self.window = cfg.get('window', 120)
        self.fee_rate = cfg.get('fee_rate', 0.001)
        self.slippage = cfg.get('slippage', 0.0005)
        self.initial_capital = cfg.get('initial_capital', 10000.0)
        self.drawdown_penalty = cfg.get('drawdown_penalty', 2.0)
        self.max_risk_per_trade = cfg.get('max_risk_per_trade', 0.02)
        self.n_bars = len(df)
        
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_idx = self.window
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.peak_capital = self.initial_capital
        self.done = False
        self.trades = []
        self.returns_history = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trailing_activated = False
        self.trailing_peak = 0.0
        self.dynamic_sl = 0.0
        self.dynamic_tp = 0.0
        self.current_sl_pct = 0.02
        self.current_tp_pct = 0.04
        
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        start_idx = self.current_idx - self.window
        end_idx = self.current_idx
        
        if start_idx < 0 or end_idx > len(self.features):
            return np.zeros((self.window, self.features.shape[1]), dtype=np.float32)
        
        return self.features[start_idx:end_idx].astype(np.float32)

    def _current_close(self) -> float:
        if self.current_idx >= len(self.df):
            return float(self.df['close'].iloc[-1])
        return float(self.df['close'].iloc[self.current_idx])

    def _current_high(self) -> float:
        if self.current_idx >= len(self.df):
            return float(self.df['high'].iloc[-1])
        return float(self.df['high'].iloc[self.current_idx])

    def _current_low(self) -> float:
        if self.current_idx >= len(self.df):
            return float(self.df['low'].iloc[-1])
        return float(self.df['low'].iloc[self.current_idx])

    def _get_atr(self) -> float:
        if 'atr' in self.df.columns and self.current_idx < len(self.df):
            return float(self.df['atr'].iloc[self.current_idx])
        return float(self._current_close() * 0.01)

    def _get_atr_mean(self) -> float:
        if 'atr' in self.df.columns and self.current_idx >= 50:
            return float(self.df['atr'].iloc[max(0, self.current_idx-50):self.current_idx].mean())
        return self._get_atr()

    def _get_adx(self) -> float:
        if 'adx' in self.df.columns and self.current_idx < len(self.df):
            return float(self.df['adx'].iloc[self.current_idx])
        return 25.0

    def _get_hurst(self) -> float:
        if 'hurst_exp' in self.df.columns and self.current_idx < len(self.df):
            return float(self.df['hurst_exp'].iloc[self.current_idx])
        return 0.5

    def _get_regime(self) -> int:
        if 'regime' in self.df.columns and self.current_idx < len(self.df):
            return int(self.df['regime'].iloc[self.current_idx])
        return 0

    def _calculate_dynamic_sl_tp(self, entry_price: float) -> Tuple[float, float]:
        atr = self._get_atr()
        atr_mean = self._get_atr_mean()
        adx = self._get_adx()
        hurst = self._get_hurst()
        regime = self._get_regime()
        
        vol_ratio = atr / (atr_mean + 1e-10)
        
        if vol_ratio > 1.5:
            sl_pct = 0.035
            tp_pct = 0.035
        elif vol_ratio < 0.7:
            sl_pct = 0.015
            tp_pct = 0.045
        else:
            sl_pct = 0.02
            tp_pct = 0.04
        
        if adx > 35:
            tp_pct *= 1.3
        elif adx > 25:
            tp_pct *= 1.15
        elif adx < 20:
            sl_pct *= 0.85
            tp_pct *= 0.85
        
        if hurst > 0.6:
            tp_pct *= 1.2
        elif hurst < 0.4:
            sl_pct *= 1.15
            tp_pct *= 0.9
        
        if regime == 1 or regime == 2:
            tp_pct *= 1.15
        elif regime == 3:
            sl_pct *= 0.9
            tp_pct *= 0.85
        
        if self.consecutive_losses >= 2:
            sl_pct *= 0.7
            tp_pct *= 0.7
        
        sl_pct = np.clip(sl_pct, 0.01, 0.05)
        tp_pct = np.clip(tp_pct, 0.02, 0.08)
        
        self.current_sl_pct = sl_pct
        self.current_tp_pct = tp_pct
        
        return sl_pct, tp_pct

    def _update_trailing(self, current_price: float, current_high: float, entry_price: float, pnl_pct: float, atr: float, atr_mean: float, adx: float) -> Tuple[float, bool]:
        vol_ratio = atr / (atr_mean + 1e-10)
        
        if not self.trailing_activated:
            if pnl_pct >= 0.02:
                self.trailing_activated = True
                self.trailing_peak = current_high
                return self.dynamic_sl, True
            
            if pnl_pct >= 0.01:
                new_sl = max(self.dynamic_sl, entry_price)
                return new_sl, False
        
        if current_high > self.trailing_peak:
            self.trailing_peak = current_high
        
        if vol_ratio > 1.3:
            trail_pct = 0.015
        elif vol_ratio < 0.7:
            trail_pct = 0.005
        else:
            trail_pct = 0.01
        
        if adx > 35:
            trail_pct *= 1.3
        elif adx < 20:
            trail_pct *= 0.7
        
        new_sl = self.trailing_peak * (1 - trail_pct)
        
        if new_sl > self.dynamic_sl:
            return new_sl, self.trailing_activated
        
        return self.dynamic_sl, self.trailing_activated

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        price = self._current_close()
        high = self._current_high()
        low = self._current_low()
        
        reward = 0.0
        
        if action == 1 and self.position == 0 and self.capital > 0:
            if self.consecutive_losses >= 3:
                return self._get_state(), -0.05, self.done
            
            buy_price = price * (1 + self.slippage)
            
            self.current_sl_pct, self.current_tp_pct = self._calculate_dynamic_sl_tp(buy_price)
            
            self.dynamic_sl = buy_price * (1 - self.current_sl_pct)
            self.dynamic_tp = buy_price * (1 + self.current_tp_pct)
            
            risk_amount = self.capital * self.current_sl_pct
            position_value = min(risk_amount * 10, self.capital * 0.5)
            
            cost = position_value * self.fee_rate
            self.position = (position_value - cost) / buy_price
            self.capital -= position_value
            self.entry_price = buy_price
            self.entry_idx = self.current_idx
            self.trailing_activated = False
            self.trailing_peak = buy_price
        
        elif self.position > 0:
            pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-10)
            vol_ratio = self._get_atr() / (self._get_atr_mean() + 1e-10)
            adx = self._get_adx()
            
            self.dynamic_sl, self.trailing_activated = self._update_trailing(
                price, high, self.entry_price, pnl_pct, self._get_atr(), self._get_atr_mean(), adx
            )
            
            exit_triggered = False
            exit_reason = ""
            
            if low <= self.dynamic_sl:
                exit_price = self.dynamic_sl * (1 - self.slippage)
                exit_reason = "stop_loss"
                exit_triggered = True
            elif high >= self.dynamic_tp:
                exit_price = self.dynamic_tp * (1 - self.slippage)
                exit_reason = "take_profit"
                exit_triggered = True
            
            if exit_triggered and action == 2:
                gross = self.position * exit_price
                cost = gross * self.fee_rate
                net = gross - cost
                
                pnl_pct = (exit_price - self.entry_price) / (self.entry_price + 1e-10)
                reward = pnl_pct - (cost / (gross + 1e-10))
                
                self.capital += net
                self.trades.append({
                    'entry_idx': self.entry_idx,
                    'exit_idx': self.current_idx,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': self.current_idx - self.entry_idx,
                    'exit_reason': exit_reason
                })
                
                self.returns_history.append(pnl_pct)
                
                if pnl_pct > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                
                self.position = 0.0
                self.entry_price = 0.0
                self.trailing_activated = False
        
        port_val = self.capital + self.position * price
        
        if port_val > self.peak_capital:
            self.peak_capital = port_val
        
        drawdown = (self.peak_capital - port_val) / (self.peak_capital + 1e-10)
        
        self.current_idx += 1
        self.done = (self.current_idx >= self.n_bars - 1)
        
        if self.position > 0 and self.done:
            sell_price = price * (1 - self.slippage)
            gross = self.position * sell_price
            cost = gross * self.fee_rate
            net = gross - cost
            
            pnl_pct = (sell_price - self.entry_price) / (self.entry_price + 1e-10)
            
            self.capital += net
            self.trades.append({
                'entry_idx': self.entry_idx,
                'exit_idx': self.current_idx,
                'entry_price': self.entry_price,
                'exit_price': sell_price,
                'pnl_pct': pnl_pct,
                'bars_held': self.current_idx - self.entry_idx,
                'exit_reason': 'forced_exit'
            })
            
            self.position = 0.0
        
        reward = np.clip(reward, -0.5, 0.5)
        
        return self._get_state(), reward, self.done

    def get_portfolio_value(self) -> float:
        price = self._current_close()
        return self.capital + self.position * price

    def get_trade_statistics(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_return': (self.get_portfolio_value() - self.initial_capital) / self.initial_capital
            }
        
        wins = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0]
        losses = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_capital': final_value
        }