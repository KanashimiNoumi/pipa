def get_signal(row, rsi_low=35, rsi_high=65):
    """Возвращает учебную метку, а не рекомендацию покупать или продавать."""
    if row["ema_fast"] > row["ema_slow"] and row["rsi"] < rsi_high:
        return "SIMULATED_LONG"
    if row["ema_fast"] < row["ema_slow"] and row["rsi"] > rsi_low:
        return "SIMULATED_EXIT"
    return "WAIT"
