import numpy as np


YEAR_TRADING_DAYS_STOCK = 252
YEAR_TRADING_DAYS_CRYPTO = 365
YEAR_RISK_FREE_RATE = 0.025


def calc_pv(rewards: np.array):
    return np.prod(np.array(rewards) + 1)


def calc_sr(rewards: np.array, reb_freq: int) -> float:
    rewards = rewards + 1
    group_size = int(30 / reb_freq)  # Monthly group
    num_group = int(len(rewards) / group_size) if len(rewards) / group_size != 0 else 1
    grouped_reward = np.array_split(rewards, num_group)
    rewards = []
    for reward_group in grouped_reward:
        rewards.append(reward_group.prod())

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    sr = reward_mean / reward_std
    return sr


def get_num_trading_per_year(type_: str, reb_freq: int):
    # Calculate the number of trading days per year for the given type of the market
    if type_ == 'Crypto':
        return int(YEAR_TRADING_DAYS_CRYPTO / reb_freq)
    elif 'stocks' in type_ or 'crypto' in type_:
        return int(YEAR_TRADING_DAYS_STOCK / reb_freq)
    else:
        raise ValueError(f"Unsupported type: {type_}")


def calc_arr(type_: str, rewards: np.array, reb_freq: int) -> float:
    # Calculate the Annualized Rate of Return (ARR) in Percentage (%)
    return np.mean(rewards * 100) * get_num_trading_per_year(type_=type_, reb_freq=reb_freq)


def calc_avol(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Annualized Volatility (AVol)
    return np.std(rewards) * np.sqrt(get_num_trading_per_year(type_=type_, reb_freq=reb_freq))


def calc_mdd(rewards: np.array, **kwargs) -> float:
    # Calculate the Maximum DrawDown (MDD) in Percentage (%)
    pv, pivot_pv, mdd = 1.0, 1.0, 0.0
    for reward in rewards:
        pv *= (1+reward)
        if pv > pivot_pv:
            pivot_pv = pv
        else:
            mdd = max(mdd, 1-pv/pivot_pv)
    return mdd * 100


def calc_asr(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Annualized Sharpe Ratio (ASR)
    arr = calc_arr(type_=type_, rewards=rewards, reb_freq=reb_freq)
    avol = calc_avol(type_=type_, rewards=rewards, reb_freq=reb_freq) * 100
    return arr / avol


def calc_cr(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Calmar Ratio (CR)
    arr = calc_arr(type_=type_, rewards=rewards, reb_freq=reb_freq)
    mdd = calc_mdd(rewards=rewards)
    return arr / mdd


def calc_sor(type_: str, rewards: np.array, reb_freq: int):
    # Calculate the Sortino Ratio (SoR)
    arr = calc_arr(type_=type_, rewards=rewards, reb_freq=reb_freq)
    down_rets = np.minimum(rewards * 100, 0)
    downside_deviation = np.linalg.norm(down_rets - np.mean(down_rets), 2) / np.sqrt(len(down_rets)) * \
                         np.sqrt(get_num_trading_per_year(type_=type_, reb_freq=reb_freq))
    return arr / downside_deviation


def calc_all_metrics(res: dict):
    return calc_arr(**res), calc_asr(**res), calc_mdd(**res), calc_avol(**res), calc_cr(**res), calc_sor(**res)




