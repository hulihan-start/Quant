import pandas as pd
import numpy as np
import json
import time

np.random.seed(42)

class Venue:
    def __init__(self, ask, ask_size, fee, rebate):
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate

def backtest(snapshots, lambda_over, lambda_under, theta_queue, total_order=5000):
    remaining = total_order
    cash = 0
    total_cost = 0
    no_exe_cnt = 0
    for ts, venues in snapshots:
        if remaining <= 0:
            break
        if no_exe_cnt >= 3:
            lambda_under *= 5
            theta_queue = 0
        alloc, cost = allocate(remaining, venues, lambda_over, lambda_under, theta_queue)
        step_exe = 0
        total_cost += cost
        for i, qty in enumerate(alloc):
            venue = venues[i]
            exe = min(qty, venue.ask_size, remaining)
            cash += exe * (venue.ask + venue.fee)
            cash -= max(qty - exe, 0) * venue.rebate
            remaining -= exe
            step_exe += exe
        if step_exe == 0:
            no_exe_cnt += 1
        else:
            no_exe_cnt = 0
    executed = total_order - remaining
    avg_price = cash / executed if executed > 0 else None
    return cash, avg_price, total_cost

def allocate(order_size, venues, lambda_over, lambda_under, theta_queue):
    step = 100
    splits = [[]]
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            if used >= order_size:
                continue
            max_v = min(order_size-used, venues[v].ask_size)
            for q in range(0, int(max_v)+1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = []
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    return best_split, best_cost

def compute_cost(split, venues, order_size, lambda_o, lambda_u, theta):
    executed = 0
    cash_spent = 0
    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i]-exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate
    underfill = max(order_size-executed, 0)
    overfill = max(executed-order_size, 0)
    risk_pen = theta * (underfill + overfill)
    cost_pen = lambda_u * underfill + lambda_o * overfill
    return cash_spent+risk_pen+cost_pen


def best_ask(snapshots, total_order):
    remaining = total_order
    cash = 0
    for ts, venues in snapshots:
        if remaining <= 0:
            break
        best = min(venues, key=lambda x: x.ask)
        exe = min(remaining, best.ask_size)
        cash += exe*best.ask
        remaining -= exe
    executed = total_order - remaining
    avg_price = cash/executed if executed>0 else None
    return cash, avg_price

def twap(snapshots, total_order, interval_sec=60):
    remaining = total_order
    time_list = [ts for ts,_ in snapshots]
    total_time = (time_list[-1] - time_list[0]).total_seconds()
    buckets = total_order / int(total_time/interval_sec)
    cash=0
    bucket_idx = 0
    for ts, venues in snapshots:
        while bucket_idx * interval_sec + time_list[0].timestamp() <= ts.timestamp() and remaining > 0:
            best = min(venues, key=lambda x: x.ask)
            exe = min(buckets, best.ask_size, remaining)
            cash += exe*best.ask
            remaining -= exe
            bucket_idx += 1
    executed = total_order - remaining
    avg_price = cash/executed if executed > 0 else None
    return cash, avg_price

def vwap(snapshots, total_order):
    size = 0
    price = 0
    for ts, venues in snapshots:
        for venue in venues:
            size += venue.ask_size
            price += venue.ask*venue.ask_size
    vwap = price / size if size >0 else None
    cash = total_order * vwap
    return cash, vwap

if __name__ == "__main__":
    df = pd.read_csv('../l1_day.csv', usecols=["ts_event", "publisher_id", "ask_px_00", "ask_sz_00"], parse_dates=["ts_event"])
    
    df.dropna()
    df = df.sort_values(['ts_event', 'publisher_id'])
    df = df.groupby(['ts_event', 'publisher_id']).first()
    
    cnt = 0
    snapshots = []
    for ts, group in df.groupby('ts_event'):
        cnt += 1
        venues = []
        for _, row in group.iterrows():
            # setting from paper page 27 r_NSDQ
            venues.append(Venue(
                ask=row.values[0],
                ask_size=row.values[1],
                fee=0.29,
                rebate=0.2
            ))
        snapshots.append((ts, venues))
    
    num_trials = 20

    lambda_over_range = (0.01, 0.1)
    lambda_under_range = (0.01, 0.1)
    theta_queue_range = (0.01, 0.05)

    lambda_over_list = np.random.uniform(*lambda_over_range, size=num_trials)
    lambda_under_list = np.random.uniform(*lambda_under_range, size=num_trials)
    theta_queue_list = np.random.uniform(*theta_queue_range, size=num_trials)

    param_grid = list(zip(lambda_over_list, lambda_under_list, theta_queue_list))

    # param_grid = [[0,0,0], [0.1,0.1,0.01], [0.5,0.5,0.05]]
    t1 = time.time()
    results = []
    for lambda_o, lambda_u, theta in param_grid:
        cash, avg, cost = backtest(snapshots, lambda_o, lambda_u, theta, total_order=5000)
        results.append({'lambda_over': lambda_o,
                        'lambda_under': lambda_u,
                        'theta_queue': theta,
                        'cash': cash,
                        'avg_price': avg,
                        'cost': cost})
    best = min(results, key=lambda x: x['cash'])
    t2 = time.time()
    ba_cash, ba_avg = best_ask(snapshots, total_order=5000)
    t3 = time.time()
    tw_cash, tw_avg = twap(snapshots, total_order=5000)
    t4 = time.time()
    vw_cash, vw_avg = vwap(snapshots, total_order=5000)
    t5 = time.time()

    print((t2-t1)/num_trials, t3-t2,t4-t3, t5-t4)

    savings = {
        'best_ask_bps':  (ba_avg - best['avg_price']) / ba_avg * 1e4 if ba_avg else None,
        'twap_bps':      (tw_avg - best['avg_price']) / tw_avg * 1e4 if tw_avg else None,
        'vwap_bps':      (vw_avg - best['avg_price']) / vw_avg * 1e4 if vw_avg else None
    }

    output = {
        'best_params': {
            'lambda_over':  best['lambda_over'],
            'lambda_under': best['lambda_under'],
            'theta_queue':  best['theta_queue']
        },
        'best': {
            'total_cash': best['cash'],
            'avg_price':  best['avg_price']
        },
        'best_ask': {
            'total_cash': ba_cash,
            'avg_price':  ba_avg
        },
        'twap': {
            'total_cash': tw_cash,
            'avg_price':  tw_avg
        },
        'vwap': {
            'total_cash': vw_cash,
            'avg_price':  vw_avg
        },
        'savings_bps': savings
    }
    print(json.dumps(output, indent=2))