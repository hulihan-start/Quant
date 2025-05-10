'''
workflow:
func: cal_trading_cost
func: find best allocation by brute force
func: tune parameters by grid_search ( in real world we use SGD)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 25) # max number or rows to be displayed
plt.rcParams['figure.figsize'] = [10, 6]
idx = pd.IndexSlice

import warnings
warnings.filterwarnings('ignore')


# Trading Cost
def compute_cost(split, venues, order_size, lambda_o, lambda_u, theta):
    executed = 0
    cash_spent = 0
    for i in range(0,len(venues)): # i is the information of time t in different venues
        exe = min(split[i], venues[i]['ask_size'])
        executed += exe
        cash_spent += exe * (venues[i]['ask'] + venues[i]['fee'])
        maker_rebate = max(split[i] - exe, 0) * venues[i]['rebate']     # why the rebate is calculated based on unexecuted size?
        cash_spent -= maker_rebate

    underfill = max(order_size-executed, 0)
    overfill  = max(executed-order_size, 0)
    risk_pen  = theta * (underfill + overfill)
    cost_pen  = lambda_u * underfill + lambda_o * overfill
    return cash_spent + risk_pen + cost_pen


#allocater
def allocate(order_size, venues, λ_over, λ_under, θ_queue):
    step        =  100                    # search in 100-share chunks
    splits      =  [[]]                   # start with an empty allocation list
    for v in range(len(venues)):
        new_splits = []

        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size-used, venues[v]['ask_size'])
            for q in range(0, int(max_v)+1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost  = float('inf')
    best_split = []
    for alloc in splits:
        if sum(alloc) != order_size:
            continue

        cost = compute_cost(alloc, venues,
                            order_size, λ_over, λ_under, θ_queue)

        if cost < best_cost:
            best_cost  = cost
            best_split = alloc
    return best_split, best_cost



# Best Ask Strategy
def execute_order_best_ask(snapshots, total_order_size, lambda_o, lambda_u, theta):
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    for t, venue in enumerate(snapshots):
        if remaining_order <= 0:
            break

        best_id = min(range(len(venue)), key=lambda i: venue[i]['ask'])
        best_venue = venue[best_id]
        best_size = best_venue['ask_size']

        executed = min(remaining_order, best_size)
        if executed == 0:
            continue

        alloc = [0] * len(venue)
        alloc[best_id] = executed

        cost = compute_cost(alloc, venue, executed, lambda_o, lambda_u, theta)
        total_cost += cost
        total_executed += executed
        remaining_order -= executed

        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None

    return execution_log, total_cost, avg_fill_price


# TWAP Strategy
def execute_order_twap(snapshots, timestamps, total_order_size, lambda_o, lambda_u, theta, bucket_seconds=60):
    start_time = timestamps[0]
    total_seconds = (timestamps[-1] - start_time).total_seconds()
    num_buckets = int(total_seconds // bucket_seconds) + 1

    shares_per_bucket = total_order_size // num_buckets
    remainder = total_order_size - shares_per_bucket * num_buckets

    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []
    used_buckets = set()

    for t in range(len(snapshots)):
        if remaining_order <= 0:
            break

        bucket_idx = int((timestamps[t] - start_time).total_seconds() // bucket_seconds)
        if bucket_idx in used_buckets:
            continue

        snapshot = snapshots[t]
        shares_to_buy = shares_per_bucket + (remainder if bucket_idx == num_buckets - 1 else 0)

        alloc = [0] * len(snapshot)
        executed = 0

        for i, venue in enumerate(snapshot):
            buy_amt = min(venue['ask_size'], shares_to_buy - executed)
            alloc[i] = buy_amt
            executed += buy_amt
            if executed >= shares_to_buy:
                break

        if executed == 0:
            continue

        cost = compute_cost(alloc, snapshot, executed, lambda_o, lambda_u, theta)
        total_cost += cost
        total_executed += executed
        remaining_order -= executed

        used_buckets.add(bucket_idx)
        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None
    return execution_log, total_cost, avg_fill_price



def execute_order_vwap(snapshots, total_order_size, lambda_o, lambda_u, theta):
    """
    Execute order using VWAP: split each execution proportionally to ask sizes.
    """
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    for t, snapshot in enumerate(snapshots):
        if remaining_order <= 0:
            break

        total_ask_size = sum(v['ask_size'] for v in snapshot)
        if total_ask_size == 0:
            continue

        alloc = [min(remaining_order * v['ask_size'] / total_ask_size, v['ask_size']) for v in snapshot]
        alloc = [int(s) for s in alloc]
        executed = sum(alloc)

        if executed == 0:
            continue

        cost = compute_cost(alloc, snapshot, executed, lambda_o, lambda_u, theta)
        total_cost += cost
        total_executed += executed
        remaining_order -= executed

        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None

    return execution_log, total_cost, avg_fill_price


# trading execution
def execute_order_my_strategy1(snapshots, total_order_size, λ_over, λ_under, θ_queue):
    remaining_order = total_order_size
    total_cost = 0
    total_executed =0
    execution_log = []

    for t, snapshot in enumerate(snapshots):
        if remaining_order <= 0:
            break

        total_ask_size = sum(venue['ask_size'] for venue in snapshot)
        if total_ask_size < remaining_order:
            continue

        alloc, cost = allocate(remaining_order, snapshot, λ_over, λ_under, θ_queue)
        executed = sum(min(alloc[i], snapshot[i]['ask_size']) for i in range(len(alloc)))
        remaining_order -= executed
        total_executed += executed
        total_cost += cost

        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None

    return execution_log, total_cost, avg_fill_price

# strategy 2: buy every 15 seconds
def execute_order_my_strategy2(snapshots, timestamps, total_order_size, λ_over, λ_under, θ_queue,
                                time_interval=15, last_minute_interval=5):
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    # Convert to UNIX time float
    float_timestamps = [ts.timestamp() for ts in timestamps]
    allocator_last_time = float_timestamps[0]
    end_time = float_timestamps[-1]

    for t in range(len(snapshots)):
        if remaining_order <= 0:
            break

        snapshot = snapshots[t]
        ts = float_timestamps[t]
        time_left = end_time - ts
        allocator_interval = last_minute_interval if time_left <= 60 else time_interval
        time_since_last_allocator = ts - allocator_last_time

        if time_since_last_allocator >= allocator_interval:
            allocator_last_time = ts  # update even if fallback is used
            total_ask_size = sum(venue['ask_size'] for venue in snapshot)

            if total_ask_size >= remaining_order:
                buy_qty = int(remaining_order // 100) * 100

                alloc, cost = allocate(buy_qty, snapshot, λ_over, λ_under, θ_queue)
                executed = sum(min(alloc[i], snapshot[i]['ask_size']) for i in range(len(alloc)))
                remaining_order -= executed
                total_executed += executed
                total_cost += cost
                execution_log.append({
                    'snapshot': t,
                    'strategy': 'allocator',
                    'alloc': alloc,
                    'executed': executed,
                    'remaining': remaining_order,
                    'cost': cost
                })

            else:
                best_venue_idx = min(range(len(snapshot)), key=lambda i: snapshot[i]['ask'])
                venue = snapshot[best_venue_idx]
                buy_qty = min(venue['ask_size'], remaining_order)
                cost = 0
                if buy_qty > 0:
                    cost = buy_qty * (venue['ask'] + venue['fee'])
                    remaining_order -= buy_qty
                    total_executed += buy_qty
                    total_cost += cost

                execution_log.append({
                    'snapshot': t,
                    'strategy': 'fallback',
                    'venue': best_venue_idx,
                    'executed': buy_qty,
                    'remaining': remaining_order,
                    'cost': cost
                })


    avg_fill_price = total_cost / total_executed if total_executed > 0 else None
    return execution_log, total_cost, avg_fill_price

def execute_order_my_strategy3(snapshots, timestamps, total_order_size, λ_over, λ_under, θ_queue,
                              last_minute_interval=30):
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    float_timestamps = [ts.timestamp() for ts in timestamps]
    end_time = float_timestamps[-1]

    # Initial buy at time 0
    t = 0
    snapshot = snapshots[0]
    best_venue_idx = min(range(len(snapshot)), key=lambda i: snapshot[i]['ask'])
    venue = snapshot[best_venue_idx]
    qty = min(100, venue['ask_size'], remaining_order)
    price = venue['ask'] + venue['fee']
    cost = qty * price
    total_cost += cost
    total_executed += qty
    remaining_order -= qty

    last_price = price
    last_qty = qty
    last_trade_time = float_timestamps[0]

    execution_log.append({
        'snapshot': t,
        'alloc': [qty],
        'executed': qty,
        'remaining': remaining_order,
        'cost': cost
    })

    for t in range(1, len(snapshots)):
        if remaining_order <= 0:
            break

        snapshot = snapshots[t]
        ts = float_timestamps[t]
        time_left = end_time - ts
        best_venue_idx = min(range(len(snapshot)), key=lambda i: snapshot[i]['ask'])
        venue = snapshot[best_venue_idx]
        current_price = venue['ask'] + venue['fee']

        if time_left <= 240:'''
workflow:
func: cal_trading_cost
func: find best allocation by brute force
func: tune parameters by grid_search ( in real world we use SGD)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 25) # max number or rows to be displayed
plt.rcParams['figure.figsize'] = [10, 6]
idx = pd.IndexSlice

import warnings
warnings.filterwarnings('ignore')


# Trading Cost
def compute_cost(split, venues, order_size, lambda_o, lambda_u, theta):
    executed = 0
    cash_spent = 0
    for i in range(0,len(venues)): # i is the information of time t in different venues
        exe = min(split[i], venues[i]['ask_size'])
        executed += exe
        cash_spent += exe * (venues[i]['ask'] + venues[i]['fee'])
        maker_rebate = max(split[i] - exe, 0) * venues[i]['rebate']     # why the rebate is calculated based on unexecuted size?
        cash_spent -= maker_rebate

    underfill = max(order_size-executed, 0)
    overfill  = max(executed-order_size, 0)
    risk_pen  = theta * (underfill + overfill)
    cost_pen  = lambda_u * underfill + lambda_o * overfill
    return cash_spent + risk_pen + cost_pen


#allocater
def allocate(order_size, venues, λ_over, λ_under, θ_queue):
    step        =  100                    # search in 100-share chunks
    splits      =  [[]]                   # start with an empty allocation list
    for v in range(len(venues)):
        new_splits = []

        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size-used, venues[v]['ask_size'])
            for q in range(0, int(max_v)+1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost  = float('inf')
    best_split = []
    for alloc in splits:
        if sum(alloc) != order_size:
            continue

        cost = compute_cost(alloc, venues,
                            order_size, λ_over, λ_under, θ_queue)

        if cost < best_cost:
            best_cost  = cost
            best_split = alloc
    return best_split, best_cost



# Best Ask Strategy
def execute_order_best_ask(snapshots, total_order_size, lambda_o, lambda_u, theta):
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    for t, venue in enumerate(snapshots):
        if remaining_order <= 0:
            break

        best_id = min(range(len(venue)), key=lambda i: venue[i]['ask'])
        best_venue = venue[best_id]
        best_size = best_venue['ask_size']

        executed = min(remaining_order, best_size)
        if executed == 0:
            continue

        alloc = [0] * len(venue)
        alloc[best_id] = executed

        cost = compute_cost(alloc, venue, executed, lambda_o, lambda_u, theta)
        total_cost += cost
        total_executed += executed
        remaining_order -= executed

        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None

    return execution_log, total_cost, avg_fill_price


# TWAP Strategy
def execute_order_twap(snapshots, timestamps, total_order_size, lambda_o, lambda_u, theta, bucket_seconds=60):
    start_time = timestamps[0]
    total_seconds = (timestamps[-1] - start_time).total_seconds()
    num_buckets = int(total_seconds // bucket_seconds) + 1

    shares_per_bucket = total_order_size // num_buckets
    remainder = total_order_size - shares_per_bucket * num_buckets

    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []
    used_buckets = set()

    for t in range(len(snapshots)):
        if remaining_order <= 0:
            break

        bucket_idx = int((timestamps[t] - start_time).total_seconds() // bucket_seconds)
        if bucket_idx in used_buckets:
            continue

        snapshot = snapshots[t]
        shares_to_buy = shares_per_bucket + (remainder if bucket_idx == num_buckets - 1 else 0)

        alloc = [0] * len(snapshot)
        executed = 0

        for i, venue in enumerate(snapshot):
            buy_amt = min(venue['ask_size'], shares_to_buy - executed)
            alloc[i] = buy_amt
            executed += buy_amt
            if executed >= shares_to_buy:
                break

        if executed == 0:
            continue

        cost = compute_cost(alloc, snapshot, executed, lambda_o, lambda_u, theta)
        total_cost += cost
        total_executed += executed
        remaining_order -= executed

        used_buckets.add(bucket_idx)
        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None
    return execution_log, total_cost, avg_fill_price



def execute_order_vwap(snapshots, total_order_size, lambda_o, lambda_u, theta):
    """
    Execute order using VWAP: split each execution proportionally to ask sizes.
    """
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    for t, snapshot in enumerate(snapshots):
        if remaining_order <= 0:
            break

        total_ask_size = sum(v['ask_size'] for v in snapshot)
        if total_ask_size == 0:
            continue

        alloc = [min(remaining_order * v['ask_size'] / total_ask_size, v['ask_size']) for v in snapshot]
        alloc = [int(s) for s in alloc]
        executed = sum(alloc)

        if executed == 0:
            continue

        cost = compute_cost(alloc, snapshot, executed, lambda_o, lambda_u, theta)
        total_cost += cost
        total_executed += executed
        remaining_order -= executed

        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None

    return execution_log, total_cost, avg_fill_price


# trading execution
def execute_order_my_strategy1(snapshots, total_order_size, λ_over, λ_under, θ_queue):
    remaining_order = total_order_size
    total_cost = 0
    total_executed =0
    execution_log = []

    for t, snapshot in enumerate(snapshots):
        if remaining_order <= 0:
            break

        total_ask_size = sum(venue['ask_size'] for venue in snapshot)
        if total_ask_size < remaining_order:
            continue

        alloc, cost = allocate(remaining_order, snapshot, λ_over, λ_under, θ_queue)
        executed = sum(min(alloc[i], snapshot[i]['ask_size']) for i in range(len(alloc)))
        remaining_order -= executed
        total_executed += executed
        total_cost += cost

        execution_log.append({
            'snapshot': t,
            'alloc': alloc,
            'executed': executed,
            'remaining': remaining_order,
            'cost': cost
        })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None

    return execution_log, total_cost, avg_fill_price

# strategy 2: buy every 5 seconds
def execute_order_my_strategy2(snapshots, timestamps, total_order_size, λ_over, λ_under, θ_queue,
                                time_interval=5, last_minute_interval=1):
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    # Convert to UNIX time float
    float_timestamps = [ts.timestamp() for ts in timestamps]
    allocator_last_time = float_timestamps[0]
    end_time = float_timestamps[-1]

    for t in range(len(snapshots)):
        if remaining_order <= 0:
            break

        snapshot = snapshots[t]
        ts = float_timestamps[t]
        time_left = end_time - ts
        allocator_interval = last_minute_interval if time_left <= 60 else time_interval
        time_since_last_allocator = ts - allocator_last_time

        if time_since_last_allocator >= allocator_interval:
            allocator_last_time = ts  # update even if fallback is used
            total_ask_size = sum(venue['ask_size'] for venue in snapshot)

            if total_ask_size >= remaining_order:
                buy_qty = int(remaining_order // 100) * 100

                alloc, cost = allocate(buy_qty, snapshot, λ_over, λ_under, θ_queue)
                executed = sum(min(alloc[i], snapshot[i]['ask_size']) for i in range(len(alloc)))
                remaining_order -= executed
                total_executed += executed
                total_cost += cost
                execution_log.append({
                    'snapshot': t,
                    'strategy': 'allocator',
                    'alloc': alloc,
                    'executed': executed,
                    'remaining': remaining_order,
                    'cost': cost
                })

            else:
                best_venue_idx = min(range(len(snapshot)), key=lambda i: snapshot[i]['ask'])
                venue = snapshot[best_venue_idx]
                buy_qty = min(venue['ask_size'], remaining_order)
                cost = 0
                if buy_qty > 0:
                    cost = buy_qty * (venue['ask'] + venue['fee'])
                    remaining_order -= buy_qty
                    total_executed += buy_qty
                    total_cost += cost

                execution_log.append({
                    'snapshot': t,
                    'strategy': 'fallback',
                    'venue': best_venue_idx,
                    'executed': buy_qty,
                    'remaining': remaining_order,
                    'cost': cost
                })


    avg_fill_price = total_cost / total_executed if total_executed > 0 else None
    return execution_log, total_cost, avg_fill_price


def execute_order_my_strategy3(snapshots, timestamps, total_order_size, λ_over, λ_under, θ_queue,
                              last_interval=15):
    remaining_order = total_order_size
    total_cost = 0
    total_executed = 0
    execution_log = []

    float_timestamps = [ts.timestamp() for ts in timestamps]
    end_time = float_timestamps[-1]

    # Initial buy at time 0
    t = 0
    snapshot = snapshots[0]
    best_venue_idx = min(range(len(snapshot)), key=lambda i: snapshot[i]['ask'])
    venue = snapshot[best_venue_idx]
    qty = min(100, venue['ask_size'], remaining_order)
    price = venue['ask'] + venue['fee']
    cost = qty * price
    total_cost += cost
    total_executed += qty
    remaining_order -= qty

    last_price = price
    last_qty = qty
    last_trade_time = float_timestamps[0]

    execution_log.append({
        'snapshot': t,
        'alloc': [qty],
        'executed': qty,
        'remaining': remaining_order,
        'cost': cost
    })

    for t in range(1, len(snapshots)):
        if remaining_order <= 0:
            break

        snapshot = snapshots[t]
        ts = float_timestamps[t]
        time_left = end_time - ts
        best_venue_idx = min(range(len(snapshot)), key=lambda i: snapshot[i]['ask'])
        venue = snapshot[best_venue_idx]
        current_price = venue['ask'] + venue['fee']

        if time_left <= 120:
            if ts - last_trade_time >= last_interval:
                buy_qty = min(venue['ask_size'], remaining_order)
                if buy_qty > 0:
                    cost = buy_qty * current_price
                    total_cost += cost
                    total_executed += buy_qty
                    remaining_order -= buy_qty
                    last_trade_time = ts

                    execution_log.append({
                        'snapshot': t,
                        'alloc': [buy_qty],
                        'executed': buy_qty,
                        'remaining': remaining_order,
                        'cost': cost
                    })
            continue

        if current_price < last_price:
            target_qty = int(last_qty + 0.1 * total_order_size)
            max_qty = int(0.1 * total_order_size)
            buy_qty = min(target_qty, max_qty, remaining_order)

            total_ask_size = sum(venue['ask_size'] for venue in snapshot)
            if total_ask_size >= buy_qty:
                buy_qty = int(buy_qty // 100) * 100
                alloc, cost = allocate(buy_qty, snapshot, λ_over, λ_under, θ_queue)
                executed = sum(min(alloc[i], snapshot[i]['ask_size']) for i in range(len(alloc)))
                if executed > 0:
                    total_cost += cost
                    total_executed += executed
                    remaining_order -= executed
                    last_price = current_price
                    last_qty = executed
                    last_trade_time = ts

                    execution_log.append({
                        'snapshot': t,
                        'alloc': alloc,
                        'executed': executed,
                        'remaining': remaining_order,
                        'cost': cost
                    })
            else:
                buy_qty = min(venue['ask_size'], remaining_order)
                if buy_qty > 0:
                    cost = buy_qty * current_price
                    total_cost += cost
                    total_executed += buy_qty
                    remaining_order -= buy_qty
                    last_price = current_price
                    last_qty = buy_qty
                    last_trade_time = ts

                    execution_log.append({
                        'snapshot': t,
                        'alloc': [buy_qty],
                        'executed': buy_qty,
                        'remaining': remaining_order,
                        'cost': cost
                    })

    avg_fill_price = total_cost / total_executed if total_executed > 0 else None
    return execution_log, total_cost, avg_fill_price



# grid search for parameters
def grid_search_parameters(snapshots, total_order_size, lambda_o_vals, lambda_u_vals, theta_vals):
    best_params = None
    best_cost = float('inf')
    best_log = None

    for lambda_o in lambda_o_vals:
        for lambda_u in lambda_u_vals:
            for theta in theta_vals:
                log, cost, _ = execute_order_my_strategy1(snapshots, total_order_size, lambda_o, lambda_u, theta)
                if cost < best_cost:
                    best_cost = cost
                    best_params = {
                        'lambda_over': lambda_o,
                        'lambda_under': lambda_u,
                        'theta_queue': theta
                    }
                    best_log = log

    return best_params, best_cost, best_log



# plot
def plot_cum_costs(logs_dict, filename='results.png'):
    n = len(logs_dict)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows), sharex=True)
    axes = axes.flatten()

    for ax, (label, log) in zip(axes, logs_dict.items()):
        snapshots = [entry['snapshot'] for entry in log]
        costs = [entry['cost'] for entry in log]
        cumulative = [sum(costs[:i+1]) for i in range(len(costs))]

        ax.plot(snapshots, cumulative, label=label, linewidth=2)
        ax.scatter(snapshots, cumulative, color='black', s=20)
        ax.set_title(label)
        ax.set_ylabel('Cumulative Cost')
        ax.grid(True)
        ax.legend()

    for i in range(len(logs_dict), len(axes)):
        fig.delaxes(axes[i])  # remove unused axes

    plt.xlabel('Execution Step')
    plt.tight_layout()
    plt.savefig(filename)


# data clean
def load_data(file_name):
    '''
    :param file_name:
    :return: snapshots, a list of venues (list of dict)
    '''
    df = pd.read_csv(file_name)
    df = df[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']]
    import matplotlib.pyplot as plt

    df['ts_event'] = pd.to_datetime(df['ts_event'])

    plt.figure(figsize=(12, 6))

    for venue, group in df.groupby('publisher_id'):
        plt.plot(group['ts_event'], group['ask_px_00'], label=f'Venue {venue}', linewidth=0.8)

    plt.xlabel('Timestamp')
    plt.ylabel('Ask Price')
    plt.title('Ask Price over Time per Venue')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values(by = ['ts_event','publisher_id'])
    df = df.drop_duplicates(subset = ['ts_event','publisher_id'],keep='first')


    snapshots = []
    timestamps = []
    for timestamp, group in df.groupby('ts_event'):
        venue = []
        for _, row in group.iterrows():
            venue.append({
                'ask': float(row['ask_px_00']),
                'ask_size':float(row['ask_sz_00']),
                'fee':0.002,
                'rebate': -0.001
            })
        snapshots.append(venue)
        timestamps.append(timestamp)

    return timestamps, snapshots


if __name__ == "__main__":
    timestamps, snapshots = load_data('../l1_day.csv')
    total_order_size = 5000

    # Step 1: Parameter tuning first
    best_params, best_cost, best_log = grid_search_parameters(
        snapshots, total_order_size,
        lambda_o_vals=[0.05, 0.1, 0.2],
        lambda_u_vals=[0.01, 0.05, 0.1],
        theta_vals=[0.005, 0.01]
    )

    lambda_o = best_params['lambda_over']
    lambda_u = best_params['lambda_under']
    theta = best_params['theta_queue']


    # Step 2: Run all strategies with tuned parameters
    logs = {}

    logs['BestAsk'], cost_ba, avg_ba = execute_order_best_ask(snapshots, total_order_size, lambda_o, lambda_u, theta)
    logs['TWAP'], cost_twap, avg_twap = execute_order_twap(snapshots, timestamps, total_order_size, lambda_o, lambda_u, theta)
    logs['VWAP'], cost_vwap, avg_vwap = execute_order_vwap(snapshots, total_order_size, lambda_o, lambda_u, theta)
    logs['strategy1'], cost_1, avg_1 = execute_order_my_strategy1(snapshots, total_order_size, lambda_o, lambda_u, theta)
    logs['strategy2'], cost_2, avg_2 = execute_order_my_strategy2(snapshots, timestamps, total_order_size, lambda_o, lambda_u, theta)
    logs['strategy3'], cost_3, avg_3 = execute_order_my_strategy3(snapshots, timestamps, total_order_size,
                                                                         lambda_o, lambda_u, theta)

    print()
    plot_cum_costs(logs, filename='results.png')

    output = {}

    # === Final output according to spec ===
    output = {
        "Tuned_Parameters": best_params,
        "SmartRouter": {
            "total_cost": best_cost,
            "avg_fill_price": best_cost / total_order_size,
            "savings_vs_bestask_bps": round(10000 * (cost_ba - best_cost) / cost_ba, 2),
            "savings_vs_twap_bps": round(10000 * (cost_twap - best_cost) / cost_twap, 2),
            "savings_vs_vwap_bps": round(10000 * (cost_vwap - best_cost) / cost_vwap, 2)
        },
        "BestAsk": {
            "total_cost": cost_ba,
            "avg_fill_price": avg_ba
        },
        "TWAP": {
            "total_cost": cost_twap,
            "avg_fill_price": avg_twap
        },
        "VWAP": {
            "total_cost": cost_vwap,
            "avg_fill_price": avg_vwap
        },
        "strategy2": {
            "total_cost": cost_2,
            "avg_fill_price": avg_2
        },
        "strategy3": {
            "total_cost": cost_3,
            "avg_fill_price": avg_3
        }
    }

    print(json.dumps(output, indent=2))