# Cont & Kukanov Model

## Overview
This project implements a Smart Order Router based on the static cost model from Cont & Kukanov (2012), "Optimal Order Placement in Limit Order Markets." The router optimally splits a 5,000-share buy order across multiple venues to minimize execution cost, penalizing underfills, overfills, and queue risk. It supports:

** `Cont-Kukanov optimization` via random search on three parameters: 
* `lambda_over`: Cost penalty for overfilling
* `lambda_under`: Cost penalty for underfilling
* `theta_queue`: Queue risk penalty

** `Baseline strategies`:
* `Best ask`: Always takes liquidity from the venue with the loIst ask.
* `TWAP`: Executes in fixed 60-second intervals using the best venue at that moment.
* `VWAP`: Uses volume-Iighted average price across all venues.

## Project Structure

* `backtest.py`: Main script that loads market data, builds venue snapshots, runs the allocator, and compares against baselines.
* `l1_day.csv`: Input file of mocked market data.
* Output is printed as a JSON object to stdout.

## Parameter Search

It supports flexible parameter tuning using `numpy.random.uniform`.

The best parameter combination is selected based on the lowest total cash spent.

## Functions

* `backtest`: Main function executing orders over snapshots using optimized allocation.
* `allocate`: Brute-force allocator that generates all valid allocations and selects the one with the lowest cost.
* `compute_cost`: Calculates execution cost including price, underfill/overfill penalties, and market impact.
* `best_ask`: Executes trades at the venue with the best ask.
* `twap`: Time-weighted average price strategy.
* `vwap`: Volume-weighted average price strategy.

## Output

The script prints a single JSON object containing:

* Best parameter set found
* Total cash spent and average price for that parameter set
* Corresponding metrics for best-ask, TWAP, and VWAP
* Savings in basis points (bps) relative to each baseline
