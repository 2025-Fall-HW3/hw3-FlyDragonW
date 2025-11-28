"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # === Strong Strategy: Top-3 Momentum + Risk Parity ===
        # lookback 用 self.lookback，預設 50，但你可以改成 60 或 90

        K = 3  # 只挑最強的 3 個 sector
        
        eps = 1e-8

        for current_date in self.price.index:

            window_returns = self.returns.loc[:current_date, assets].tail(self.lookback)

            # 如果天數還沒到 lookback，就跳過
            if len(window_returns) < self.lookback:
                continue

            # === 1. Momentum: 過去 lookback 天累積報酬 ===
            cumret = (1.0 + window_returns).prod() - 1.0

            # 排序，挑出最強的 Top-K
            topK = cumret.sort_values(ascending=False).head(K).index

            # === 2. Top-K 裡做 inverse-vol risk parity ===
            vol = window_returns[topK].std()
            vol = vol.replace(0, np.nan)
            inv_vol = 1.0 / vol
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

            if inv_vol.isna().all():
                # 特殊情況：均分
                weights_topk = pd.Series(1.0 / K, index=topK)
            else:
                weights_topk = inv_vol / inv_vol.sum()

            # === 3. 填寫今天的權重 ===
            for col in self.price.columns:
                if col == self.exclude:
                    self.portfolio_weights.loc[current_date, col] = 0.0
                elif col in topK:
                    self.portfolio_weights.loc[current_date, col] = weights_topk[col]
                else:
                    self.portfolio_weights.loc[current_date, col] = 0.0

        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
