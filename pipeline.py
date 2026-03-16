"""
Lab 2 — Data Pipeline: Retail Sales Analysis
Module 2 — Programming for AI & Data Science

Complete each function below. Remove the TODO: comments and pass statements
as you implement each function. Do not change the function signatures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = 'data/sales_records.csv'
OUTPUT_DIR = 'output'


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} records from {filepath}")

    return df


def clean_data(df):
    df = df.copy()

    # fill missing values
    df['quantity'] = df['quantity'].fillna(df['quantity'].median())
    df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())

    # parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    print(f"Cleaned data: {len(df)} records")

    return df


def add_features(df):
    df = df.copy()

    df['revenue'] = df['quantity'] * df['unit_price']
    df['day_of_week'] = df['date'].dt.day_name()

    return df


def generate_summary(df):
    total_revenue = df['revenue'].sum()
    avg_order_value = df['revenue'].mean()
    top_category = df.groupby('product_category')['revenue'].sum().idxmax()
    record_count = len(df)

    return {
        'total_revenue': total_revenue,
        'avg_order_value': avg_order_value,
        'top_category': top_category,
        'record_count': record_count
    }


def create_visualizations(df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # --- Chart 1: revenue by category ---
    revenue_by_cat = df.groupby('product_category')['revenue'].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(revenue_by_cat.index, revenue_by_cat.values)
    ax.set_title("Revenue by Product Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue")

    fig.savefig(f'{output_dir}/revenue_by_category.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Chart 2: daily revenue trend ---
    daily_rev = df.groupby('date')['revenue'].sum().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_rev.index, daily_rev.values)
    ax.set_title("Daily Revenue Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")

    fig.savefig(f'{output_dir}/daily_revenue_trend.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Chart 3: avg order by payment ---
    avg_payment = df.groupby('payment_method')['revenue'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(avg_payment.index, avg_payment.values)
    ax.set_title("Average Order Value by Payment Method")
    ax.set_xlabel("Average Revenue")
    ax.set_ylabel("Payment Method")

    fig.savefig(f'{output_dir}/avg_order_by_payment.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_features(df)

    summary = generate_summary(df)

    print("=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    create_visualizations(df)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
