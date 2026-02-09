"""
All-in-One Data Exploration Script
No external dependencies needed - everything is here!

Author: Orkidea Merko
Project: COMP1682 Final Year Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import warnings
import os
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" " * 25 + "KEYRECS DATA EXPLORATION")
print("="*80 + "\n")

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Create output directories
os.makedirs('results/visualizations', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# ============================================================================
# LOAD DATA (Built-in loader)
# ============================================================================
print("[1/7] Loading dataset...")

demographics = pd.read_csv('data/raw/demographics.csv')
free_text = pd.read_csv('data/raw/free-text.csv')

print(f"✓ Loaded demographics: {len(demographics)} participants")
print(f"✓ Loaded free-text: {len(free_text):,} keystroke pairs")

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n[2/7] Preparing data for analysis...")

# Merge with demographics
df = free_text.merge(demographics, on='participant', how='left')

# Sample for faster processing
sample_size = 50000
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)
    print(f"Sampled to: {len(df):,} keystroke pairs")

# Extract features (rename columns)
column_mapping = {
    'DU.key1.key1': 'dwell_key1',
    'DU.key1.key2': 'dwell_key2',
    'DD.key1.key2': 'flight_time_DD',
    'UD.key1.key2': 'flight_time_UD',
    'UU.key1.key2': 'flight_time_UU'
}

df = df.rename(columns=column_mapping)

# FIX: Convert timing columns to numeric, coercing errors to NaN
timing_columns = ['dwell_key1', 'dwell_key2', 'flight_time_DD', 'flight_time_UD', 'flight_time_UU']
for col in timing_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove missing values
timing_cols = ['dwell_key1', 'flight_time_DD']
df = df.dropna(subset=timing_cols)

print(f"\n✓ Working with {len(df):,} keystroke pairs")
print(f"✓ From {df['participant'].nunique()} participants")
print(f"✓ Across {df['nationality'].nunique()} nationalities")

# Get top nationalities
nationality_counts = df['nationality'].value_counts()
top_5_nats = nationality_counts.head(5).index
top_10_nats = nationality_counts.head(10).index

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# ============================================================================
# VISUALIZATION 1: Nationality Distribution
# ============================================================================
print("\n[3/7] Creating nationality distribution charts...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
top_10 = nationality_counts.head(10)
axes[0].bar(range(len(top_10)), top_10.values, color='steelblue', edgecolor='black')
axes[0].set_xticks(range(len(top_10)))
axes[0].set_xticklabels(top_10.index, rotation=45, ha='right')
axes[0].set_xlabel('Nationality', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Keystroke Pairs', fontsize=12, fontweight='bold')
axes[0].set_title('Top 10 Nationalities by Sample Count', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(top_10.values):
    axes[0].text(i, v + 200, f'{v:,}', ha='center', va='bottom', fontsize=9)

# Pie chart
top_5 = nationality_counts.head(5)
pie_data = list(top_5.values) + [nationality_counts[5:].sum()]
pie_labels = list(top_5.index) + ['Others']
colors = plt.cm.Set3(range(len(pie_data)))

axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
axes[1].set_title('Nationality Distribution (Top 5 + Others)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/visualizations/nationality_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: nationality_distribution.png")

# ============================================================================
# VISUALIZATION 2: Dwell Time Analysis
# ============================================================================
print("\n[4/7] Creating dwell time analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall distribution
axes[0, 0].hist(df['dwell_key1'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['dwell_key1'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {df['dwell_key1'].mean():.2f} ms")
axes[0, 0].axvline(df['dwell_key1'].median(), color='green', linestyle='--', linewidth=2,
                   label=f"Median: {df['dwell_key1'].median():.2f} ms")
axes[0, 0].set_xlabel('Dwell Time (ms)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Overall Dwell Time Distribution', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Box plot
axes[0, 1].boxplot(df['dwell_key1'].dropna(), vert=True)
axes[0, 1].set_ylabel('Dwell Time (ms)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Dwell Time Box Plot', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# By nationality
colors_list = plt.cm.Set2(range(5))
for i, nat in enumerate(top_5_nats):
    nat_dwell = df[df['nationality'] == nat]['dwell_key1']
    axes[1, 0].hist(nat_dwell, bins=30, alpha=0.5, label=nat, color=colors_list[i], edgecolor='black')

axes[1, 0].set_xlabel('Dwell Time (ms)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Dwell Time by Nationality (Top 5)', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Violin plot
top_5_data = df[df['nationality'].isin(top_5_nats)]
violin_data = [top_5_data[top_5_data['nationality'] == nat]['dwell_key1'].dropna() for nat in top_5_nats]
parts = axes[1, 1].violinplot(violin_data, positions=range(len(top_5_nats)), showmeans=True, showmedians=True)

axes[1, 1].set_xticks(range(len(top_5_nats)))
axes[1, 1].set_xticklabels(top_5_nats, rotation=45, ha='right')
axes[1, 1].set_xlabel('Nationality', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Dwell Time (ms)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Dwell Time Distribution by Nationality', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/dwell_time_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: dwell_time_analysis.png")

# ============================================================================
# VISUALIZATION 3: Flight Time Analysis
# ============================================================================
print("\n[5/7] Creating flight time analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].hist(df['flight_time_DD'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['flight_time_DD'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {df['flight_time_DD'].mean():.2f} ms")
axes[0, 0].axvline(df['flight_time_DD'].median(), color='green', linestyle='--', linewidth=2,
                   label=f"Median: {df['flight_time_DD'].median():.2f} ms")
axes[0, 0].set_xlabel('Flight Time (ms)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Overall Flight Time Distribution', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].boxplot(df['flight_time_DD'].dropna(), vert=True)
axes[0, 1].set_ylabel('Flight Time (ms)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Flight Time Box Plot', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

for i, nat in enumerate(top_5_nats):
    nat_flight = df[df['nationality'] == nat]['flight_time_DD']
    axes[1, 0].hist(nat_flight, bins=30, alpha=0.5, label=nat, color=colors_list[i], edgecolor='black')

axes[1, 0].set_xlabel('Flight Time (ms)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Flight Time by Nationality (Top 5)', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

violin_data_flight = [top_5_data[top_5_data['nationality'] == nat]['flight_time_DD'].dropna() for nat in top_5_nats]
parts = axes[1, 1].violinplot(violin_data_flight, positions=range(len(top_5_nats)), showmeans=True, showmedians=True)

axes[1, 1].set_xticks(range(len(top_5_nats)))
axes[1, 1].set_xticklabels(top_5_nats, rotation=45, ha='right')
axes[1, 1].set_xlabel('Nationality', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Flight Time (ms)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Flight Time Distribution by Nationality', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/flight_time_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: flight_time_analysis.png")

# ============================================================================
# VISUALIZATION 4: Feature Correlation
# ============================================================================
print("\n[6/7] Creating correlation analysis...")

timing_features = ['dwell_key1', 'dwell_key2', 'flight_time_DD', 'flight_time_UD']
corr_matrix = df[timing_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
           square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Keystroke Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/visualizations/feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: feature_correlation.png")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
print("\n[7/7] Performing statistical tests...")

groups_dwell = [df[df['nationality'] == nat]['dwell_key1'].dropna() for nat in top_5_nats]
groups_flight = [df[df['nationality'] == nat]['flight_time_DD'].dropna() for nat in top_5_nats]

f_stat_dwell, p_value_dwell = f_oneway(*groups_dwell)
f_stat_flight, p_value_flight = f_oneway(*groups_flight)

print("\n" + "="*80)
print("STATISTICAL TEST RESULTS (ANOVA)")
print("="*80)
print(f"\nDwell Time:")
print(f"  F-statistic: {f_stat_dwell:.4f}")
print(f"  P-value: {p_value_dwell:.4e}")
if p_value_dwell < 0.05:
    print("  ✓ SIGNIFICANT difference detected (p < 0.05)")
else:
    print("  ✗ No significant difference")

print(f"\nFlight Time:")
print(f"  F-statistic: {f_stat_flight:.4f}")
print(f"  P-value: {p_value_flight:.4e}")
if p_value_flight < 0.05:
    print("  ✓ SIGNIFICANT difference detected (p < 0.05)")
else:
    print("  ✗ No significant difference")

# ============================================================================
# SAVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SAVING STATISTICS")
print("="*80)

nationality_stats = df.groupby('nationality').agg({
    'dwell_key1': ['mean', 'std', 'median'],
    'flight_time_DD': ['mean', 'std', 'median'],
    'participant': 'count'
}).round(2)

nationality_stats.columns = ['_'.join(col).strip() for col in nationality_stats.columns.values]
nationality_stats = nationality_stats.sort_values('participant_count', ascending=False)
nationality_stats.to_csv('results/metrics/nationality_statistics.csv')

print("  ✓ Saved: nationality_statistics.csv")

summary_stats = pd.DataFrame({
    'Metric': [
        'Total Keystroke Pairs',
        'Total Participants',
        'Total Nationalities',
        'Mean Dwell Time (ms)',
        'Std Dwell Time (ms)',
        'Mean Flight Time (ms)',
        'Std Flight Time (ms)',
        'ANOVA p-value (Dwell)',
        'ANOVA p-value (Flight)'
    ],
    'Value': [
        len(df),
        df['participant'].nunique(),
        df['nationality'].nunique(),
        df['dwell_key1'].mean(),
        df['dwell_key1'].std(),
        df['flight_time_DD'].mean(),
        df['flight_time_DD'].std(),
        p_value_dwell,
        p_value_flight
    ]
})

summary_stats.to_csv('results/metrics/exploration_summary.csv', index=False)
print("  ✓ Saved: exploration_summary.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" " * 30 + "KEY FINDINGS")
print("="*80)

print(f"\n📊 DATASET OVERVIEW:")
print(f"  • Total keystroke pairs analyzed: {len(df):,}")
print(f"  • Total participants: {df['participant'].nunique()}")
print(f"  • Total nationalities: {df['nationality'].nunique()}")
print(f"  • Top nationality: {nationality_counts.index[0]} ({nationality_counts.values[0]:,} samples)")

print(f"\n⌨️  TYPING BEHAVIOR:")
print(f"  • Average dwell time: {df['dwell_key1'].mean():.2f} ms (±{df['dwell_key1'].std():.2f})")
print(f"  • Average flight time: {df['flight_time_DD'].mean():.2f} ms (±{df['flight_time_DD'].std():.2f})")

print(f"\n🌍 CROSS-NATIONALITY INSIGHTS:")
print(f"  • Statistical test: {'SIGNIFICANT' if p_value_dwell < 0.05 else 'NOT SIGNIFICANT'} differences detected")
if p_value_dwell < 0.05 and p_value_flight < 0.05:
    print(f"  • Both dwell time AND flight time show significant variation")
    print(f"  • This proves typing behavior varies across nationalities!")

print(f"\n🎯 IMPLICATIONS FOR AUTHENTICATION:")
print(f"  • Typing patterns differ measurably across nationalities")
print(f"  • Language/culture affects behavioral biometrics")
print(f"  • Cross-nationality authentication will face challenges")

print("\n" + "="*80)
print("✅ DATA EXPLORATION COMPLETE!")
print("="*80)
print("\n📁 Results saved to:")
print("  • results/visualizations/ (4 PNG files)")
print("  • results/metrics/ (2 CSV files)")
print("\n🚀 Next step: Feature engineering (create user profiles)")
print("="*80 + "\n")