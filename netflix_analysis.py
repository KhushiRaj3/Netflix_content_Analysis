# Netflix Content Trends Analysis for Visual Studio Code (VS Code)
# ---------------------------------------------------------------
# This is a fully working Python script (not a Jupyter notebook)
# suitable for running directly in VS Code terminal.
# Make sure you have the dataset file `netflix_titles.csv` in the same folder.
# ---------------------------------------------------------------

# Install dependencies:
# pip install pandas matplotlib seaborn numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for saving figures
if not os.path.exists('figs'):
    os.makedirs('figs')

# Load dataset
DATA_PATH = 'netflix_titles.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Please place netflix_titles.csv in the same directory as this script.")

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

# Clean and preprocess data
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['type'] = df['type'].fillna('Unknown').str.strip()

def parse_duration(value):
    if pd.isna(value):
        return np.nan
    value = str(value)
    if 'min' in value:
        return int(value.replace('min', '').strip())
    elif 'Season' in value:
        return int(value.split()[0])
    return np.nan

df['duration_parsed'] = df['duration'].apply(parse_duration)

df['genres_list'] = df['listed_in'].fillna('Unknown').apply(lambda x: [g.strip() for g in x.split(',')])
df['country_primary'] = df['country'].fillna('Unknown').apply(lambda x: str(x).split(',')[0].strip())

# Detect Netflix originals using heuristic search
search_columns = ['title', 'description', 'cast', 'director']
def detect_original(row):
    text = ' '.join([str(row.get(col, '')).lower() for col in search_columns])
    return 'netflix' in text

df['is_original'] = df.apply(detect_original, axis=1)

# Year columns
df['year_added'] = df['date_added'].dt.year
df['release_year'] = df['release_year'].astype('Int64')

# --- Analysis 1: Growth trend (2010–2024) ---
years = list(range(2010, 2025))
added_per_year = df[df['year_added'].between(2010, 2024, inclusive='both')].groupby('year_added').size().reindex(years, fill_value=0)
originals_per_year = df[df['is_original'] & df['year_added'].between(2010, 2024, inclusive='both')].groupby('year_added').size().reindex(years, fill_value=0)

plt.figure(figsize=(10, 6))
plt.plot(years, added_per_year, marker='o', label='All Titles Added')
plt.plot(years, originals_per_year, marker='o', label='Netflix Originals')
plt.title('Netflix Content Growth (2010–2024)')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figs/content_growth.png')
plt.show()

# --- Analysis 2: Top Genres ---
genres_exploded = df.explode('genres_list')
top_genres = genres_exploded['genres_list'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title('Top 10 Genres on Netflix')
plt.xlabel('Number of Titles')
plt.tight_layout()
plt.savefig('figs/top_genres.png')
plt.show()

# --- Analysis 3: Genre Trends Over Time ---
top5 = top_genres.index[:5]
genre_time = genres_exploded[genres_exploded['genres_list'].isin(top5)].copy()
genre_time['year_plot'] = genre_time['year_added'].fillna(genre_time['release_year'])
genre_trend = genre_time.groupby(['year_plot', 'genres_list']).size().unstack(fill_value=0).reindex(years, fill_value=0)

plt.figure(figsize=(10, 6))
for genre in top5:
    plt.plot(years, genre_trend[genre], marker='o', label=genre)
plt.title('Genre Trends (2010–2024)')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figs/genre_trends.png')
plt.show()

# --- Analysis 4: Regional Insights ---
top_countries = df['country_primary'].value_counts().head(10).index
country_genre = genres_exploded[genres_exploded['country_primary'].isin(top_countries)]
country_pivot = country_genre.pivot_table(index='country_primary', columns='genres_list', values='show_id', aggfunc='count', fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(country_pivot[top_genres.index[:10]], cmap='YlGnBu', annot=True, fmt='d')
plt.title('Top Countries vs Popular Genres')
plt.tight_layout()
plt.savefig('figs/country_genre_heatmap.png')
plt.show()

# --- Analysis 5: Monthly Additions ---
if df['date_added'].notna().sum() > 0:
    df['month_added'] = df['date_added'].dt.month
    month_counts = df['month_added'].value_counts().sort_index()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    plt.figure(figsize=(10,6))
    plt.bar(months, [month_counts.get(i, 0) for i in range(1, 13)], color='coral')
    plt.title('Monthly Netflix Additions')
    plt.xlabel('Month')
    plt.ylabel('Number of Titles')
    plt.tight_layout()
    plt.savefig('figs/monthly_additions.png')
    plt.show()

# Save cleaned dataset
df.to_csv('netflix_titles_cleaned.csv', index=False)
print("Cleaned dataset saved as netflix_titles_cleaned.csv")
print("Figures saved in figs/ folder.")
