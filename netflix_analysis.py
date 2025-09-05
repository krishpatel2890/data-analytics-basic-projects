# Netflix Data: Cleaning, Analysis, and Visualization (Beginner ML Project)

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configure visualization settings
sns.set_theme(style="whitegrid")   # instead of plt.style.use('seaborn')

# Step 2: Load the Dataset
data = pd.read_csv("netflix1.csv")
print("First 5 rows of dataset:\n", data.head(), "\n")

# Step 3: Data Cleaning
print("Missing values before cleaning:\n", data.isnull().sum(), "\n")

# Drop duplicates
data.drop_duplicates(inplace=True)

# Drop rows with missing critical information (only existing columns)
critical_cols = [col for col in ['director', 'cast', 'country'] if col in data.columns]
data.dropna(subset=critical_cols, inplace=True)

# Convert 'date_added' to datetime
data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')

# Show data types after conversion
print("Data types after cleaning:\n", data.dtypes, "\n")

# Step 4: Exploratory Data Analysis (EDA)

# 1. Content Type Distribution (Movies vs. TV Shows)
type_counts = data['type'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=type_counts.index, y=type_counts.values, palette='Set2')
plt.title('Distribution of Content by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# 2. Most Common Genres
data['genres'] = data['listed_in'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
all_genres = sum(data['genres'], [])
genre_counts = pd.Series(all_genres).value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='Set3')
plt.title('Most Common Genres on Netflix')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# 3. Content Added Over Time
data['year_added'] = data['date_added'].dt.year
data['month_added'] = data['date_added'].dt.month

plt.figure(figsize=(12, 6))
sns.countplot(x='year_added', data=data, palette='coolwarm')
plt.title('Content Added Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 4. Top 10 Directors with the Most Titles
top_directors = data['director'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='Blues_d')
plt.title('Top 10 Directors with the Most Titles')
plt.xlabel('Number of Titles')
plt.ylabel('Director')
plt.show()

# 5. Word Cloud of Movie Titles
movie_titles = data[data['type'] == 'Movie']['title'].dropna()
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(movie_titles))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Movie Titles")
plt.show()
