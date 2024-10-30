import kagglehub
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Basic Data Exploration
# Download latest version
path = kagglehub.dataset_download("shivamb/netflix-shows")
print("Path to dataset files:", path)

# Confirm dataset location and path
for dirname, _, filenames in os.walk('/Users/chanbormey/.cache/kagglehub/datasets/shivamb/netflix-shows/versions/5'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        print(f"Dataset file path: {file_path}")



#  2. Data Cleaning and Handling Missing Values
# Load the dataset
netflix_data = pd.read_csv(file_path)

# Preview the first few rows
print(netflix_data.head())
# Display the number of rows and columns
print(f"Rows: {netflix_data.shape[0]}, Columns: {netflix_data.shape[1]}")
# Display data types of each column
print(netflix_data.dtypes)
# Summary of numerical columns
print(netflix_data.describe())


# Identify the missing values
print("Missing values per column:")
print(netflix_data.isnull().sum())

# Drop 'show_id' column if it exists
if 'show_id' in netflix_data.columns:
    netflix_data.drop(columns=['show_id'], inplace=True)

# Fill missing values with "Unknown" if columns exist
if 'director' in netflix_data.columns:
    netflix_data.fillna({'director': "Unknown"}, inplace=True)
if 'cast' in netflix_data.columns:
    netflix_data.fillna({'cast': "Unknown"}, inplace=True)
if 'country' in netflix_data.columns:
    netflix_data.fillna({'country': "Unknown"}, inplace=True)
if 'duration' in netflix_data.columns:
    netflix_data.fillna({'duration': "Unknown"}, inplace=True)

# Drop rows with missing values in 'date_added' and 'rating' if columns exist
required_columns = ['date_added', 'rating']
existing_columns = [col for col in required_columns if col in netflix_data.columns]
if existing_columns:
    netflix_data.dropna(subset=existing_columns, inplace=True)

# Recheck for missing values to confirm cleaning
print("Missing values after cleaning:")
print(netflix_data.isnull().sum())


# 3. Analyze and Manipulate Data with Pandas
# Calculate the number of Movies and TV Shows
content_counts = netflix_data['type'].value_counts()
print("Count of Movies vs. TV Shows:")
print(content_counts)

# Identify the top 5 directors with the most titles on Netflix
top_directors = netflix_data['director'].value_counts().head(5)
print("Top 5 Directors with the Most Titles:")
print(top_directors)

# Find the average duration of movies in minutes
# Remove " min" from duration strings and convert to numerical
netflix_data['duration'] = netflix_data['duration'].astype(str)  # Convert to string
netflix_data['duration'] = netflix_data['duration'].str.extract('(\d+)').astype(float)
avg_movie_duration = netflix_data[netflix_data['type'] == 'Movie']['duration'].mean()
print(f"Average movie duration: {avg_movie_duration:.2f} minutes")


# 4. Data Analysis Using NumPy
# Use NumPy to calculate the mean, median, and standard deviation of movie durations
movie_durations = netflix_data[netflix_data['type'] == 'Movie']['duration'].dropna().values
mean_duration = np.mean(movie_durations)
median_duration = np.median(movie_durations)
std_dev_duration = np.std(movie_durations)

print(f"Mean duration: {mean_duration:.2f} minutes")
print(f"Median duration: {median_duration:.2f} minutes")
print(f"Standard deviation of duration: {std_dev_duration:.2f} minutes")

# Use NumPy arrays to analyze release years
release_years = netflix_data['release_year'].dropna().values  # Ensure no NaN values
mean_year = np.mean(release_years)
median_year = np.median(release_years)
std_dev_year = np.std(release_years)

print(f"Mean release year: {mean_year:.0f}")
print(f"Median release year: {median_year:.0f}")
print(f"Standard deviation of release years: {std_dev_year:.2f}")


# 5. Visualization
sns.set_style(style="whitegrid")

# Figure 1: Count of Movies vs. TV Shows
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='type', data=netflix_data, palette='mako', hue='type', ax=ax, legend=False)
plt.title("Count of Movies vs. TV Shows")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# Figure 2: Distribution of Release Years
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(netflix_data['release_year'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Release Years")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()

# Figure 3: Average Movie Duration Distribution
fig, ax = plt.subplots(figsize=(10, 6))
movie_durations_series = netflix_data[netflix_data['type'] == 'Movie']['duration']
sns.histplot(movie_durations_series, bins=30, kde=True, color='salmon')
plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (minutes)")
plt.ylabel("Count")
plt.show()
