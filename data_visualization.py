import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
data = sns.load_dataset('titanic')

# Display the first few rows
print("Dataset Overview:")
print(data.head())

# Set a style for Seaborn
sns.set_theme(style="whitegrid")

# 1. Bar Chart: Passenger Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='class', palette='viridis')
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 2. Scatter Plot: Age vs. Fare by Survival Status
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='age', y='fare', hue='survived', palette='coolwarm', alpha=0.7)
plt.title("Age vs Fare (Colored by Survival)")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived", loc='upper right')
plt.show()

# 3. Heatmap: Correlation Matrix of Numerical Features
plt.figure(figsize=(8, 6))
correlation_matrix = data.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# 4. Box Plot: Fare by Class and Survival Status
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='class', y='fare', hue='survived', palette='Set2')
plt.title("Fare Distribution by Class and Survival Status")
plt.xlabel("Class")
plt.ylabel("Fare")
plt.legend(title="Survived")
plt.show()

# Insights
print("\nInsights:")
print("1. The bar chart shows the majority of passengers were in Third Class.")
print("2. The scatter plot reveals that younger passengers often paid less, but fare varies significantly.")
print("3. The heatmap highlights that 'fare' has a weak correlation with survival, while 'age' has almost no correlation.")
print("4. The box plot indicates that First Class passengers generally paid higher fares, and survival rates were higher in First Class.")
