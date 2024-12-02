import pandas as pd

# Load the uploaded dataset to examine its structure and content
file_path = "/Users/yaqiduan/Desktop/sda/project/ecommerce_customer_data_large.csv"
ecommerce_data = pd.read_csv(file_path)

# Display basic information and the first few rows to understand the dataset
ecommerce_data.info(), ecommerce_data.head()

# Step 1: Data Cleaning and Preprocessing

# Check for missing values in the dataset
missing_values = ecommerce_data.isnull().sum()

# Convert 'Purchase Date' to datetime format
ecommerce_data['Purchase Date'] = pd.to_datetime(ecommerce_data['Purchase Date'], errors='coerce')

# Replace missing values in the 'Returns' column with 0
ecommerce_data['Returns'].fillna(0, inplace=True)

# Step 2: Summary Statistics
numerical_summary = ecommerce_data.describe()

# Step 3: Visualize distributions
import matplotlib.pyplot as plt

# Distribution of Total Purchase Amount
plt.figure(figsize=(8, 6))
plt.hist(ecommerce_data['Total Purchase Amount'], bins=50, alpha=0.7)
plt.title('Distribution of Total Purchase Amount', fontsize=14)
plt.xlabel('Total Purchase Amount', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Distribution of Customer Age
plt.figure(figsize=(8, 6))
plt.hist(ecommerce_data['Customer Age'], bins=20, alpha=0.7, color='orange')
plt.title('Distribution of Customer Age', fontsize=14)
plt.xlabel('Customer Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Step 4: Explore Product Category and Total Purchase Amount
category_summary = ecommerce_data.groupby('Product Category')['Total Purchase Amount'].mean().sort_values()

# Plot the mean total purchase amount for each product category
plt.figure(figsize=(10, 6))
category_summary.plot(kind='bar', color='skyblue')
plt.title('Average Total Purchase Amount by Product Category', fontsize=14)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Average Total Purchase Amount', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Step 5: Analyze Customer Age and Total Purchase Amount
age_summary = ecommerce_data.groupby('Customer Age')['Total Purchase Amount'].mean()

# Plot relationship between Customer Age and Average Total Purchase Amount
plt.figure(figsize=(12, 6))
plt.plot(age_summary.index, age_summary.values, marker='o', linestyle='-', color='purple')
plt.title('Average Total Purchase Amount by Customer Age', fontsize=14)
plt.xlabel('Customer Age', fontsize=12)
plt.ylabel('Average Total Purchase Amount', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# Step 6: Combine Customer Age and Product Category for analysis
age_category_summary = ecommerce_data.groupby(['Customer Age', 'Product Category'])['Total Purchase Amount'].mean().unstack()

# Heatmap visualization
plt.figure(figsize=(14, 8))
plt.imshow(age_category_summary.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Average Total Purchase Amount')
plt.title('Impact of Customer Age and Product Category on Total Purchase Amount', fontsize=14)
plt.xlabel('Customer Age', fontsize=12)
plt.ylabel('Product Category', fontsize=12)
plt.xticks(range(0, len(age_category_summary.index), 5), age_category_summary.index[::5], rotation=45)
plt.yticks(range(len(age_category_summary.columns)), age_category_summary.columns)
plt.show()