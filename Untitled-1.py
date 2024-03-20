# %%
import pandas as pd

# %%
ds = pd.read_csv('dataset_1655023.csv')
ds.head()

# %%
# Check for and remove missing data
ds.dropna(inplace=True)

# Calculate the total number of transactions
total_transactions = len(ds)

# Calculate the number of transactions with more than 2 items sold
transactions_more_than_2_items = len(ds[ds['transaction_qty'] > 2])

# Calculate the probability that a transaction contains more than 2 items
probability_more_than_2_items = transactions_more_than_2_items / total_transactions

# Print the results
print(f"Total transactions: {total_transactions}")
print(f"Transactions with more than 2 items: {transactions_more_than_2_items}")
print(f"Probability of a transaction having more than 2 items: {probability_more_than_2_items:.2%}")



# %%
descriptive_stats = ds['unit_price'].describe()
print(descriptive_stats)


# %%
skewness = ds['unit_price'].skew()
print(f"Skewness: {skewness}")


# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(x='unit_price', y='product_type', data=ds)
plt.title('Unit Price Distribution by Product Type')
plt.xlabel('Unit Price')
plt.ylabel('Product Type')
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Check for and remove missing data
ds.dropna(inplace=True)

# Descriptive statistics for unit_price
descriptive_stats = ds['unit_price'].describe()

# Calculate skewness
skewness = ds['unit_price'].skew()

# Boxplot for unit prices by product type
plt.figure(figsize=(10, 6))
sns.boxplot(x='unit_price', y='product_type', data=ds)
plt.title('Unit Price Distribution by Product Type')
plt.xlabel('Unit Price')
plt.ylabel('Product Type')
plt.tight_layout()

# Histogram of unit prices
plt.figure(figsize=(10, 6))
sns.histplot(ds['unit_price'], kde=True)
plt.title('Histogram of Unit Prices')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.tight_layout()

descriptive_stats, skewness


# %%
from scipy import stats

n = len(ds)  # Sample size
confidence_level = 0.95
degrees_of_freedom = n - 1
t_crit = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

margin_of_error = t_crit * (std_revenue / (n ** 0.5))
confidence_interval = (mean_revenue - margin_of_error, mean_revenue + margin_of_error)
print(f"95% confidence interval for the mean revenue per transaction: {confidence_interval}")

# %%
from scipy import stats

# Check for and remove missing data
ds.dropna(inplace=True)

ds['revenue'] = ds['transaction_qty'] * ds['unit_price']

#descriptive statistics
revenue_descriptive_stats = ds['revenue'].describe()
print(revenue_descriptive_stats)


# %%
#mean and std
mean_revenue = ds['revenue'].mean()
std_revenue = ds['revenue'].std()

# %%
n = len(ds)  # Sample size
confidence_level = 0.95
degrees_of_freedom = n - 1
t_crit = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

margin_of_error = t_crit * (std_revenue / (n ** 0.5))
confidence_interval = (mean_revenue - margin_of_error, mean_revenue + margin_of_error)
print(f"95% confidence interval for the mean revenue per transaction: {confidence_interval}")


