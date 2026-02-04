""" 
Guided Project Credit Card Customer Segmentation

Our analysis has identified 7 distinct customer profiles within the dataset, differentiated by factors such as gender, age, marital status, income, and spending patterns.

By deeply understanding these unique customer segments, we can develop tailored strategies to better serve their needs and preferences.

This granular customer intelligence will enable more effective acquisition, retention and relationship-building efforts, helping us adapt to evolving market dynamics over time.

Moving forward, continued monitoring and refinement of these customer clusters will be crucial to sustaining a competitive edge.

The goal of this project is to segment the customer data into distinct, well-defined groups, with clear identifying features for each segment. By the end of the project, we will have a comprehensive understanding of the different customer personas, enabling more targeted and effective marketing, product development, and customer service strategies.

##The Data
The following is some boilerplate code that loads the required libraries, reads the data and displays some initial information about it:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('customer_segmentation.csv')
data.shape
data.info
data.head()

"""The dataset has records of 10,127 customers, each with 13 features (plus customer_id, which isn't analyzed). There are no missing features from any of the records.

The features are as follows:

1.customer_id: unique identifier for each customer.

2.age: customer age in years.

3.gender: customer gender (M or F).

4.dependent_count: number of dependents of each customer.

5.education_level: level of education (High School, Graduate, etc.).

6.marital_status: marital status (Single, Married, etc.).

7.estimated_income: the estimated income for the customer projected by the data science team.

8.months_on_book: time as a customer in months.

9.total_relationship_count: number of times the customer contacted the company.

10.months_inactive_12_mon: number of months the customer did not use the credit card in the last 12 months.

11.credit_limit: customer's credit limit.

12.total_trans_amount: the overall amount of money spent on the card by the customer.

13.total_trans_count: the overall number of times the customer used the card.

14.avg_utilization_ratio: daily average utilization ratio.

Of the features, 3 are categorical (gender, education_level and marital_status) and the other 10 are numeric."""

#Familiarize ourselves with the dataset. Answer questions such as:How big is the dataset? How many columns does it have? Do we see any particular column that doesn't use the analysis? What's the type of data contained in each column? Are there many categorical variables? How are we dealing with them? Are there any missing values? Look at the correlation between the columns and explain what we see. Plot the distribution of each numeric column and comment on the results.

categorical_features = ['gender', 'education_level', 'marital_status']

for feature in categorical_features:
    print(data[feature].value_counts(), end='\n\n')

"""The categorical variables appear to be clean, with no typos or irrelevant entries.

The gender distribution is relatively balanced, although there is a slightly higher proportion of male patients (53%) compared to female patients (47%).

Next, we examine the correlations between the different features. Since we are not comparing them against a specific target variable at this stage, the goal is simply to identify noteworthy relationships and patterns within the dataset."""

sns.heatmap(round(data.drop(categorical_features + ['customer_id'], axis=1).corr(), 2), cmap='coolwarm', annot=True)
plt.figure(figsize=(12,8))
plt.tight_layout()
plt.show()

"""Most features exhibit only weak correlations. However, a few relationships stand out as potentially meaningful:

total_trans_amount and total_trans_count show a strong positive correlation (0.81). This is expected, as customers who use their card more frequently tend to accumulate higher transaction amounts.

months_on_book and age are also strongly correlated, which is reasonable: older customers have had more time to hold the card.

credit_limit shows a moderate positive correlation (0.52) with estimated_income, suggesting that customers with higher income tend to receive higher credit limits. It is also moderately negatively correlated (-0.48) with avg_utilization_ratio, indicating that customers with higher credit limits tend to use a smaller proportion of their available credit.

Next, we examine the distributions of the numerical features to better understand their spread, skewness, and potential outliers."""

data.drop(categorical_features + ['customer_id'], axis=1).hist(figsize=(12, 10))

plt.tight_layout()
plt.show()

data.drop('customer_id', axis=1).describe()

"""Based on the histogram plots and the summary table, several observations can be made:

    There are no obvious or extreme outliers in the numerical features.

    estimated_income, credit_limit, total_trans_amount, and avg_utilization_ratio exhibit right‑skewed distributions, indicating that most customers fall on the lower end of these ranges, with fewer individuals showing very high values.

    In contrast, age, months_on_book, and total_trans_count display distributions that are closer to normal, suggesting a more even spread of values around their respective means.
    
    ## Feature Engineering"""

#Create a copy of the original DataFrame. Call it customers_modif, for instance.
data_modif = data.copy()

#Replace the values in the gender column with 1 for "M" and 0 for "F". Use the map() or replace() methods, a lambda function with apply(), or even the numpy.where() function to perform this task.
data_modif['gender'] = data_modif['gender'].map({'M':1, 'F':0})

#Replace the values in the education_level column in the order below. Choose how to execute this replacement. Uneducated - 0; High School - 1; College - 2; Graduate - 3; Post-Graduate - 4; Doctorate - 5
education_level_to_int = {
    'Uneducated' : 0,
    'High School' : 1,
    'College' : 2,
    'Graduate' : 3,
    'Post-Graduate' : 4,
    'Doctorate' : 5
}

data_modif['education_level'] = data_modif['education_level'].map(education_level_to_int)

"""Regarding marital_status, there is no meaningful or objective way to impose an ordinal structure on the different categories. For this reason, we apply one‑hot encoding to transform the marital status values into a set of dummy variables, allowing the model to process each category independently without implying any inherent order."""

#Use the pandas.get_dummies() function to create a dataframe containing dummy variables for the unique values in the marital_status. Combine the new DataFrame with the original.Drop the original marital_status column.
data_modif = pd.get_dummies(data_modif, columns=['marital_status'], dtype=int, drop_first=True)
data_modif

"""## Scaling the Data
To ensure that the data can be processed efficiently and that no feature disproportionately influences the model due to differences in scale, it is important to standardize the numerical variables so that they are all on a comparable range."""

#Create a new DataFrame without the column customer_id
clean_data = data_modif.drop('customer_id', axis=1).copy()

#Instantiate an object from the scikit-learn's StandardScaler() class and fit the new DataFrame. Use the transform method to scale the data. Assign it to a different variable and print it. As the outcome of the scaler is an array, we can transform it back to a DataFrame using pd.DataFrame().
scaler = StandardScaler()
scaler.fit(clean_data)
scaled_data = pd.DataFrame(scaler.transform(clean_data), columns=clean_data.columns)
scaled_data


"""
## Choosing K
The dataset is now ready for machine learning. The next step is to determine the optimal number of clusters for customer segmentation using the inertia metric. To achieve this, we will train multiple K‑Means models with different values of k and compute the inertia for each one. This will allow us to plot the Elbow Curve and identify the point at which increasing the number of clusters no longer provides substantial improvement. Once the optimal number of clusters has been selected, we can proceed with the final clustering process."""

#Create an empty list to store the inertia from every model
inertias = []
#Use a for loop to the following process for different numbers of K. Loop in a range from 1 to 10, for example.
for i in range(1, 11):
    #Instantiate a Kmeans object setting n_clusters=k.
    model = KMeans(n_clusters = i)
    #Use fit_predict() to create clusters.
    model.fit_predict(scaled_data)
    #Append the inertia_ attribute of the model to the empty list.
    inertias.append(model.inertia_)

#Use the list to plot the elbow curve. Decide how many clusters to use and explain this decision.
plt.plot(range(1, 11), inertias, marker='o')
plt.title("Inertia vs. No. of Clusters")
plt.xticks(ticks=range(1, 11), labels=range(1, 11))
plt.show()

"""Since the Elbow Curve does not reveal a clear inflection point, we can instead examine the percentage decrease in inertia between successive values of k. This allows us to identify where the marginal improvement begins to diminish and helps us select a reasonable candidate for the number of clusters."""

percent_decrease = [(1 - (inertias[i] / inertias[i - 1])) * 100 for i in range(1, 10)]

ax = plt.bar(range(2, 11), percent_decrease)
plt.bar_label(ax, fmt='%.1f%%')
plt.title("Percentual decrease in Inertia from Previous K")
plt.xticks(ticks=range(2, 11), labels=range(2, 11))
plt.show()

"""The differences in the percentage decrease of inertia between successive cluster counts are relatively small, indicating that there is no clearly dominant elbow point. However, the drop in inertia from 5 to 6 clusters is noticeably larger than the surrounding changes. This sharper decline suggests that the 5‑cluster solution offers a strong balance between model complexity and explanatory power, making it a reasonable choice for the final segmentation."""

#Instantiate a new Kmeans object, but this time use the decided number of clusters as K.
model = KMeans(n_clusters = 5)
#fit_predict the data and print the outcome.
clusters = model.fit_predict(scaled_data)

"""
## Analyzing Results
All that remains is to analyze the clustering results. To understand what each cluster represents, we need to examine how the variables used in the segmentation differ across clusters. By identifying the key characteristics that define each group, we can interpret the underlying customer profiles and assess how these insights may inform business decisions and marketing strategies. This analysis will allow us to determine how each cluster behaves, what distinguishes it from the others, and how the company can tailor its approach to better serve each customer segment."""

#Create a new column called CLUSTER in the original customers DataFrame. This column should contain the cluster assigned to each customer by the algorithm.
data['CLUSTER'] = clusters + 1
#Group each numeric variable by the CLUSTER column and plot a bar chart. Analyze the clusters' characteristics regarding each variable. Explain the conclusions.
ax = plt.bar(range(1, 6), data['CLUSTER'].value_counts().sort_index())
plt.title("Number of customers per cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of customers")
plt.bar_label(ax)
plt.show()

"""We observe two larger clusters (Clusters 1 and 2) and four smaller ones. Among these, Cluster 2 is the smallest, containing only 953 customers."""
plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.2)
plt.suptitle("Averages of Numerical Features by Cluster", fontsize=18, y=1.0)

numeric_features = data.drop(categorical_features + ['customer_id', 'CLUSTER'], axis=1).columns
    
# set number of columns
ncols = 3
# calculate number of rows
nrows = len(numeric_features) // ncols + (len(numeric_features) % ncols > 0)

# loop through the length of numeric_features and keep track of index
for i, feature in enumerate(numeric_features):
    # add a new subplot iteratively using nrows and cols
    ax = plt.subplot(nrows, ncols, i + 1)

    # group the data by clusters and plot the feature on the new subplot axis
    data.groupby('CLUSTER')[feature].mean().plot.bar(ax=ax, figsize=(12, 10), color=sns.color_palette('muted'))

    # chart formatting
    ax.set_title(feature)
    ax.set_xlabel("")

plt.tight_layout()
plt.show()

"""We can now examine the numerical features that exhibit stronger correlations. To do this, we will generate scatter plots for each pair of highly correlated variables and colour the data points according to their assigned cluster. This visual approach allows us to identify potential patterns, relationships, or separations between clusters that may not be immediately apparent from summary statistics alone."""

#Create a scatter plot with different colors for each cluster of pairs of variables with a high correlation. Use seaborn.scatterplot() function with the hue parameter.
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

sns.scatterplot(data=data, y='total_trans_amount', x='total_trans_count', hue='CLUSTER', palette='muted', alpha=0.5, ax=ax1)
sns.scatterplot(data=data, y='months_on_book', x='age', hue='CLUSTER', palette='muted', alpha=0.5, legend=False, ax=ax2)
sns.scatterplot(data=data, y='credit_limit', x='estimated_income', hue='CLUSTER', palette='muted', alpha=0.5, legend=False, ax=ax3)
sns.scatterplot(data=data, y='credit_limit', x='avg_utilization_ratio', hue='CLUSTER', palette='muted', alpha=0.5, ax=ax4)
plt.show()

"""From these initial observations, we can already identify a few meaningful patterns:

    Cluster 4 consists primarily of customers with a high number of transactions.

    Cluster 2 is largely composed of older customers.

    Cluster 5 includes customers who tend to fall within the lower end of the credit‑limit range.

Next, we turn to the categorical variables. By plotting the percentage distribution of each categorical feature across clusters, we can identify additional behavioural or demographic patterns that further differentiate the customer segments."""

#Use the [pandas.crosstab()] function to calculate the percentual distribution of each variable per cluster for the categorical columns. Use this data to plot a stacked bar chart.

n = len(categorical_features) 
rows = 1 
cols = n 
plt.figure(figsize=(6 * cols, 6)) 
for i, col in enumerate(categorical_features): 
    ax = plt.subplot(rows, cols, i + 1) 
    plot_df = pd.crosstab( index=data['CLUSTER'], columns=data[col], values=data[col], aggfunc='size', normalize='index' ) 
    
    plot_df.plot.bar(stacked=True, ax=ax, color=sns.color_palette('muted')) 
    ax.set_title(f"% {col.title()} per cluster") 
    ax.set_ylim(0, 1.4) 
    ax.legend(frameon=False) 
    ax.xaxis.grid(False)
    
plt.tight_layout()
plt.show()

"""Based on the data visualizations, several patterns emerge across the customer clusters:

    Gender distribution: Clusters 4 and 7 are predominantly composed of male customers, whereas Cluster 5 contains a higher proportion of female customers.

    Marital status: The distribution of marital status varies considerably between clusters. Clusters 4 and 5 consist mainly of married customers, while Clusters 1 and 6 are primarily composed of single customers. Cluster 2, however, contains a substantial number of customers for whom marital status information is missing.

    Education level: The distribution of education levels remains relatively stable across clusters, suggesting that education does not play a major role in differentiating the customer segments."""