#!/usr/bin/env python
# coding: utf-8

# ### Function to create a coefficient plot after using MrDeep

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Parameters:
# - df1, df2: pandas DataFrames containing the variables, point estimates, and standard errors.
# - var_column: String name of the column containing the variable names.
# - est_column: String name of the column containing the point estimates.
# - se_column: String name of the column containing the standard errors.
# - difference: Boolean, if True, plot the difference between point estimates of df1 and df2.
# - conf_int_multiplier: Multiplier for the confidence interval (default is 1.96 for 95% CI).
# - color1, color2: Colors for the plots of df1 and df2, respectively.

# In[12]:


def plot_model_comparisons(df1, df2, var_column, est_column, se_column, difference=False, conf_int_multiplier=1.96, color1='blue', color2='red'):
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()

    # Generate a range of positions for the y-axis based on the number of variables
    y_positions = np.arange(len(df1[var_column]))

    if difference:
        # Calculate the differences in point estimates between the two models
        differences = df1[est_column].values - df2[est_column].values
        # Calculate the combined standard error for the differences
        se_differences = np.sqrt(df1[se_column].values**2 + df2[se_column].values**2)
        # Plot the differences with error bars representing the confidence intervals
        ax.errorbar(differences, y_positions, xerr=se_differences * conf_int_multiplier, fmt='o', color='black', label='Difference')
    else:
        # Plot the point estimates and confidence intervals for the first model
        ax.errorbar(df1[est_column].values, y_positions, xerr=df1[se_column].values * conf_int_multiplier, fmt='o', color=color1, label='Model 1')
        # Plot the point estimates and confidence intervals for the second model
        ax.errorbar(df2[est_column].values, y_positions, xerr=df2[se_column].values * conf_int_multiplier, fmt='o', color=color2, label='Model 2')

    # Set the y-axis ticks and labels based on the variable names
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df1[var_column].values)
    # Label the x-axis
    ax.set_xlabel('Point Estimate')
    # Draw a vertical line at x=0 for reference
    ax.axvline(x=0, color='black', linestyle='--')
    # Add a legend to the plot
    ax.legend()
    # Adjust layout for tight fitting
    plt.tight_layout()
    # Display the plot
    plt.show()

 


# ### Cook some data for an example plot

# In[13]:


# Sample data for Model 1 (Before MRDeep)
np.random.seed(0)  # For reproducibility
variables = ['Gender', 'PolicyApproval', 'Age', 'Income']
point_estimates1 = np.random.normal(loc=0, scale=1, size=len(variables))
standard_errors1 = np.random.uniform(low=0.1, high=0.5, size=len(variables))

df1 = pd.DataFrame({
    'Variable': variables,
    'PointEstimate': point_estimates1,
    'SE': standard_errors1
})

# Sample data for Model 2 (After MRDeep)
point_estimates2 = point_estimates1 + np.random.normal(loc=0, scale=0.5, size=len(variables))  # Slight changes
standard_errors2 = np.random.uniform(low=0.1, high=0.5, size=len(variables))

df2 = pd.DataFrame({
    'Variable': variables,
    'PointEstimate': point_estimates2,
    'SE': standard_errors2
})


print(df1.head())
print(df2.head())


# In[14]:


df1, df2 = create_sample_data()
plot_model_comparisons(df1, df2, 'Variable', 'PointEstimate', 'SE', difference=False)


# In[15]:


df1, df2 = create_sample_data()
plot_model_comparisons(df1, df2, 'Variable', 'PointEstimate', 'SE', difference=True)

