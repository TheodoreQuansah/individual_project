import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats import chi2_contingency


def smoking_vs_heartdisease(df):
    # Bar chart for Smoking vs. Heart Disease
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='smoking', hue='heart_disease')
    plt.title('Smoking vs. Heart Disease')
    plt.xlabel('Smoking')
    plt.ylabel('Count')
    plt.show()


def stats_test(df, x):
    # Create a contingency table
    contingency_table = pd.crosstab(df[x], df['heart_disease'])
    
    # Perform the chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Print the test statistics and p-value
    print(f"Chi-squared value: {chi2}")
    print(f"P-value: {p}")
    
    # Determine the significance level (alpha)
    alpha = 0.05
    
    # Check if the p-value is less than alpha to make a decision
    if p < alpha:
        print(f"Reject the null hypothesis: There is a significant association between {x} and heart disease.")
    else:
        print(f"Fail to reject the null hypothesis: There is no significant association between {x} and heart disease.")


def BMI_vs_heartdisease(df):
    # Box plot for BMI vs. Heart Disease
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='heart_disease', y='b_m_i')
    plt.title('BMI vs. Heart Disease')
    plt.xlabel('Heart Disease')
    plt.ylabel('BMI')
    plt.show()


def pa_vs_heartdisease(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    sns.countplot(data=df, x="physical_activity", hue="heart_disease")
    plt.title("Physical Activity vs. Heart Disease")
    plt.xlabel("Physical Activity")
    plt.ylabel("Count")
    
    plt.show()


def pa_stats_test(df):
    # Contingency table of physical activity vs. heart disease
    contingency_table = pd.crosstab(df['physical_activity'], df['heart_disease'])
    
    # Performing the chi-squared test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Printing the results
    print(f"Chi-squared statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of freedom: {dof}")
    print("Expected frequencies table:")
    print(expected)



def age_vs_heartdisease(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Define the custom order for age categories in reverse
    custom_order = ['80 or older', '75-79', '70-74', '65-69', '60-64', '55-59', '50-54', '45-49', '40-44', '35-39', '30-34', '25-29', '18-24']
    
    sns.countplot(data=df, x="age_category", hue="heart_disease", order=custom_order)
    plt.title("Age Category vs. Heart Disease")
    plt.xlabel("Age Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.show()



def sleeptime_boxplot(df):    
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(data=df, x="heart_disease", y="sleep_time")
    plt.title("Sleep Time vs. Heart Disease")
    plt.xlabel("Heart Disease")
    plt.ylabel("Sleep Time")
    
    plt.show()


def sleeptime_vs_heartdisease(df):
    # Create a contingency table
    contingency_table = pd.crosstab(df['sleep_time'], df['heart_disease'])
    
    # Perform the chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Output the test statistics and p-value
    print("Chi-squared statistic:", chi2)
    print("Degrees of freedom:", dof)
    print("p-value:", p)