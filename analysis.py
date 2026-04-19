# Hero FinCorp - Data Analysis Assignment
# Name: Your Name
# Roll No: Your Roll No
# Date: April 2026

# NOTE:
# Use "data/" paths when running locally or from GitHub
# If using Google Colab, remove "data/" from file paths

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================
# LOAD DATA (GITHUB PATHS)
# ==============================
apps = pd.read_csv("data/applications.csv")
branches = pd.read_csv("data/branches.csv")
cust = pd.read_csv("data/customers.csv")
deflt = pd.read_csv("data/defaults.csv")
loans = pd.read_csv("data/loans.csv")
trans = pd.read_csv("data/transactions.csv")

print("datasets loaded")
print("apps:", apps.shape)
print("branches:", branches.shape)
print("customers:", cust.shape)
print("defaults:", deflt.shape)
print("loans:", loans.shape)
print("transactions:", trans.shape)


# ==============================
# TASK 1 - DATA CLEANING
# ==============================
print("\nTask 1 - Data Cleaning")

for name, df in [("apps", apps), ("branches", branches), ("cust", cust),
                 ("deflt", deflt), ("loans", loans), ("trans", trans)]:
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    print(f"\n{name}:")
    print(mv if len(mv) > 0 else "no missing values")

print("\nDuplicates:")
print("apps:", apps.duplicated(subset="Application_ID").sum())
print("customers:", cust.duplicated(subset="Customer_ID").sum())
print("loans:", loans.duplicated(subset="Loan_ID").sum())
print("defaults:", deflt.duplicated(subset="Default_ID").sum())
print("transactions:", trans.duplicated(subset="Transaction_ID").sum())

# Date conversion
apps["Application_Date"] = pd.to_datetime(apps["Application_Date"], errors="coerce")
apps["Approval_Date"] = pd.to_datetime(apps["Approval_Date"], errors="coerce")
loans["Disbursal_Date"] = pd.to_datetime(loans["Disbursal_Date"], errors="coerce")
loans["Repayment_Start_Date"] = pd.to_datetime(loans["Repayment_Start_Date"], errors="coerce")
loans["Repayment_End_Date"] = pd.to_datetime(loans["Repayment_End_Date"], errors="coerce")
deflt["Default_Date"] = pd.to_datetime(deflt["Default_Date"], errors="coerce")
trans["Transaction_Date"] = pd.to_datetime(trans["Transaction_Date"], errors="coerce")

# Fill missing
apps["Rejection_Reason"] = apps["Rejection_Reason"].fillna("N/A")
deflt["Legal_Action"] = deflt["Legal_Action"].fillna("No")

print("Task 1 done")


# ==============================
# TASK 2 - DESCRIPTIVE
# ==============================
print("\nTask 2 - Descriptive")

print(loans[["Loan_Amount","EMI_Amount","Overdue_Amount"]].describe())
print(cust[["Credit_Score","Annual_Income"]].describe())

plt.hist(loans["Loan_Amount"], bins=30)
plt.title("Loan Amount Distribution")
plt.savefig("loan_distribution.png")
plt.close()

plt.hist(cust["Credit_Score"], bins=30)
plt.title("Credit Score Distribution")
plt.savefig("credit_score_distribution.png")
plt.close()

print("Task 2 done")


# ==============================
# TASK 3 - DEFAULT
# ==============================
print("\nTask 3 - Default")

loans["Default_Flag"] = loans["Loan_ID"].isin(deflt["Loan_ID"]).astype(int)

df = loans.merge(cust, on="Customer_ID", how="left")

default_rate = df["Default_Flag"].mean()
print("Default Rate:", default_rate)

df["Default_Flag"].value_counts().plot(kind='bar')
plt.title("Default vs Non Default")
plt.savefig("default_chart.png")
plt.close()

print("Task 3 done")


# ==============================
# TASK 4 - CORRELATION
# ==============================
print("\nTask 4 - Correlation")

cols = ["Loan_Amount","Interest_Rate","Credit_Score","Default_Flag"]
corr = df[cols].corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("Task 4 done")


# ==============================
# TASK 5 - BRANCH
# ==============================
print("\nTask 5 - Branch")

branches["Default_Rate"] = branches["Delinquent_Loans"] / branches["Total_Active_Loans"]

branches.groupby("Region")["Default_Rate"].mean().plot(kind='bar')
plt.title("Default Rate by Region")
plt.savefig("region_default_rate.png")
plt.close()

print("Task 5 done")


# ==============================
# TASK 6 - SEGMENTATION
# ==============================
print("\nTask 6 - Segmentation")

def segment(x):
    if x >= 750: return "High"
    elif x < 600: return "Low"
    else: return "Medium"

df["Segment"] = df["Credit_Score"].apply(segment)
print(df["Segment"].value_counts())


# ==============================
# TASK 8 - EMI
# ==============================
print("\nTask 8 - EMI")

df["EMI"] = (df["Loan_Amount"] * (df["Interest_Rate"]/100)) / 12

df["EMI_Bucket"] = pd.cut(df["EMI"], bins=[0,10000,20000,50000,1000000],
                         labels=["Low","Medium","High","Very High"])

emi_default = df.groupby("EMI_Bucket")["Default_Flag"].mean()
emi_default.plot(kind='bar')
plt.title("Default by EMI")
plt.savefig("emi_analysis.png")
plt.close()

print("Task 8 done")


# ==============================
# TASK 11 - EFFICIENCY
# ==============================
print("\nTask 11 - Efficiency")

apps["Processing_Time"] = (apps["Approval_Date"] - apps["Application_Date"]).dt.days

apps["Processing_Time"].hist()
plt.title("Processing Time")
plt.savefig("processing_time.png")
plt.close()

print("Task 11 done")


# ==============================
# TASK 15 - BRANCH EFFICIENCY
# ==============================
print("\nTask 15 - Branch Efficiency")

top = branches.sort_values("Default_Rate", ascending=False).head(10)

top.set_index("Branch_ID")["Default_Rate"].plot(kind='bar')
plt.title("Top Risky Branches")
plt.savefig("branch_efficiency.png")
plt.close()

print("Task 15 done")


print("\nALL TASKS COMPLETED")
