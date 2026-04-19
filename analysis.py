import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# loading all the datasets
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


# Task 1 - Data Cleaning and Preparation

# checking missing values in each dataset
print("\nMissing values:")
for name, df in [("apps", apps), ("branches", branches), ("cust", cust),
                 ("deflt", deflt), ("loans", loans), ("trans", trans)]:
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    print(f"\n{name}:")
    print(mv if len(mv) > 0 else "no missing values")

# checking duplicates
print("\nDuplicates:")
print("apps:", apps.duplicated(subset="Application_ID").sum())
print("customers:", cust.duplicated(subset="Customer_ID").sum())
print("loans:", loans.duplicated(subset="Loan_ID").sum())
print("defaults:", deflt.duplicated(subset="Default_ID").sum())
print("transactions:", trans.duplicated(subset="Transaction_ID").sum())

# converting date columns to proper format
apps["Application_Date"] = pd.to_datetime(apps["Application_Date"], errors="coerce")
apps["Approval_Date"] = pd.to_datetime(apps["Approval_Date"], errors="coerce")
loans["Disbursal_Date"] = pd.to_datetime(loans["Disbursal_Date"], errors="coerce")
loans["Repayment_Start_Date"] = pd.to_datetime(loans["Repayment_Start_Date"], errors="coerce")
loans["Repayment_End_Date"] = pd.to_datetime(loans["Repayment_End_Date"], errors="coerce")
deflt["Default_Date"] = pd.to_datetime(deflt["Default_Date"], errors="coerce")
trans["Transaction_Date"] = pd.to_datetime(trans["Transaction_Date"], errors="coerce")

# filling missing values
apps["Rejection_Reason"] = apps["Rejection_Reason"].fillna("N/A")
deflt["Legal_Action"] = deflt["Legal_Action"].fillna("No")

# checking outliers using IQR method
def check_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    outliers = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
    return outliers

print("\nOutliers:")
for col in ["Loan_Amount", "Interest_Rate", "Overdue_Amount", "EMI_Amount"]:
    print(f"  loans[{col}]:", check_outliers(loans[col].dropna()))
for col in ["Default_Amount", "Recovery_Amount"]:
    print(f"  deflt[{col}]:", check_outliers(deflt[col].dropna()))

print("\nTask 1 done")


# Task 2 - Descriptive Analysis

print("\nBasic stats for loan columns:")
print(loans[["Loan_Amount", "EMI_Amount", "Overdue_Amount"]].describe().round(2))

print("\nBasic stats for customer columns:")
print(cust[["Credit_Score", "Annual_Income"]].describe().round(2))

# distributions of loan amount, emi, credit score
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(loans["Loan_Amount"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Loan Amount Distribution")
axes[0].set_xlabel("Loan Amount")
axes[0].set_ylabel("Count")

axes[1].hist(loans["EMI_Amount"], bins=40, color="coral", edgecolor="white")
axes[1].set_title("EMI Amount Distribution")
axes[1].set_xlabel("EMI Amount")

axes[2].hist(cust["Credit_Score"], bins=40, color="green", edgecolor="white")
axes[2].set_title("Credit Score Distribution")
axes[2].set_xlabel("Credit Score")

plt.tight_layout()
plt.savefig("task2_distributions.png")
plt.close()

# merging loans with customer region info
loans_merged = loans.merge(apps[["Loan_ID", "Loan_Purpose"]], on="Loan_ID", how="left")
loans_merged = loans_merged.merge(cust[["Customer_ID", "Region"]], on="Customer_ID", how="left")

# regional loan disbursement
region_disb = loans_merged.groupby("Region")["Loan_Amount"].sum().sort_values(ascending=False)

# defaults by region
deflt_region = deflt.merge(cust[["Customer_ID", "Region"]], on="Customer_ID", how="left")
deflt_by_region = deflt_region.groupby("Region").size()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(region_disb.index, region_disb.values / 1e7, color="steelblue")
axes[0].set_title("Loan Disbursement by Region (Crores)")
axes[0].set_ylabel("Amount (Cr)")

axes[1].bar(deflt_by_region.index, deflt_by_region.values, color="salmon")
axes[1].set_title("Defaults by Region")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("task2_regional.png")
plt.close()

# monthly loan approval trend
apps["YearMonth"] = apps["Application_Date"].dt.to_period("M")
monthly = apps.groupby(["YearMonth", "Approval_Status"]).size().unstack(fill_value=0)
monthly.index = monthly.index.astype(str)

fig, ax = plt.subplots(figsize=(13, 5))
monthly.plot(ax=ax, marker="o", linewidth=1.2)
ax.set_title("Monthly Loan Applications")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("task2_monthly.png")
plt.close()

print("Task 2 done")


# Task 3 - Default Risk Analysis

# creating default flag column
loans["Default_Flag"] = loans["Loan_ID"].isin(deflt["Loan_ID"]).astype(int)

# merging with customer data for credit score
df = loans.merge(cust[["Customer_ID", "Credit_Score", "Annual_Income"]], on="Customer_ID", how="left")

# correlation between loan attributes and default
cols = ["Loan_Amount", "Interest_Rate", "EMI_Amount", "Overdue_Amount", "Credit_Score", "Default_Flag"]
corr = df[cols].dropna().corr()

print("\nCorrelation with Default_Flag:")
print(corr["Default_Flag"].drop("Default_Flag").sort_values(ascending=False))

# heatmap
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap - Loan Attributes")
plt.tight_layout()
plt.savefig("task3_heatmap.png")
plt.close()

# branch level correlation
branches["Default_Rate"] = branches["Delinquent_Loans"] / branches["Total_Active_Loans"]
branch_corr = branches[["Total_Active_Loans", "Delinquent_Loans",
                         "Loan_Disbursement_Amount", "Avg_Processing_Time", "Default_Rate"]].corr()

print("\nBranch level correlations:")
print(branch_corr.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(branch_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Branch Metrics Correlation")
plt.tight_layout()
plt.savefig("task3_branch_corr.png")
plt.close()

print("Task 3 done")


# Task 4 - Branch and Regional Performance

# ranking branches by disbursement
top_disb = branches.sort_values("Loan_Disbursement_Amount", ascending=False).head(15)
top_dr = branches.sort_values("Default_Rate", ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].barh(top_disb["Branch_Name"], top_disb["Loan_Disbursement_Amount"] / 1e7, color="steelblue")
axes[0].set_title("Top 15 Branches - Disbursement (Cr)")
axes[0].invert_yaxis()

axes[1].barh(top_dr["Branch_Name"], top_dr["Default_Rate"], color="salmon")
axes[1].set_title("Top 15 Branches - Default Rate")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("task4_branch_performance.png")
plt.close()

# regional comparison
region_perf = branches.groupby("Region").agg(
    Avg_Default_Rate=("Default_Rate", "mean"),
    Total_Disbursement=("Loan_Disbursement_Amount", "sum"),
    Avg_Processing_Time=("Avg_Processing_Time", "mean")
).reset_index()

print("\nRegional Performance:")
print(region_perf.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].bar(region_perf["Region"], region_perf["Total_Disbursement"] / 1e7, color="steelblue")
axes[0].set_title("Total Disbursement by Region (Cr)")

axes[1].bar(region_perf["Region"], region_perf["Avg_Default_Rate"], color="salmon")
axes[1].set_title("Avg Default Rate by Region")

axes[2].bar(region_perf["Region"], region_perf["Avg_Processing_Time"], color="green")
axes[2].set_title("Avg Processing Time by Region")

plt.tight_layout()
plt.savefig("task4_regional.png")
plt.close()

print("Task 4 done")


# Task 5 - Customer Segmentation

cust["Income_Segment"] = pd.qcut(cust["Annual_Income"], q=4,
                                  labels=["Low", "Lower-Mid", "Upper-Mid", "High"])
cust["Credit_Segment"] = pd.cut(cust["Credit_Score"],
                                 bins=[0, 580, 670, 740, 800, 900],
                                 labels=["Very Poor", "Fair", "Good", "Very Good", "Exceptional"])

# merging customer segments with loan data
cust_loans = cust.merge(loans[["Customer_ID", "Loan_ID", "Loan_Status",
                                "Overdue_Amount", "Default_Flag"]],
                         on="Customer_ID", how="left")

seg = cust_loans.groupby("Income_Segment", observed=True).agg(
    Avg_Credit=("Credit_Score", "mean"),
    Avg_Overdue=("Overdue_Amount", "mean"),
    Default_Rate=("Default_Flag", "mean"),
    Count=("Loan_ID", "count")
).reset_index()

print("\nSegmentation by Income:")
print(seg.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(seg["Income_Segment"].astype(str), seg["Default_Rate"], color="salmon")
axes[0].set_title("Default Rate by Income Segment")
axes[0].set_ylabel("Default Rate")

credit_seg = cust_loans.groupby("Credit_Segment", observed=True)["Default_Flag"].mean().reset_index()
axes[1].bar(credit_seg["Credit_Segment"].astype(str), credit_seg["Default_Flag"], color="steelblue")
axes[1].set_title("Default Rate by Credit Score Segment")
axes[1].set_ylabel("Default Rate")

plt.tight_layout()
plt.savefig("task5_segmentation.png")
plt.close()

print("Task 5 done")


# Task 6 - Advanced Statistical Analysis

adv = loans.merge(cust[["Customer_ID", "Credit_Score", "Annual_Income"]], on="Customer_ID", how="left")
adv = adv.merge(deflt[["Loan_ID", "Default_Amount", "Recovery_Amount"]], on="Loan_ID", how="left")
adv["Recovery_Rate"] = (adv["Recovery_Amount"] / adv["Default_Amount"]).clip(0, 1)

adv_corr = adv[["Credit_Score", "Loan_Amount", "Interest_Rate", "Overdue_Amount",
                 "EMI_Amount", "Default_Amount", "Recovery_Rate", "Default_Flag"]].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(adv_corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
ax.set_title("Advanced Pairwise Correlation Heatmap")
plt.tight_layout()
plt.savefig("task6_advanced_heatmap.png")
plt.close()

print("Task 6 done")


# Task 7 - Transaction and Recovery Analysis

penalty_trans = trans[trans["Payment_Type"] == "Penalty"]
print("\nTotal penalty transactions:", len(penalty_trans))
print("Total penalty amount:", penalty_trans["Amount"].sum())

# monthly overdue trend
trans["Trans_Month"] = trans["Transaction_Date"].dt.to_period("M")
monthly_overdue = trans.groupby("Trans_Month")["Overdue_Fee"].sum().reset_index()
monthly_overdue["Trans_Month"] = monthly_overdue["Trans_Month"].astype(str)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly_overdue["Trans_Month"], monthly_overdue["Overdue_Fee"], color="coral", linewidth=1.5)
ax.set_title("Monthly Overdue Fee Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Overdue Fee")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("task7_overdue_trend.png")
plt.close()

# recovery by default reason and legal action
deflt["Recovery_Rate_pct"] = (deflt["Recovery_Amount"] / deflt["Default_Amount"].replace(0, np.nan)).clip(0, 1) * 100

reason_rec = deflt.groupby("Default_Reason")["Recovery_Rate_pct"].mean().sort_values(ascending=False)
legal_rec = deflt.groupby("Legal_Action")["Recovery_Rate_pct"].mean()

print("\nRecovery Rate by Default Reason:")
print(reason_rec.round(2))
print("\nRecovery Rate by Legal Action:")
print(legal_rec.round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
reason_rec.plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Recovery Rate by Default Reason")
axes[0].set_ylabel("Recovery Rate (%)")
axes[0].tick_params(axis="x", rotation=30)

legal_rec.plot(kind="bar", ax=axes[1], color="green")
axes[1].set_title("Recovery Rate by Legal Action")
axes[1].set_ylabel("Recovery Rate (%)")

print("\nNote: Branch-level recovery analysis not performed due to lack of direct mapping between loans and branches in the dataset.")

plt.tight_layout()
plt.savefig("task7_recovery.png")
plt.close()

print("Task 7 done")


# Task 8 - EMI Analysis

emi_df = loans[["Loan_ID", "EMI_Amount", "Default_Flag"]].copy()
emi_df = emi_df.merge(apps[["Loan_ID", "Loan_Purpose"]], on="Loan_ID", how="left")

emi_df["EMI_Bucket"] = pd.cut(emi_df["EMI_Amount"],
                               bins=[0, 10000, 20000, 35000, 50000, 200000],
                               labels=["<10K", "10K-20K", "20K-35K", "35K-50K", ">50K"])

emi_default = emi_df.groupby("EMI_Bucket", observed=True)["Default_Flag"].mean()
print("\nDefault Rate by EMI Bucket:")
print(emi_default.round(4))

emi_purpose = emi_df.groupby("Loan_Purpose")["EMI_Amount"].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
emi_default.plot(kind="bar", ax=axes[0], color="salmon")
axes[0].set_title("Default Rate by EMI Bucket")
axes[0].set_ylabel("Default Rate")
axes[0].tick_params(axis="x", rotation=30)

emi_purpose.plot(kind="bar", ax=axes[1], color="steelblue")
axes[1].set_title("Avg EMI by Loan Purpose")
axes[1].set_ylabel("Avg EMI")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("task8_emi.png")
plt.close()

# EMI distribution histogram
plt.figure(figsize=(6,4))
plt.hist(emi_df["EMI_Amount"], bins=30)
plt.title("EMI Amount Distribution")
plt.xlabel("EMI Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("task8_distribution.png")
plt.close()

print("Task 8 done")


# Task 9 - Loan Application Insights

approval = apps["Approval_Status"].value_counts()
print("\nApproval Status:")
print(approval)
print("Approval Rate:", round(approval.get("Approved", 0) / len(apps) * 100, 2), "%")

rejection_reasons = apps[apps["Approval_Status"] == "Rejected"]["Rejection_Reason"].value_counts()
print("\nTop Rejection Reasons:")
print(rejection_reasons.head(10))

fee_by_status = apps.groupby("Approval_Status")["Processing_Fee"].mean()
print("\nAvg Processing Fee:")
print(fee_by_status.round(2))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].pie(approval, labels=approval.index, autopct="%1.1f%%", colors=["steelblue", "salmon"])
axes[0].set_title("Approval vs Rejection")

rejection_reasons.head(8).plot(kind="barh", ax=axes[1], color="coral")
axes[1].set_title("Top Rejection Reasons")
axes[1].invert_yaxis()

fee_by_status.plot(kind="bar", ax=axes[2], color=["steelblue", "salmon"])
axes[2].set_title("Avg Processing Fee by Status")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("task9_applications.png")
plt.close()

print("Task 9 done")


# Task 10 - Recovery Effectiveness

overall_rec = deflt["Recovery_Amount"].sum() / deflt["Default_Amount"].sum()
print("\nOverall Recovery Rate:", round(overall_rec * 100, 2), "%")

legal_compare = deflt.groupby("Legal_Action").apply(
    lambda x: x["Recovery_Amount"].sum() / x["Default_Amount"].sum()
).reset_index(name="Recovery_Rate")
print("\nRecovery by Legal Action:")
print(legal_compare)

region_rec = deflt_region.groupby("Region").apply(
    lambda x: x["Recovery_Amount"].sum() / x["Default_Amount"].sum()
).reset_index(name="Recovery_Rate")
print("\nRecovery Rate by Region:")
print(region_rec)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(legal_compare["Legal_Action"].astype(str), legal_compare["Recovery_Rate"], color=["steelblue", "green"])
axes[0].set_title("Recovery Rate - Legal Action vs No")
axes[0].set_ylabel("Recovery Rate")

axes[1].bar(region_rec["Region"], region_rec["Recovery_Rate"], color="coral")
axes[1].set_title("Recovery Rate by Region")
axes[1].set_ylabel("Recovery Rate")

plt.tight_layout()
plt.savefig("task10_recovery.png")
plt.close()

print("Task 10 done")


# Task 11 - Loan Disbursement Efficiency

apps["Processing_Days"] = (apps["Approval_Date"] - apps["Application_Date"]).dt.days
apps_clean = apps[apps["Processing_Days"].between(0, 365)]

print("\nAvg Processing Time:", round(apps_clean["Processing_Days"].mean(), 1), "days")
print("Median Processing Time:", round(apps_clean["Processing_Days"].median(), 1), "days")

purpose_time = apps_clean.groupby("Loan_Purpose")["Processing_Days"].mean().sort_values()
print("\nAvg Days by Loan Purpose:")
print(purpose_time.round(2))

status_time = apps_clean.groupby("Approval_Status")["Processing_Days"].mean()
print("\nAvg Processing Days by Status:")
print(status_time.round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(apps_clean["Processing_Days"].dropna(), bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of Processing Days")
axes[0].set_xlabel("Days")

purpose_time.plot(kind="barh", ax=axes[1], color="coral")
axes[1].set_title("Avg Processing Days by Loan Purpose")

plt.tight_layout()
plt.savefig("task11_efficiency.png")
plt.close()

print("Task 11 done")


# Task 12 - Profitability Analysis

# simple interest = principal * rate * time
loans["Interest_Income"] = loans["Loan_Amount"] * (loans["Interest_Rate"] / 100) * (loans["Loan_Term"] / 12)
total_interest = loans["Interest_Income"].sum()
print("\nTotal Interest Income: Rs.", round(total_interest / 1e7, 2), "Crores")

loans_p = loans.merge(apps[["Loan_ID", "Loan_Purpose"]], on="Loan_ID", how="left")
purpose_profit = loans_p.groupby("Loan_Purpose")["Interest_Income"].sum().sort_values(ascending=False)
print("\nInterest Income by Loan Purpose:")
print(purpose_profit.round(0))

loans_r = loans_p.merge(cust[["Customer_ID", "Region"]], on="Customer_ID", how="left")
region_profit = loans_r.groupby("Region")["Interest_Income"].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
purpose_profit.plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Interest Income by Loan Purpose")
axes[0].tick_params(axis="x", rotation=30)

region_profit.plot(kind="bar", ax=axes[1], color="green")
axes[1].set_title("Interest Income by Region")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("task12_profitability.png")
plt.close()

print("Task 12 done")


# Task 13 - Geospatial Analysis
# note: dataset has region names not lat/lon so using bar charts as proxy

active_loans = loans_r[loans_r["Loan_Status"] == "Active"].groupby("Region").agg(
    Active_Count=("Loan_ID", "count"),
    Total_Amount=("Loan_Amount", "sum")
).reset_index()

print("\nActive Loans by Region:")
print(active_loans)

default_rate_region = loans_r.groupby("Region")["Default_Flag"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(active_loans["Region"], active_loans["Active_Count"], color="steelblue")
axes[0].set_title("Active Loans by Region")
axes[0].set_ylabel("Count")

axes[1].bar(default_rate_region["Region"], default_rate_region["Default_Flag"], color="salmon")
axes[1].set_title("Default Rate by Region")
axes[1].set_ylabel("Default Rate")

plt.tight_layout()
plt.savefig("task13_geospatial.png")
plt.close()

print("Task 13 done")


# Task 14 - Default Trends

deflt["Default_Month"] = deflt["Default_Date"].dt.to_period("M")
monthly_def = deflt.groupby("Default_Month").size().reset_index(name="Count")
monthly_def["Default_Month"] = monthly_def["Default_Month"].astype(str)

deflt_p = deflt.merge(apps[["Loan_ID", "Loan_Purpose"]], on="Loan_ID", how="left")
avg_def_purpose = deflt_p.groupby("Loan_Purpose")["Default_Amount"].mean().sort_values(ascending=False)

deflt_c = deflt.merge(cust[["Customer_ID", "Annual_Income"]], on="Customer_ID", how="left")
deflt_c["Income_Segment"] = pd.qcut(deflt_c["Annual_Income"].dropna(), q=4,
                                     labels=["Low", "Lower-Mid", "Upper-Mid", "High"])
def_by_income = deflt_c.groupby("Income_Segment", observed=True).size()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].plot(monthly_def["Default_Month"], monthly_def["Count"], color="coral", linewidth=1.5)
axes[0].set_title("Monthly Default Trend")
axes[0].set_xlabel("Month")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

avg_def_purpose.plot(kind="bar", ax=axes[1], color="steelblue")
axes[1].set_title("Avg Default Amount by Loan Purpose")
axes[1].tick_params(axis="x", rotation=30)

def_by_income.plot(kind="bar", ax=axes[2], color="green")
axes[2].set_title("Defaults by Income Segment")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("task14_default_trends.png")
plt.close()

print("Task 14 done")


# Task 15 - Branch Efficiency

print("\nBranch Processing Times:")
print(branches[["Branch_Name", "Region", "Avg_Processing_Time"]].sort_values("Avg_Processing_Time").to_string(index=False))

rejected = apps[apps["Approval_Status"] == "Rejected"]
print("\nTotal Rejected Applications:", len(rejected))

fig, ax = plt.subplots(figsize=(12, 6))
b_sorted = branches.sort_values("Avg_Processing_Time", ascending=False).head(15)
ax.barh(b_sorted["Branch_Name"], b_sorted["Avg_Processing_Time"], color="coral")
ax.set_title("Top 15 Branches by Avg Processing Time")
ax.set_xlabel("Days")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("task15_branch_efficiency.png")
plt.close()

print("Task 15 done")


# Task 16 - Time Series Analysis

loans["Disbursal_Month"] = loans["Disbursal_Date"].dt.to_period("M")
monthly_disb = loans.groupby("Disbursal_Month").agg(
    Total=("Loan_Amount", "sum"),
    Count=("Loan_ID", "count")
).reset_index()
monthly_disb["Disbursal_Month"] = monthly_disb["Disbursal_Month"].astype(str)

# seasonal pattern by month number
loans["Month_Num"] = loans["Disbursal_Date"].dt.month
seasonal = loans.groupby("Month_Num")["Loan_Amount"].sum()
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

deflt_region2 = deflt_region.copy()
deflt_region2["Default_Month"] = deflt_region2["Default_Date"].dt.to_period("M")
monthly_def_region = deflt_region2.groupby(["Default_Month", "Region"]).size().unstack(fill_value=0)
monthly_def_region.index = monthly_def_region.index.astype(str)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].plot(monthly_disb["Disbursal_Month"], monthly_disb["Total"] / 1e7, color="steelblue")
axes[0].set_title("Monthly Disbursement Trend (Cr)")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

axes[1].bar(months, seasonal / 1e7, color="coral")
axes[1].set_title("Seasonal Disbursement Pattern")
axes[1].set_ylabel("Amount (Cr)")

monthly_def_region.plot(ax=axes[2], linewidth=1.2)
axes[2].set_title("Monthly Defaults by Region")
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig("task16_timeseries.png")
plt.close()

print("Task 16 done")


# Task 17 - Customer Behavior Analysis

cust_beh = loans.groupby("Customer_ID").agg(
    Total_Loans=("Loan_ID", "count"),
    Total_Overdue=("Overdue_Amount", "sum"),
    Defaults=("Default_Flag", "sum")
).reset_index()

def label_behavior(row):
    if row["Defaults"] == 0 and row["Total_Overdue"] == 0:
        return "Always On Time"
    elif row["Defaults"] >= 2:
        return "Frequent Defaulter"
    elif row["Defaults"] == 1:
        return "Occasional Defaulter"
    else:
        return "Overdue But No Default"

cust_beh["Behavior"] = cust_beh.apply(label_behavior, axis=1)
print("\nCustomer Behavior:")
print(cust_beh["Behavior"].value_counts())

# high value customers
good = cust_beh[cust_beh["Behavior"] == "Always On Time"]
good = good.merge(cust[["Customer_ID", "Credit_Score", "Annual_Income"]], on="Customer_ID", how="left")
print("\nTop 10 high value customers:")
print(good.nlargest(10, "Annual_Income")[["Customer_ID", "Credit_Score", "Annual_Income"]].to_string(index=False))

apps_c = apps.merge(cust[["Customer_ID", "Employment_Status"]], on="Customer_ID", how="left")
emp_app = apps_c.groupby(["Employment_Status", "Approval_Status"]).size().unstack(fill_value=0)
print("\nApproval by Employment Status:")
print(emp_app)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cust_beh["Behavior"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Customer Repayment Behavior")
axes[0].tick_params(axis="x", rotation=30)

emp_app.plot(kind="bar", stacked=True, ax=axes[1])
axes[1].set_title("Loan Outcome by Employment Status")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("task17_behavior.png")
plt.close()

print("Task 17 done")


# Task 18 - Risk Assessment

risk = loans.merge(apps[["Loan_ID", "Loan_Purpose"]], on="Loan_ID", how="left")
risk = risk.merge(deflt[["Loan_ID", "Default_Amount"]], on="Loan_ID", how="left")

risk_matrix = risk.groupby("Loan_Purpose").agg(
    Avg_Default_Amt=("Default_Amount", "mean"),
    Avg_Term=("Loan_Term", "mean"),
    Avg_Rate=("Interest_Rate", "mean"),
    Default_Rate=("Default_Flag", "mean")
).reset_index()

# normalizing and computing risk score
for col in ["Avg_Default_Amt", "Avg_Term", "Avg_Rate", "Default_Rate"]:
    mn, mx = risk_matrix[col].min(), risk_matrix[col].max()
    risk_matrix[col + "_n"] = (risk_matrix[col] - mn) / (mx - mn + 1e-9)

risk_matrix["Risk_Score"] = (risk_matrix["Avg_Default_Amt_n"] * 0.4 +
                              risk_matrix["Default_Rate_n"] * 0.4 +
                              risk_matrix["Avg_Rate_n"] * 0.2)

risk_matrix = risk_matrix.sort_values("Risk_Score", ascending=False)
print("\nRisk Matrix:")
print(risk_matrix[["Loan_Purpose", "Default_Rate", "Avg_Default_Amt", "Risk_Score"]].to_string(index=False))

# high risk customers by credit score
cust["Risk_Group"] = pd.cut(cust["Credit_Score"],
                             bins=[0, 580, 670, 740, 850],
                             labels=["High Risk", "Medium Risk", "Low Risk", "Very Low Risk"])
print("\nCustomers by Risk Group:")
print(cust["Risk_Group"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
risk_matrix.plot(x="Loan_Purpose", y="Risk_Score", kind="bar", ax=axes[0], legend=False, color="salmon")
axes[0].set_title("Risk Score by Loan Purpose")
axes[0].tick_params(axis="x", rotation=30)

cust["Risk_Group"].value_counts().plot(kind="bar", ax=axes[1],
                                        color=["red", "orange", "yellowgreen", "green"])
axes[1].set_title("Customer Risk Groups")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("task18_risk.png")
plt.close()

print("Task 18 done")


# Task 19 - Time to Default Analysis

ttd = deflt.merge(loans[["Loan_ID", "Disbursal_Date"]], on="Loan_ID", how="left")
ttd["Days_to_Default"] = (ttd["Default_Date"] - ttd["Disbursal_Date"]).dt.days
ttd = ttd[ttd["Days_to_Default"].between(1, 3650)]

print("\nAvg Days to Default:", round(ttd["Days_to_Default"].mean(), 1))
print("Median:", round(ttd["Days_to_Default"].median(), 1))

ttd_p = ttd.merge(apps[["Loan_ID", "Loan_Purpose"]], on="Loan_ID", how="left")
ttd_purpose = ttd_p.groupby("Loan_Purpose")["Days_to_Default"].mean().sort_values()
print("\nAvg Days to Default by Purpose:")
print(ttd_purpose.round(1))

ttd_c = ttd.merge(cust[["Customer_ID", "Employment_Status"]], on="Customer_ID", how="left")
ttd_emp = ttd_c.groupby("Employment_Status")["Days_to_Default"].mean().sort_values()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].hist(ttd["Days_to_Default"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Days to Default Distribution")
axes[0].set_xlabel("Days")

ttd_purpose.plot(kind="barh", ax=axes[1], color="coral")
axes[1].set_title("Avg Days to Default by Purpose")

ttd_emp.plot(kind="barh", ax=axes[2], color="green")
axes[2].set_title("Avg Days to Default by Employment")

plt.tight_layout()
plt.savefig("task19_time_to_default.png")
plt.close()

print("Task 19 done")


# Task 20 - Transaction Pattern Analysis

cust_t = trans.groupby("Customer_ID").agg(
    Total_Amount=("Amount", "sum"),
    Total_Overdue_Fee=("Overdue_Fee", "sum"),
    Penalty_Count=("Payment_Type", lambda x: (x == "Penalty").sum()),
    Total_Txns=("Transaction_ID", "count")
).reset_index()

cust_t["Penalty_Pct"] = cust_t["Penalty_Count"] / cust_t["Total_Txns"]
irregular = cust_t[cust_t["Penalty_Pct"] > 0.3]
print("\nIrregular customers (penalty > 30%):", len(irregular))

penalty_pct = trans[trans["Payment_Type"] == "Penalty"]["Amount"].sum() / trans["Amount"].sum()
print("Penalty as % of total transactions:", round(penalty_pct * 100, 2), "%")

loans_ov = loans[["Loan_ID", "Overdue_Amount"]].copy()
loans_ov["Is_Overdue"] = loans_ov["Overdue_Amount"] > 0
trans_ov = trans.merge(loans_ov[["Loan_ID", "Is_Overdue"]], on="Loan_ID", how="left")
ov_compare = trans_ov.groupby("Is_Overdue")["Amount"].mean()
print("\nAvg Transaction Amount - Overdue vs Non-Overdue:")
print(ov_compare.round(2))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].hist(cust_t["Penalty_Pct"], bins=30, color="salmon", edgecolor="white")
axes[0].set_title("Penalty % per Customer")
axes[0].set_xlabel("Penalty Proportion")

trans["Payment_Type"].value_counts().plot(kind="bar", ax=axes[1], color="steelblue")
axes[1].set_title("Transaction Type Breakdown")
axes[1].tick_params(axis="x", rotation=30)

ov_compare.plot(kind="bar", ax=axes[2], color=["green", "coral"])
axes[2].set_xticklabels(["Non-Overdue", "Overdue"], rotation=0)
axes[2].set_title("Avg Txn Amount: Overdue vs Non-Overdue")

plt.tight_layout()
plt.savefig("task20_transactions.png")
plt.close()

print("Task 20 done")

print("\nAll tasks completed! Check the PNG files for charts.")

