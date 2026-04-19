import pandas as pd
customers = pd.read_csv("data/customers.csv")
loans = pd.read_csv("data/loans.csv")
defaults = pd.read_csv("data/defaults.csv")

df = loans.merge(customers, on="Customer_ID", how="left")


df['Default_Flag'] = df['Loan_ID'].isin(defaults['Loan_ID']).astype(int)


print("===== BASIC SUMMARY =====")
print("Total Loans:", len(df))
print("Default Rate:", df['Default_Flag'].mean())


print("\n===== CREDIT SCORE ANALYSIS =====")
print("Average Credit Score (Defaulted):", 
      df[df['Default_Flag'] == 1]['Credit_Score'].mean())

print("Average Credit Score (Non-Default):", 
      df[df['Default_Flag'] == 0]['Credit_Score'].mean())


print("\n===== LOAN AMOUNT ANALYSIS =====")
print("Average Loan Amount (Defaulted):", 
      df[df['Default_Flag'] == 1]['Loan_Amount'].mean())

print("Average Loan Amount (Non-Default):", 
      df[df['Default_Flag'] == 0]['Loan_Amount'].mean())


print("\n===== INTEREST RATE ANALYSIS =====")
print("Average Interest Rate (Defaulted):", 
      df[df['Default_Flag'] == 1]['Interest_Rate'].mean())

print("Average Interest Rate (Non-Default):", 
      df[df['Default_Flag'] == 0]['Interest_Rate'].mean())


def segment(row):
    if row['Credit_Score'] >= 750:
        return "High Value"
    elif row['Credit_Score'] < 600:
        return "High Risk"
    else:
        return "Moderate"

df['Customer_Segment'] = df.apply(segment, axis=1)

print("\n===== CUSTOMER SEGMENT DISTRIBUTION =====")
print(df['Customer_Segment'].value_counts())


# FINAL 
print("\n===== KEY INSIGHTS =====")
print("- Default rate shows overall risk level of portfolio")
print("- Credit score alone may not fully explain defaults")
print("- Higher loan amounts may contribute to defaults")
print("- Interest rates also influence repayment behavior")
