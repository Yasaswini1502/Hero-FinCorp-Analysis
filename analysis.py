import pandas as pd


customers = pd.read_csv("data/customers.csv")
loans = pd.read_csv("data/loans.csv")
defaults = pd.read_csv("data/defaults.csv")


df = loans.merge(customers, on="Customer_ID", how="left")


df['Default_Flag'] = df['Loan_ID'].isin(defaults['Loan_ID']).astype(int)


print("Total Loans:", len(df))
print("Default Rate:", df['Default_Flag'].mean())


print("Average Credit Score (Defaulted):", df[df['Default_Flag']==1]['Credit_Score'].mean())
print("Average Credit Score (Non-Default):", df[df['Default_Flag']==0]['Credit_Score'].mean())
