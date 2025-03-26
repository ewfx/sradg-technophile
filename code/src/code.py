import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from difflib import SequenceMatcher

# Load CSV from src folder
csv_path = "src/mockData.csv"
df = pd.read_csv(csv_path)

# Function to check similarity
def similar(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

# Compare based on Company, Account, AU, and Currency
df["Match_Status"] = df.apply(lambda x: (
    "Match" if x["Company"] == x["Company"] and 
    x["Account"] == x["Account"] and 
    x["AU"] == x["AU"] and 
    x["Currency"] == x["Currency"] and 
    similar(x["General Ledger"], x["IHub"]) >= 0.99 else "Mismatch"
), axis=1)

# Filter mismatches
mismatches = df[df["Match_Status"] == "Mismatch"]
print("Mismatched Entries:\n", mismatches)



# Initialize OpenAI LLM
llm = OpenAI(api_token="your_openai_api_key")
pandas_ai = PandasAI(llm)

# AI-driven reconciliation query
query = """
Compare the 'General Ledger' and 'IHub' columns while ensuring 'Company', 'Account', 'AU', and 'Currency' match.
Identify discrepancies and provide suggestions for remediation.
"""
response = pandas_ai.run(df, prompt=query)

# Display AI-powered reconciliation results
print(response)


# Auto-fix if similarity > 0.90
for index, row in mismatches.iterrows():
    if similar(row["General Ledger"], row["IHub"]) > 0.90:
        df.at[index, "IHub"] = row["General Ledger"]  # Fix in IHub column

# Save corrected file back to src folder
df.to_csv("src/ledger_vs_ihub_fixed.csv", index=False)
print("Corrected file saved as src/ledger_vs_ihub_fixed.csv")


# AI-Powered Remediation for Mismatches
def ai_remediation(row):
    prompt = f"""
    Given the following mismatch:
    Company: {row['Company']}
    Account: {row['Account']}
    AU: {row['AU']}
    Currency: {row['Currency']}
    General Ledger Value: {row['General Ledger']}
    IHub Value: {row['IHub']}
    
    Suggest the best way to correct this discrepancy.
    """
    return pandas_ai.run(df, prompt=prompt)

# Apply AI remediation to mismatches
mismatches["AI_Remediation"] = mismatches.apply(ai_remediation, axis=1)

# Save AI remediation results
mismatches.to_csv("src/ledger_vs_ihub_mismatches.csv", index=False)
print("AI Remediation file saved as src/ledger_vs_ihub_mismatches.csv")


