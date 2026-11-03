from src.loader import load_dataset
from src.profiler import profile_data
from src.cleaner import *
from src.reporter import *
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
file_path = "data/spam.csv"

# Load dataset
df = load_dataset(file_path)

# Profile original data
profile = profile_data(df)
print("Before cleaning:")
print(df.head())

df = standardize_missing(df)
df = fix_data_types(df)
df = handle_missing(df)
df = standardize_text(df)
df = handle_outliers(df)
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

print("After cleaning:")
print(df.head())
# Save cleaned dataset
df.to_csv("cleaned_data/spam_cleaned.csv", index=False)

# Generate reports
generate_json_report(profile, "reports/cleaning_report.json")
generate_pdf_report(profile, "reports/cleaning_report.pdf")

print("Data Cleaning Completed")
print("Cleaned dataset saved to cleaned_data/spam_cleaned.csv")