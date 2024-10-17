import pandas as pd

def process_excel_file(input_file, output_file):
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Keep only the necessary columns
    columns_to_keep = ['Incident ID', 'Summary', 'Resolution', 'Status']
    df = df[columns_to_keep]
    
    # Filter rows to keep only the ones with 'Resolved' status
    df = df[df['Status'] == 'Resolved']
    
    # Remove rows where 'Resolution' has unwanted values (case insensitive)
    unwanted_solutions = ['.', '...', 'fixed', 'resolved', 'test', 'duplicate', 'other']
    df = df[~df['Resolution'].str.strip().str.lower().isin(unwanted_solutions)]
    
    # Drop the 'Status' column since it's no longer needed
    df = df.drop(columns=['Status'])
    
    # Rename the columns
    df = df.rename(columns={
        'Incident ID': 'Ticket #',
        'Summary': 'Problem',
        'Resolution': 'Solution'
    })
    
    # Export the result to a CSV file
    df.to_csv(output_file, index=False)

# Example usage
input_file = 'MIR Exports2024_17_10_18_41_21.xlsx'  # replace with your input Excel file path
output_file = 'tickets_dataset_NEW.csv'  # replace with your desired output CSV file path
process_excel_file(input_file, output_file)
