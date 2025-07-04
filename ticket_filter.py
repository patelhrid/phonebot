import pandas as pd


def dataset_setup(input_file="MIR Exports2025_04_7_15_09_53.xlsx", output_file="tickets_dataset_NEW.csv"):
    try:
        # Load the main dataset
        df = pd.read_excel(input_file)

        # Keep only the necessary columns
        columns_to_keep = ['Incident ID', 'Summary', 'Resolution', 'Status']
        df = df[columns_to_keep]

        # Filter rows to keep only the ones with 'Resolved' status
        df = df[df['Status'] == 'Resolved']

        # Remove rows where 'Resolution' has unwanted values (case-insensitive)
        # unwanted_solutions = ['.', '...', 'fixed', 'resolved', 'test', 'duplicate', 'other']
        # df = df[~df['Resolution'].str.strip().str.lower().isin(unwanted_solutions)]

        # Remove rows where 'Resolution' is empty
        df = df[df['Resolution'].str.strip() != '']

        # Drop the 'Status' column since it's no longer needed
        df = df.drop(columns=['Status'])

        # Rename the columns
        df = df.rename(columns={
            'Incident ID': 'Ticket #',
            'Summary': 'Problem',
            'Resolution': 'Solution'
        })

        # Export the result to a CSV file
        df.to_csv("filtered_tickets_dataset.csv", index=False, encoding="utf-8")

        # Paths to the files
        knowledge_articles_file = "knowledge_articles_export.xlsx"  # Update with your file path
        tickets_file = "filtered_tickets_dataset.csv"  # Update with your file path

        # Load the knowledge articles
        if knowledge_articles_file.endswith('.xlsx'):
            knowledge_articles = pd.read_excel(knowledge_articles_file)
        else:
            knowledge_articles = pd.read_csv(
                knowledge_articles_file,
                encoding="utf-8",  # Use UTF-8 for better character support
                on_bad_lines='skip'  # Skip malformed lines
            )

        # Filter for 'Published' articles
        knowledge_articles = knowledge_articles[knowledge_articles['Status'] == "Published"]

        # Select the required columns
        knowledge_articles = knowledge_articles[['Title']]

        # Create new rows for the tickets dataset
        knowledge_articles['Problem'] = knowledge_articles['Title']
        knowledge_articles['Solution'] = knowledge_articles['Title'].apply(
            lambda title: f"Refer to the '{title}' knowledge article"
        )
        knowledge_articles['Ticket #'] = ""  # Leave Ticket # blank

        # Keep only the required columns
        knowledge_articles = knowledge_articles[['Ticket #', 'Problem', 'Solution']]

        # Load the existing tickets dataset
        tickets_df = pd.read_csv(tickets_file, encoding="utf-8")

        # Append the knowledge articles to the tickets dataset
        updated_tickets_df = pd.concat([tickets_df, knowledge_articles], ignore_index=True)

        # Save the updated dataset
        updated_tickets_df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"Updated tickets dataset saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


dataset_setup()
