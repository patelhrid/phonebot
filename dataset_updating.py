import pandas as pd
import tkinter as tk
from tkinter import messagebox


# Function to append new ticket data to the CSV
def append_to_csv(ticket, problem, solution, csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Ticket #', 'Problem', 'Solution'])

    new_entry = pd.DataFrame({'Ticket #': [ticket], 'Problem': [problem], 'Solution': [solution]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file, index=False)


# Function to remove a ticket by Ticket #
def remove_from_csv(ticket, csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        messagebox.showerror("Error", "CSV file not found.")
        return

    # Ensure Ticket # and user input are compared as strings and strip spaces
    ticket = str(ticket).strip()

    # Check if the ticket exists (Ticket # column as string)
    if ticket not in df['Ticket #'].astype(str).values:
        messagebox.showwarning("Not Found", f"Ticket #{ticket} not found.")
        return

    # Remove the row and save the updated CSV
    df = df[df['Ticket #'].astype(str) != ticket]
    df.to_csv(csv_file, index=False)
    messagebox.showinfo("Success", f"Ticket #{ticket} removed successfully.")


# Submit button action to add a new ticket
def submit_data():
    ticket = entry_ticket.get().strip()
    problem = entry_problem.get().strip()
    solution = entry_solution.get().strip()

    if not ticket or not problem or not solution:
        messagebox.showwarning("Input Error", "All fields must be filled out.")
        return

    append_to_csv(ticket, problem, solution, 'tickets_dataset_NEW.csv')
    messagebox.showinfo("Success", "Ticket added successfully!")
    entry_ticket.delete(0, tk.END)
    entry_problem.delete(0, tk.END)
    entry_solution.delete(0, tk.END)


# Remove button action to remove a ticket by Ticket #
def remove_data():
    ticket = entry_ticket.get().strip()

    if not ticket:
        messagebox.showwarning("Input Error", "Please enter a Ticket # to remove.")
        return

    remove_from_csv(ticket, 'tickets_dataset_NEW.csv')
    entry_ticket.delete(0, tk.END)


# Create the GUI window
root = tk.Tk()
root.title("Ticket Management")
root.geometry("400x400")

# Ticket # input
label_ticket = tk.Label(root, text="Ticket #")
label_ticket.pack(pady=5)
entry_ticket = tk.Entry(root)
entry_ticket.pack(pady=5)

# Problem input
label_problem = tk.Label(root, text="Problem")
label_problem.pack(pady=5)
entry_problem = tk.Entry(root)
entry_problem.pack(pady=5)

# Solution input
label_solution = tk.Label(root, text="Solution")
label_solution.pack(pady=5)
entry_solution = tk.Entry(root)
entry_solution.pack(pady=5)

# Submit button to add a new ticket
submit_button = tk.Button(root, text="Add Ticket", command=submit_data)
submit_button.pack(pady=10)

# Remove button to remove a ticket by Ticket #
remove_button = tk.Button(root, text="Remove Ticket", command=remove_data)
remove_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
