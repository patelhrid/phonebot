import os
import subprocess


def run_script():
    # Step 1: Change directory to the 'v3' folder
    v3_folder = os.path.join(os.getcwd(), 'v3')  # Assuming 'v3' is a subfolder in the current directory
    os.chdir(v3_folder)

    print(f"Changed to directory: {os.getcwd()}")

    # Step 2: Run 'main_knn_new.py' and wait for it to complete
    try:
        print("Running 'main_knn_new.py'...")
        subprocess.run(['python', 'main_knn_new.py'], check=True)
        print("'main_knn_new.py' finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running 'main_knn_new.py': {e}")
        return

    # Step 3: Change directory back to the project root where 'venv' is located
    root_folder = os.path.dirname(os.getcwd())  # Assuming the root folder is one level up
    os.chdir(root_folder)

    print(f"Changed back to root directory: {os.getcwd()}")

    # Step 4: Activate the virtual environment and run 'chat_ui_new.py' using Streamlit
    venv_path = os.path.join(root_folder, 'venv', 'Scripts',
                             'activate')  # Adjust for your OS (Scripts for Windows, bin for Linux/Mac)

    try:
        print("Activating virtual environment and running 'chat_ui_new.py' with Streamlit...")

        # Run the Streamlit app using the virtual environment
        subprocess.run([f'{venv_path} && streamlit run v3/chat_ui_new.py'], shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running 'chat_ui_new.py' with Streamlit: {e}")


if __name__ == "__main__":
    run_script()
