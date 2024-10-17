import os
import subprocess


def run_script():
    # Get the directory where the executable or script is located
    base_folder = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Change directory to the 'v3' folder
    v3_folder = os.path.join(base_folder, 'v3')
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

    # Step 3: Change directory back to the root folder where 'venv' is located
    root_folder = base_folder
    os.chdir(root_folder)

    print(f"Changed back to root directory: {os.getcwd()}")

    # Step 4: Activate the virtual environment and run 'chat_ui_new.py' using Streamlit
    venv_activate_script = os.path.join(root_folder, 'venv', 'Scripts', 'activate.bat')  # Windows path

    # Check for Linux/macOS
    if not os.path.exists(venv_activate_script):
        venv_activate_script = os.path.join(root_folder, 'venv', 'bin', 'activate')  # Unix path

    try:
        print("Activating virtual environment...")

        # For Windows
        if os.name == 'nt':  # Windows system
            # Use cmd.exe to run the activation script and Streamlit in sequence
            subprocess.run(f'cmd /c "{venv_activate_script} && streamlit run v3/chat_ui_new.py"', shell=True,
                           check=True)
        else:  # For Linux/macOS
            # Use bash for Unix-based systems to activate and run Streamlit
            subprocess.run(f'bash -c "source {venv_activate_script} && streamlit run v3/chat_ui_new.py"', shell=True,
                           check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running 'chat_ui_new.py' with Streamlit: {e}")


if __name__ == "__main__":
    run_script()
