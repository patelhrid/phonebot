import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_script():
    # Get the base folder (use _MEIPASS for PyInstaller bundle)
    if hasattr(sys, '_MEIPASS'):
        base_folder = sys._MEIPASS
    else:
        base_folder = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Reference the 'main_knn_new_copy.py' script
    main_knn_script = os.path.join(base_folder, 'main_knn_new_copy.py')
    chat_ui_script = os.path.join(base_folder, 'chat_ui_new_copy.py')

    # Log paths
    logger.info(f"Base folder: {base_folder}")
    logger.info(f"Path to 'main_knn_new_copy.py': {main_knn_script}")
    logger.info(f"Path to 'chat_ui_new_copy.py': {chat_ui_script}")

    if not os.path.exists(main_knn_script):
        logger.error(f"'main_knn_new_copy.py' not found.")
        return

    try:
        # Run 'main_knn_new_copy.py' using the same interpreter that runs this script
        logger.info("Running 'main_knn_new_copy.py'...")
        subprocess.run([sys.executable, main_knn_script], check=True)
        logger.info("'main_knn_new_copy.py' finished successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'main_knn_new_copy.py': {e}")
        return

    try:
        # Now run Streamlit with 'chat_ui_new_copy.py' using the same interpreter
        logger.info("Running 'chat_ui_new_copy.py' with Streamlit...")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', chat_ui_script], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'chat_ui_new_copy.py' with Streamlit: {e}")


if __name__ == "__main__":
    run_script()
