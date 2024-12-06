import subprocess
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def is_running_from_exe():
    """Determine if the script is running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)

def run_streamlit_app():
    # Avoid recursive launch attempts
    if os.getenv("IS_STREAMLIT_SUBPROCESS") == "true":
        logger.error("Detected recursive launch attempt. Exiting.")
        sys.exit(1)

    # Path to the Streamlit script
    script_path = os.path.abspath("chat_ui_new_copy.py")
    if is_running_from_exe():
        script_path = os.path.join(sys._MEIPASS, "chat_ui_new_copy.py")

    # Set an environment variable to mark this as a Streamlit subprocess
    env = os.environ.copy()
    env["IS_STREAMLIT_SUBPROCESS"] = "true"

    # Launch Streamlit in a detached process
    try:
        logger.info(f"Launching Streamlit app from script: {script_path}")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", script_path, "--server.enableXsrfProtection=false"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Prevent inheriting signals
        )
    except Exception as e:
        logger.error(f"Failed to launch Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()
