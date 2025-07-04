@echo off
cd /d %~dp0

echo Starting BramBot...

REM Use the Python from your virtual environment
call .venv\Scripts\activate

REM Launch Streamlit using that Python
.venv\Scripts\python.exe -m streamlit run site_launch_conv.py --server.enableXsrfProtection=false --server.port=8501

pause
