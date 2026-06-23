@echo off  
cd /d "%~dp0"  
pip install numpy gradio  
python app.py  
pause