@echo off
echo Starting IPL AI Web Server...
cd /d "C:\Users\Sanjay Sharma\Desktop\IPL prediction agent"
call conda activate base
echo Open your browser at: http://localhost:8000
echo Press Ctrl+C to stop
echo.
python -m uvicorn web.app:app --reload --port 8000
pause
