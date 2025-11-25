@echo off
echo ============================================
echo Starting ML Server, Node Backend, and Gradio Frontend
echo ============================================
echo.

REM === Project Paths ===
set BASE=D:\Code_Playground\Projects\Image-demo
set ML_DIR=%BASE%\ml-model
set NODE_DIR=%BASE%\backend-node
set FRONT_DIR=%BASE%\frontend-gradio

REM === Environment Variables ===
set JWT_SECRET=mysecret123
set MONGO_URI=mongodb://127.0.0.1:27017/image_demo

REM === Start ML Server ===
echo [1/3] Starting ML Server...
start "ML SERVER" cmd /k "cd /d %ML_DIR% && if exist venv\Scripts\activate.bat (venv\Scripts\activate.bat) && python ml_server.py"

REM === Start Node Backend ===
echo [2/3] Starting Node Backend...
start "NODE BACKEND" cmd /k "cd /d %NODE_DIR% && set JWT_SECRET=%JWT_SECRET% && set MONGO_URI=%MONGO_URI% && node server.js"

REM === Start Gradio Frontend ===
echo [3/3] Starting Gradio Frontend...
start "GRADIO FRONTEND" cmd /k "cd /d %FRONT_DIR% && if exist ..\ml-model\venv\Scripts\activate.bat (..\ml-model\venv\Scripts\activate.bat) && python gradio_auth_modern.py"

echo.
echo ============================================
echo All services launched successfully!
echo ============================================
echo.
echo Windows:
echo   - ML Server:        http://127.0.0.1:5000
echo   - Node Backend:     http://127.0.0.1:3000
echo   - Gradio Frontend:  http://127.0.0.1:7860
echo.
pause