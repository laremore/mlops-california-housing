@echo off
echo ========================================
echo start MLOps pipeline 
echo ========================================

echo 1. collecting Docker ...
docker build -t california-housing-ml .

echo.
echo 2. starting services Docker Compose...
docker-compose up -d

echo.
echo 3. cheching work...
timeout /t 5 /nobreak > nul

echo.
echo 4. checking api...
curl http://localhost:8000/health

echo.
echo 5. checking MLflow UI...
echo MLflow UI is available: http://localhost:5000

echo.
echo ========================================
echo all service work!
echo ========================================
echo API service: http://localhost:8000/docs
echo MLflow UI: http://localhost:5000
echo ========================================
pause