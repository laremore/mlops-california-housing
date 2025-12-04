@echo off
echo ========================================
echo Deploying California Housing ML Service
echo ========================================

echo 1. Stopping existing containers...
docker-compose down

echo.
echo 2. Pulling latest changes...
git pull origin main

echo.
echo 3. Building new Docker image...
docker-compose build

echo.
echo 4. Starting services...
docker-compose up -d

echo.
echo 5. Waiting for services to start...
timeout /t 10 /nobreak > nul

echo.
echo 6. Testing API...
curl http://localhost:8000/health

echo.
echo ========================================
echo Deployment completed successfully!
echo ========================================
echo API: http://localhost:8000/docs
echo MLflow: http://localhost:5000
echo ========================================