@echo off
if "%1"=="build" goto build
if "%1"=="run" goto run
if "%1"=="stop" goto stop
if "%1"=="test" goto test
if "%1"=="clean" goto clean
if "%1"=="restart" goto restart

echo Available commands:
echo   build    - Build Docker image
echo   run      - Run Docker container
echo   stop     - Stop and remove container
echo   test     - Test health endpoint
echo   clean    - Remove image and container
echo   restart  - Restart container
goto:eof

:build
docker build -t california-housing-ml .
goto:eof

:run
docker run -d -p 8000:8000 --name housing-ml-service california-housing-ml
echo Service running at http://localhost:8000
goto:eof

:stop
docker stop housing-ml-service 2>nul || echo Container not running
docker rm housing-ml-service 2>nul || echo Container not found
goto:eof

:test
curl http://localhost:8000/health
goto:eof

:clean
call :stop
docker rmi california-housing-ml 2>nul || echo Image not found
goto:eof

:restart
call :stop
call :run
goto:eof