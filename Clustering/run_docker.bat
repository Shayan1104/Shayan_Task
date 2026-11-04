@echo off
setlocal

echo ========================================================
echo    Pull and Run Clustering Studio from Docker Hub
echo ========================================================
echo.

set IMAGE_NAME=%1
if "%IMAGE_NAME%"=="" (
    set /p IMAGE_NAME="Enter full Docker Hub image tag (e.g. yourusername/clustering-app:latest): "
)

if "%IMAGE_NAME%"=="" (
    echo [ERROR] Image name cannot be empty.
    pause
    exit /b 1
)

echo.
echo [1/2] Pulling image from Docker Hub...
docker pull %IMAGE_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to pull image. Please verify image name and Docker status.
    pause
    exit /b 1
)

echo.
echo [2/2] Running container on port 8501...
docker stop clustering-app 2>nul
docker rm clustering-app 2>nul
docker run -d -p 8501:8501 --name clustering-app %IMAGE_NAME%

echo.
echo ========================================================
echo [SUCCESS] Clustering Studio is now running!
echo URL: http://localhost:8501
echo.
echo To view container logs: docker logs -f clustering-app
echo To stop container:       docker stop clustering-app
echo ========================================================
start http://localhost:8501
pause

