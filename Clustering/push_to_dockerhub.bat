@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo    Docker Hub Build and Push Tool - Clustering App
echo ========================================================
echo.

set DOCKER_USER=%1
if "%DOCKER_USER%"=="" (
    set /p DOCKER_USER="Enter your Docker Hub username: "
)

if "%DOCKER_USER%"=="" (
    echo [ERROR] Docker Hub username cannot be empty.
    pause
    exit /b 1
)

set IMAGE_NAME=%DOCKER_USER%/clustering-app:latest

echo.
echo [1/3] Logging in to Docker Hub...
docker login
if errorlevel 1 (
    echo [ERROR] Docker login failed. Please ensure Docker Desktop is running and credentials are correct.
    pause
    exit /b 1
)

echo.
echo [2/3] Building Docker image '%IMAGE_NAME%'...
docker build -t %IMAGE_NAME% .
if errorlevel 1 (
    echo [ERROR] Docker build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Pushing '%IMAGE_NAME%' to Docker Hub...
docker push %IMAGE_NAME%
if errorlevel 1 (
    echo [ERROR] Docker push failed.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [SUCCESS] Image successfully pushed to Docker Hub!
echo Image: %IMAGE_NAME%
echo.
echo To run this image on ANY machine with just Docker installed:
echo   docker pull %IMAGE_NAME%
echo   docker run -p 8501:8501 %IMAGE_NAME%
echo.
echo Then open your browser at: http://localhost:8501
echo ========================================================
pause

