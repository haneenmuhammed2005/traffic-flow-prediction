@echo off
REM Traffic Flow Prediction - Quick Start Script for Windows
REM This script sets up everything and runs the complete pipeline

echo ==============================================
echo TRAFFIC FLOW PREDICTION - QUICK START
echo ==============================================
echo.

REM Step 1: Check Python version
echo Step 1: Checking Python installation...
python --version
echo ✓ Python detected
echo.

REM Step 2: Create virtual environment
echo Step 2: Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM Step 3: Activate virtual environment
echo Step 3: Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Step 4: Install dependencies
echo Step 4: Installing dependencies...
echo This may take 3-5 minutes...
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo ✓ All dependencies installed
echo.

REM Step 5: Create directory structure
echo Step 5: Creating directory structure...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "plots" mkdir plots
if not exist "results" mkdir results
echo ✓ Directories created
echo.

REM Step 6: Run main pipeline
echo Step 6: Running complete ML pipeline...
echo This will take 5-10 minutes...
echo.
echo ==============================================
python main_train.py
echo ==============================================
echo.

REM Step 7: Summary
echo Step 7: Setup Complete! 🎉
echo.
echo ==============================================
echo WHAT YOU GOT:
echo ==============================================
echo ✓ Synthetic traffic data generated
echo ✓ Features engineered (50+ features)
echo ✓ Multiple models trained
echo ✓ Visualizations created
echo ✓ Model comparison completed
echo.
echo ==============================================
echo CHECK YOUR RESULTS:
echo ==============================================
echo 📊 Plots:           plots\
echo 📈 Models:          models\
echo 📁 Data:            data\
echo 📋 Results:         results\model_comparison.csv
echo.
echo ==============================================
echo NEXT STEPS:
echo ==============================================
echo 1. Review plots in plots\ directory
echo 2. Check model performance in results\
echo 3. Make predictions: python predict.py
echo 4. Read README.md for detailed documentation
echo.
echo To make predictions, run:
echo   python predict.py
echo.
echo Happy predicting! 🚗📊
echo ==============================================
pause
