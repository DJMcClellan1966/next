@echo off
echo ============================================================
echo ML Toolbox - Installing Essential Dependencies Only
echo ============================================================
echo.
echo This installs only the essential packages to run ML Toolbox.
echo Optional features will use simplified implementations.
echo.
pause

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing essential packages...
python -m pip install numpy scikit-learn pandas psutil py-cpuinfo flask

echo.
echo ============================================================
echo Essential Installation Complete!
echo ============================================================
echo.
echo Core ML Toolbox features are now available.
echo Some advanced features may use simplified implementations.
echo.
echo To install all dependencies, run: INSTALL_ALL_DEPENDENCIES.bat
echo.
pause
