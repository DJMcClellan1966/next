@echo off
echo ============================================================
echo ML Toolbox - Quick Fix for Dependencies
echo ============================================================
echo.
echo This will install the most commonly needed packages
echo to eliminate the warnings you're seeing.
echo.
pause

echo.
echo Installing essential packages...
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn pandas psutil py-cpuinfo sentence-transformers imbalanced-learn statsmodels pgmpy hmmlearn shap lime h5py flask

echo.
echo ============================================================
echo Done!
echo ============================================================
echo.
echo Most warnings should now be resolved.
echo The ML Toolbox will work much better now!
echo.
pause
