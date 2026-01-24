@echo off
echo ============================================================
echo ML Toolbox - Installing All Dependencies
echo ============================================================
echo.
echo This will install all required packages for ML Toolbox.
echo This may take 5-10 minutes depending on your internet speed.
echo.
pause

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...
python -m pip install numpy scikit-learn pandas

echo.
echo Installing advanced ML libraries...
python -m pip install imbalanced-learn statsmodels xgboost lightgbm

echo.
echo Installing NLP libraries...
python -m pip install sentence-transformers transformers

echo.
echo Installing interpretability tools...
python -m pip install shap lime

echo.
echo Installing probabilistic models...
python -m pip install pgmpy hmmlearn

echo.
echo Installing system monitoring...
python -m pip install psutil py-cpuinfo

echo.
echo Installing data storage...
python -m pip install h5py joblib

echo.
echo Installing web framework...
python -m pip install flask flask-cors

echo.
echo Installing visualization...
python -m pip install matplotlib seaborn plotly

echo.
echo Installing utilities...
python -m pip install tqdm python-dateutil

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo All dependencies have been installed.
echo You can now run the ML Toolbox without warnings.
echo.
pause
