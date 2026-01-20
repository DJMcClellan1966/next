@echo off
echo Fixing Expo setup...
echo.

REM Clean Expo cache
if exist .expo rmdir /s /q .expo
if exist node_modules\.cache rmdir /s /q node_modules\.cache

REM Reinstall if needed
echo Reinstalling dependencies...
call npm install

echo.
echo Done! Now try: npm start
pause
