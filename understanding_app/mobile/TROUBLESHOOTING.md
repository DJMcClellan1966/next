# 🔧 Troubleshooting Expo Mobile App

## **Common Issues & Fixes**

### **Issue 1: ENOENT Error on Startup**

**Error:**
```
Error: ENOENT: no such file or directory, mkdir '...\node:sea'
```

**Fix:**
1. **Clear Expo cache:**
   ```bash
   # Delete .expo folder
   rm -rf .expo
   # or on Windows
   rmdir /s /q .expo
   ```

2. **Clear npm cache:**
   ```bash
   npm cache clean --force
   ```

3. **Reinstall:**
   ```bash
   npm install
   ```

4. **Start with clear cache:**
   ```bash
   npm start -- --clear
   ```

---

### **Issue 2: Long Path Issue (Windows/OneDrive)**

**Problem:** Path is too long or OneDrive sync issues

**Fix:**
1. **Move project closer to root:**
   - Current: `C:\Users\DJMcC\OneDrive\Desktop\next\next\understanding_app\mobile`
   - Better: `C:\dev\understanding-app\mobile`

2. **Disable OneDrive for project folder:**
   - Right-click project folder
   - Choose "Always keep on this device"

3. **Enable long paths in Windows:**
   - Open Registry Editor
   - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`

---

### **Issue 3: Can't Connect to Backend**

**Problem:** Mobile app can't reach API

**Fix:**
1. **For iOS Simulator / Android Emulator:**
   ```json
   // app.json
   {
     "extra": {
       "apiUrl": "http://localhost:8003"
     }
   }
   ```

2. **For Physical Device:**
   ```json
   // app.json - use your computer's IP
   {
     "extra": {
       "apiUrl": "http://192.168.1.100:8003"
     }
   }
   ```
   
   Find your IP:
   ```bash
   # Windows
   ipconfig
   # Look for IPv4 Address
   ```

3. **Make sure backend is running:**
   ```bash
   cd understanding_app
   python api.py
   ```

4. **Test connection:**
   ```bash
   curl http://localhost:8003/api/health
   ```

---

### **Issue 4: Module Not Found**

**Error:**
```
Module not found: Can't resolve 'date-fns'
```

**Fix:**
```bash
npm install date-fns
# or
npm install
```

---

### **Issue 5: Expo Start Hangs**

**Fix:**
1. **Kill all Node processes:**
   ```bash
   # Windows
   taskkill /F /IM node.exe
   ```

2. **Clear everything:**
   ```bash
   rmdir /s /q .expo
   rmdir /s /q node_modules
   npm cache clean --force
   npm install
   ```

3. **Start fresh:**
   ```bash
   npm start -- --clear
   ```

---

### **Issue 6: Metro Bundler Issues**

**Fix:**
1. **Reset Metro bundler:**
   ```bash
   npm start -- --reset-cache
   ```

2. **Or clear watchman:**
   ```bash
   watchman watch-del-all
   ```

---

## 🚀 **Quick Fix Script**

Run `fix_expo.bat` (Windows) or manually:

```bash
# Clean everything
rm -rf .expo
rm -rf node_modules/.cache
npm cache clean --force

# Reinstall
npm install

# Start with clear cache
npm start -- --clear
```

---

## ✅ **Still Having Issues?**

1. **Check Node version:**
   ```bash
   node --version
   # Should be 16+ or 18+
   ```

2. **Check Expo CLI:**
   ```bash
   npx expo --version
   ```

3. **Try web version first:**
   ```bash
   npm start
   # Press 'w' for web
   # This tests if the app works without mobile issues
   ```

4. **Check backend:**
   - Make sure Python backend is running
   - Test API endpoint: `curl http://localhost:8003/api/health`

---

**Most common fix:** Clear cache and restart!

```bash
npm start -- --clear
```
