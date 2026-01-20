# 🔧 Fix Expo Startup Error

## **The Error You're Seeing**

```
Error: ENOENT: no such file or directory, mkdir '...\node:sea'
```

This is a **Windows path length** or **OneDrive sync** issue.

---

## **Quick Fix (Try This First)**

### **Step 1: Clean Everything**

```powershell
# Navigate to mobile directory
cd understanding_app/mobile

# Delete .expo folder
Remove-Item -Recurse -Force .expo -ErrorAction SilentlyContinue

# Delete node_modules cache
Remove-Item -Recurse -Force node_modules\.cache -ErrorAction SilentlyContinue

# Clear npm cache
npm cache clean --force
```

### **Step 2: Reinstall**

```powershell
npm install
```

### **Step 3: Start with Clear Cache**

```powershell
npm start -- --clear
```

---

## **Alternative Fix: Move Project Closer to Root**

The path is very long, which can cause issues on Windows:

**Current path:**
```
C:\Users\DJMcC\OneDrive\Desktop\next\next\understanding_app\mobile
```

**Better path:**
```
C:\dev\understanding-app\mobile
```

**Or even shorter:**
```
C:\ub\mobile
```

**To move:**
1. Copy `understanding_app` folder to shorter path
2. Update imports if needed
3. Test

---

## **OneDrive Fix (Recommended)**

If project is in OneDrive:

1. **Right-click project folder** → "Always keep on this device"
2. **Wait for OneDrive to sync**
3. **Then try starting again**

---

## **If Still Not Working**

Try **web version first** to test if app works:

```powershell
npm start
# Press 'w' for web
```

If web works, it's a mobile-specific issue. If web doesn't work, it's a code issue.

---

## **Final Solution: Use Shorter Path**

Move your entire project to a shorter path:

```powershell
# Create new location
mkdir C:\ub

# Copy project there
# Then update all paths
```

---

**Try the quick fix first - it usually works!** 🚀
