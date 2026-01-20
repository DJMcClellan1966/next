# 📖 Understanding Bible App - Mobile

Cross-platform mobile app (iOS, Android, Windows) for deep Bible understanding.

---

## 🚀 **Quick Start**

### **1. Install Dependencies**

```bash
cd understanding_app/mobile
npm install
```

**OR if using Expo CLI:**

```bash
npx expo install
```

### **2. Start Backend**

In a separate terminal, start the backend:

```bash
cd understanding_app
python api.py
```

Backend will run on `http://localhost:8003`

### **3. Update API URL (if needed)**

Edit `app.json` to set your backend URL:

```json
{
  "extra": {
    "apiUrl": "http://your-server:8003"
  }
}
```

Or set environment variable:

```bash
export EXPO_PUBLIC_API_URL=http://your-server:8003
```

### **4. Start App**

```bash
# Start Expo
npm start
# or
npx expo start

# Then choose:
# - Press 'i' for iOS simulator
# - Press 'a' for Android emulator
# - Press 'w' for web
# - Press 'w' then 'w' for Windows
```

---

## 📱 **Build for Platforms**

### **iOS (requires Mac)**

```bash
npx expo build:ios
# or
eas build --platform ios
```

### **Android**

```bash
npx expo build:android
# or
eas build --platform android
```

### **Windows**

```bash
npx expo build:windows
# or
npm run windows
```

---

## 🏗️ **Project Structure**

```
mobile/
├── App.js                    # Main app entry
├── app.json                  # Expo config
├── package.json              # Dependencies
├── src/
│   ├── screens/              # App screens
│   │   ├── TodayScreen.js    # Daily understanding
│   │   ├── UnderstandingScreen.js  # Deep understanding view
│   │   ├── ScholarScreen.js  # Scholar voice generation
│   │   ├── DiscoverScreen.js # Verse search & connections
│   │   ├── JournalScreen.js  # Personal journal
│   │   └── SearchScreen.js   # Verse search
│   ├── services/
│   │   └── api.js            # API service
│   └── theme.js              # App theme
└── assets/                   # Images, icons
```

---

## 📋 **Features**

### **✅ Implemented:**
- ✅ Daily understanding screen
- ✅ Deep understanding view
- ✅ Scholar voice generation
- ✅ Verse search (semantic)
- ✅ Connection discovery
- ✅ Journal for insights
- ✅ Beautiful UI
- ✅ Cross-platform (iOS, Android, Windows)

### **🚧 TODO:**
- [ ] Offline mode (cached content)
- [ ] Push notifications (daily reminders)
- [ ] User authentication
- [ ] Cloud sync
- [ ] Reading plans (optional)
- [ ] Audio narration

---

## 🔧 **Configuration**

### **API URL**

Set in `app.json`:
```json
{
  "extra": {
    "apiUrl": "http://localhost:8003"
  }
}
```

Or use environment variable:
```bash
EXPO_PUBLIC_API_URL=http://your-server:8003 npm start
```

---

## 🎨 **Design**

- **Primary Color:** Blue (#2563eb)
- **Secondary Color:** Purple (#7c3aed)
- **Accent Color:** Amber (#f59e0b)
- **Clean, minimal design**
- **Readable typography**
- **Smooth navigation**

---

## 📦 **Dependencies**

- **expo** - React Native framework
- **react-navigation** - Navigation
- **react-native-paper** - UI components
- **axios** - HTTP client
- **expo-vector-icons** - Icons

---

## 🚀 **Deployment**

### **Using Expo EAS (Easiest)**

```bash
# Install EAS CLI
npm install -g eas-cli

# Login
eas login

# Build
eas build --platform all

# Submit to stores
eas submit --platform ios
eas submit --platform android
```

### **Manual Build**

See Expo documentation for platform-specific builds.

---

**Your cross-platform Bible app is ready!** 📱✨
