# 🚀 Understanding Bible App - Setup Guide

Complete setup guide for the Understanding Bible App (backend + mobile app).

---

## 📋 **Prerequisites**

- Python 3.9+
- Node.js 16+
- npm or yarn
- Expo CLI (optional, can use npx)

---

## 🔧 **Backend Setup**

### **1. Install Python Dependencies**

```bash
cd understanding_app
pip install -r ../requirements.txt
# Also install FastAPI if not already installed
pip install fastapi uvicorn requests
```

### **2. Start Backend**

```bash
python api.py
```

Backend will start on `http://localhost:8003` (or next available port)

### **3. Load Bible Data**

You'll need to load Bible verses. Create a script or use the API:

```python
import requests

verses = [
    {
        "reference": "Psalm 23:1",
        "text": "The Lord is my shepherd, I lack nothing.",
        "book": "Psalm",
        "chapter": 23,
        "verse": "1",
        "theme": "trust"
    },
    # ... more verses
]

response = requests.post(
    "http://localhost:8003/api/bible/add-content",
    json=verses
)
```

---

## 📱 **Mobile App Setup**

### **1. Install Dependencies**

```bash
cd understanding_app/mobile
npm install
```

### **2. Configure API URL**

Edit `app.json` to set your backend URL:

```json
{
  "extra": {
    "apiUrl": "http://localhost:8003"
  }
}
```

**For physical device testing:**
- Use your computer's IP address (e.g., `http://192.168.1.100:8003`)
- Make sure device and computer are on same network

### **3. Start App**

```bash
# Start Expo
npm start
# or
npx expo start

# Then:
# - Press 'i' for iOS
# - Press 'a' for Android  
# - Press 'w' for web
# - Press 'w' then 'w' for Windows
```

---

## 🎯 **Testing**

### **1. Test Backend**

```bash
# Health check
curl http://localhost:8003/api/health

# Test understanding generation
curl -X POST http://localhost:8003/api/understanding/generate \
  -H "Content-Type: application/json" \
  -d '{
    "verse_reference": "Psalm 23:1",
    "verse_text": "The Lord is my shepherd, I lack nothing.",
    "depth_level": "deep"
  }'
```

### **2. Test Mobile App**

1. Start backend
2. Start mobile app
3. Open "Today" tab
4. Should see daily verse with understanding

---

## 🚀 **Production Deployment**

### **Backend**

Deploy to:
- Heroku
- AWS (EC2, Lambda)
- Google Cloud
- Azure
- Your own server

Update mobile app `app.json` with production URL.

### **Mobile App**

**iOS:**
```bash
eas build --platform ios
eas submit --platform ios
```

**Android:**
```bash
eas build --platform android
eas submit --platform android
```

**Windows:**
```bash
npm run windows
# Build Windows package
```

---

## 📝 **API Endpoints**

### **Understanding**
- `POST /api/understanding/generate` - Generate deep understanding
- `POST /api/understanding/scholar-voice` - Scholar voice generation
- `POST /api/understanding/daily` - Daily understanding
- `POST /api/understanding/connections` - Discover connections

### **Bible**
- `POST /api/bible/add-content` - Load Bible verses
- `POST /api/bible/search` - Semantic verse search

### **Journal**
- `POST /api/journal/save` - Save journal entry
- `GET /api/journal/entries` - Get journal entries

### **Health**
- `GET /api/health` - Health check

---

## 🔍 **Troubleshooting**

### **Backend won't start**
- Check if port 8003 is available
- Verify Python dependencies installed
- Check logs for errors

### **Mobile app can't connect**
- Verify backend is running
- Check API URL in `app.json`
- For physical device: use IP address, not localhost
- Check firewall settings

### **Generation takes too long**
- Normal for long explanations (book-length can take minutes)
- Reduce `max_length` in API calls
- Use shorter lengths initially

---

## ✅ **Next Steps**

1. ✅ Backend running
2. ✅ Mobile app running
3. ✅ Load Bible data
4. ✅ Test all features
5. ✅ Customize UI/theme
6. ✅ Deploy to production

---

**Your Understanding Bible App is ready!** 📖✨
