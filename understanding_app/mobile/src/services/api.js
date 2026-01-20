/**
 * API Service
 * Handles all API calls to the Understanding Bible backend
 */
import axios from 'axios';
import Constants from 'expo-constants';

// Get API URL from environment or use default
const API_URL = Constants.expoConfig?.extra?.apiUrl || 'http://localhost:8003';

const api = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30 seconds (for long generations)
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Generate deep understanding of a verse
 */
export async function generateUnderstanding(verseReference, verseText, depthLevel = 'deep', scholarStyle = null) {
  try {
    const response = await api.post('/api/understanding/generate', {
      verse_reference: verseReference,
      verse_text: verseText,
      depth_level: depthLevel,
      scholar_style: scholarStyle,
    });
    return response.data;
  } catch (error) {
    console.error('Error generating understanding:', error);
    throw error;
  }
}

/**
 * Generate explanation in scholar's voice
 */
export async function generateScholarVoice(verseReference, verseText, scholarStyle, length = 'medium') {
  try {
    const response = await api.post('/api/understanding/scholar-voice', {
      verse_reference: verseReference,
      verse_text: verseText,
      scholar_style: scholarStyle,
      length: length,
    });
    return response.data;
  } catch (error) {
    console.error('Error generating scholar voice:', error);
    throw error;
  }
}

/**
 * Get daily understanding
 */
export async function getDailyUnderstanding(date = null) {
  try {
    const response = await api.post('/api/understanding/daily', {
      date: date,
    });
    return response.data;
  } catch (error) {
    console.error('Error getting daily understanding:', error);
    throw error;
  }
}

/**
 * Discover connections between verses
 */
export async function discoverConnections(verseReference, verseText, topK = 10) {
  try {
    const response = await api.post('/api/understanding/connections', {
      verse_reference: verseReference,
      verse_text: verseText,
      top_k: topK,
    });
    return response.data;
  } catch (error) {
    console.error('Error discovering connections:', error);
    throw error;
  }
}

/**
 * Search verses semantically
 */
export async function searchVerses(query, topK = 10) {
  try {
    const response = await api.post('/api/understanding/search', {
      verse_reference: 'Search',
      verse_text: query,
      top_k: topK,
    });
    return response.data;
  } catch (error) {
    console.error('Error searching verses:', error);
    throw error;
  }
}

/**
 * Save journal entry
 */
export async function saveJournalEntry(entry) {
  try {
    const response = await api.post('/api/journal/save', entry);
    return response.data;
  } catch (error) {
    console.error('Error saving journal entry:', error);
    throw error;
  }
}

/**
 * Get journal entries
 */
export async function getJournalEntries(userId = null) {
  try {
    const response = await api.get('/api/journal/entries', {
      params: { user_id: userId },
    });
    return response.data;
  } catch (error) {
    console.error('Error getting journal entries:', error);
    throw error;
  }
}

/**
 * Health check
 */
export async function checkHealth() {
  try {
    const response = await api.get('/api/health');
    return response.data;
  } catch (error) {
    console.error('Error checking health:', error);
    return { status: 'error', message: error.message };
  }
}

export default api;
