/**
 * Today Screen
 * Daily verse with deep understanding - no reading plan pressure
 */
import React, { useState, useEffect } from 'react';
import {
  View,
  ScrollView,
  Text,
  StyleSheet,
  ActivityIndicator,
  RefreshControl,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { Card, Button, Paragraph, Title } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme';
import { getDailyUnderstanding, saveJournalEntry } from '../services/api';
import { format } from 'date-fns';

export default function TodayScreen({ navigation }) {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [dailyData, setDailyData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDailyUnderstanding();
  }, []);

  const loadDailyUnderstanding = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getDailyUnderstanding();
      setDailyData(data);
    } catch (err) {
      setError('Failed to load daily understanding. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadDailyUnderstanding();
  };

  const handleDeepUnderstanding = () => {
    if (dailyData) {
      navigation.navigate('Understanding', {
        verseReference: dailyData.verse_reference,
        verseText: dailyData.verse_text,
        understanding: dailyData.understanding,
        relatedVerses: dailyData.related_verses,
      });
    }
  };

  const handleSaveInsight = async () => {
    try {
      await saveJournalEntry({
        verse_reference: dailyData.verse_reference,
        insights: `Daily understanding for ${dailyData.date}`,
      });
      Alert.alert('Success', 'Insight saved to journal!');
    } catch (err) {
      Alert.alert('Error', 'Failed to save insight');
    }
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading today's understanding...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Ionicons name="alert-circle" size={64} color={theme.colors.error} />
        <Text style={styles.errorText}>{error}</Text>
        <Button
          mode="contained"
          onPress={loadDailyUnderstanding}
          style={styles.retryButton}
        >
          Retry
        </Button>
      </View>
    );
  }

  if (!dailyData) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>No data available</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Date Header */}
      <View style={styles.dateHeader}>
        <Text style={styles.dateText}>
          {format(new Date(dailyData.date), 'EEEE, MMMM d, yyyy')}
        </Text>
      </View>

      {/* Verse Card */}
      <Card style={styles.verseCard}>
        <Card.Content>
          <View style={styles.verseHeader}>
            <Ionicons name="book" size={24} color={theme.colors.primary} />
            <Title style={styles.verseReference}>
              {dailyData.verse_reference}
            </Title>
          </View>
          <Paragraph style={styles.verseText}>
            "{dailyData.verse_text}"
          </Paragraph>
        </Card.Content>
      </Card>

      {/* Understanding Preview */}
      <Card style={styles.understandingCard}>
        <Card.Content>
          <View style={styles.cardHeader}>
            <Ionicons name="bulb" size={24} color={theme.colors.accent} />
            <Title style={styles.cardTitle}>Deep Understanding</Title>
          </View>
          <Paragraph style={styles.understandingPreview} numberOfLines={4}>
            {dailyData.understanding?.explanation || 
             dailyData.understanding?.context || 
             'Tap below for deep understanding...'}
          </Paragraph>
          <TouchableOpacity
            onPress={handleDeepUnderstanding}
            style={styles.readMoreButton}
          >
            <Text style={styles.readMoreText}>Read Full Understanding →</Text>
          </TouchableOpacity>
        </Card.Content>
      </Card>

      {/* Related Verses */}
      {dailyData.related_verses && dailyData.related_verses.length > 0 && (
        <Card style={styles.relatedCard}>
          <Card.Content>
            <View style={styles.cardHeader}>
              <Ionicons name="link" size={24} color={theme.colors.secondary} />
              <Title style={styles.cardTitle}>Related Verses</Title>
            </View>
            {dailyData.related_verses.slice(0, 3).map((verse, index) => (
              <TouchableOpacity
                key={index}
                style={styles.relatedVerseItem}
                onPress={() => {
                  navigation.navigate('Understanding', {
                    verseReference: verse.reference,
                    verseText: verse.text,
                  });
                }}
              >
                <Text style={styles.relatedReference}>{verse.reference}</Text>
                <Text style={styles.relatedText} numberOfLines={2}>
                  {verse.text}
                </Text>
              </TouchableOpacity>
            ))}
          </Card.Content>
        </Card>
      )}

      {/* Reflection Questions */}
      {dailyData.reflection_questions && dailyData.reflection_questions.length > 0 && (
        <Card style={styles.reflectionCard}>
          <Card.Content>
            <View style={styles.cardHeader}>
              <Ionicons name="create" size={24} color={theme.colors.primary} />
              <Title style={styles.cardTitle}>Reflection Questions</Title>
            </View>
            {dailyData.reflection_questions.map((question, index) => (
              <View key={index} style={styles.questionItem}>
                <Text style={styles.questionText}>{question}</Text>
              </View>
            ))}
          </Card.Content>
        </Card>
      )}

      {/* Action Buttons */}
      <View style={styles.actionButtons}>
        <Button
          mode="contained"
          onPress={handleDeepUnderstanding}
          style={styles.actionButton}
          icon="book-open"
        >
          Full Understanding
        </Button>
        <Button
          mode="outlined"
          onPress={handleSaveInsight}
          style={styles.actionButton}
          icon="bookmark"
        >
          Save to Journal
        </Button>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.lg,
  },
  loadingText: {
    marginTop: theme.spacing.md,
    color: theme.colors.textSecondary,
  },
  errorText: {
    marginTop: theme.spacing.md,
    color: theme.colors.error,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: theme.spacing.md,
  },
  dateHeader: {
    padding: theme.spacing.md,
    backgroundColor: theme.colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border,
  },
  dateText: {
    fontSize: 18,
    fontWeight: '600',
    color: theme.colors.text,
  },
  verseCard: {
    margin: theme.spacing.md,
    elevation: 2,
  },
  verseHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
  },
  verseReference: {
    marginLeft: theme.spacing.sm,
    fontSize: 20,
    fontWeight: 'bold',
    color: theme.colors.primary,
  },
  verseText: {
    fontSize: 18,
    lineHeight: 28,
    color: theme.colors.text,
    fontStyle: 'italic',
  },
  understandingCard: {
    margin: theme.spacing.md,
    marginTop: 0,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
  },
  cardTitle: {
    marginLeft: theme.spacing.sm,
    fontSize: 18,
    fontWeight: '600',
  },
  understandingPreview: {
    fontSize: 16,
    lineHeight: 24,
    color: theme.colors.text,
    marginBottom: theme.spacing.sm,
  },
  readMoreButton: {
    padding: theme.spacing.sm,
  },
  readMoreText: {
    color: theme.colors.primary,
    fontWeight: '600',
  },
  relatedCard: {
    margin: theme.spacing.md,
    marginTop: 0,
    elevation: 2,
  },
  relatedVerseItem: {
    padding: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
    backgroundColor: theme.colors.background,
    borderRadius: theme.borderRadius.md,
  },
  relatedReference: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.primary,
    marginBottom: 4,
  },
  relatedText: {
    fontSize: 14,
    color: theme.colors.textSecondary,
  },
  reflectionCard: {
    margin: theme.spacing.md,
    marginTop: 0,
    elevation: 2,
  },
  questionItem: {
    padding: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
    backgroundColor: theme.colors.background,
    borderRadius: theme.borderRadius.md,
  },
  questionText: {
    fontSize: 16,
    lineHeight: 24,
    color: theme.colors.text,
  },
  actionButtons: {
    padding: theme.spacing.md,
    gap: theme.spacing.sm,
  },
  actionButton: {
    marginBottom: theme.spacing.sm,
  },
});
