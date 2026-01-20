/**
 * Journal Screen
 * Personal insights and growth tracking
 */
import React, { useState, useEffect } from 'react';
import {
  View,
  ScrollView,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
} from 'react-native';
import { Card, Title, Paragraph, Button, FAB } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme';
import { getJournalEntries, saveJournalEntry } from '../services/api';

export default function JournalScreen({ navigation }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadEntries();
  }, []);

  const loadEntries = async () => {
    try {
      const data = await getJournalEntries();
      setEntries(data.entries || []);
    } catch (error) {
      console.error('Error loading entries:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewEntry = () => {
    // TODO: Navigate to new entry screen
    navigation.navigate('Search');
  };

  // Sample entries for demo (replace with actual API data)
  const sampleEntries = [
    {
      id: '1',
      verse_reference: 'Psalm 23:1',
      verse_text: 'The Lord is my shepherd...',
      insights: 'I realized today that God knows me personally, just like a shepherd knows each sheep.',
      date: '2024-01-20',
    },
    {
      id: '2',
      verse_reference: 'John 10:11',
      verse_text: 'I am the good shepherd...',
      insights: 'Jesus connects with David\'s metaphor. He is the ultimate shepherd who laid down his life.',
      date: '2024-01-19',
    },
  ];

  const displayEntries = entries.length > 0 ? entries : sampleEntries;

  return (
    <View style={styles.container}>
      {displayEntries.length > 0 ? (
        <FlatList
          data={displayEntries}
          keyExtractor={(item) => item.id || item.entry_id || Math.random().toString()}
          renderItem={({ item }) => (
            <Card style={styles.entryCard}>
              <Card.Content>
                <View style={styles.entryHeader}>
                  <Text style={styles.entryReference}>
                    {item.verse_reference}
                  </Text>
                  <Text style={styles.entryDate}>{item.date}</Text>
                </View>
                <Paragraph style={styles.entryVerse}>
                  "{item.verse_text || item.insights?.substring(0, 50)}..."
                </Paragraph>
                <Paragraph style={styles.entryInsights}>
                  {item.insights}
                </Paragraph>
                {item.questions && (
                  <View style={styles.questionsContainer}>
                    <Text style={styles.questionsLabel}>Questions:</Text>
                    <Text style={styles.questionsText}>{item.questions}</Text>
                  </View>
                )}
                <TouchableOpacity
                  style={styles.viewButton}
                  onPress={() => {
                    navigation.navigate('Understanding', {
                      verseReference: item.verse_reference,
                      verseText: item.verse_text,
                    });
                  }}
                >
                  <Text style={styles.viewButtonText}>View Understanding →</Text>
                </TouchableOpacity>
              </Card.Content>
            </Card>
          )}
          contentContainerStyle={styles.listContent}
          ListHeaderComponent={
            <View style={styles.header}>
              <Title style={styles.headerTitle}>My Journal</Title>
              <Paragraph style={styles.headerSubtitle}>
                Your personal insights and growth
              </Paragraph>
            </View>
          }
        />
      ) : (
        <ScrollView style={styles.emptyContainer}>
          <Ionicons
            name="journal-outline"
            size={64}
            color={theme.colors.inactive}
          />
          <Title style={styles.emptyTitle}>Your Journal</Title>
          <Paragraph style={styles.emptyText}>
            Save your insights, questions, and reflections here. Your
            understanding will grow over time.
          </Paragraph>
          <Button
            mode="contained"
            onPress={handleNewEntry}
            style={styles.newEntryButton}
            icon="plus"
          >
            Add Your First Entry
          </Button>
        </ScrollView>
      )}

      <FAB
        style={styles.fab}
        icon="plus"
        onPress={handleNewEntry}
        label="New Entry"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  listContent: {
    padding: theme.spacing.md,
  },
  header: {
    marginBottom: theme.spacing.lg,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: theme.spacing.xs,
  },
  headerSubtitle: {
    fontSize: 16,
    color: theme.colors.textSecondary,
  },
  entryCard: {
    marginBottom: theme.spacing.md,
    elevation: 2,
  },
  entryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
  },
  entryReference: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
  },
  entryDate: {
    fontSize: 14,
    color: theme.colors.textSecondary,
  },
  entryVerse: {
    fontSize: 15,
    fontStyle: 'italic',
    color: theme.colors.text,
    marginBottom: theme.spacing.sm,
  },
  entryInsights: {
    fontSize: 16,
    lineHeight: 24,
    color: theme.colors.text,
    marginBottom: theme.spacing.sm,
  },
  questionsContainer: {
    marginTop: theme.spacing.sm,
    padding: theme.spacing.sm,
    backgroundColor: theme.colors.background,
    borderRadius: theme.borderRadius.md,
  },
  questionsLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.textSecondary,
    marginBottom: theme.spacing.xs,
  },
  questionsText: {
    fontSize: 15,
    color: theme.colors.text,
  },
  viewButton: {
    marginTop: theme.spacing.sm,
    padding: theme.spacing.xs,
  },
  viewButtonText: {
    color: theme.colors.primary,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    padding: theme.spacing.xl,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: theme.spacing.md,
    marginBottom: theme.spacing.sm,
  },
  emptyText: {
    fontSize: 16,
    lineHeight: 24,
    color: theme.colors.textSecondary,
    textAlign: 'center',
    marginBottom: theme.spacing.xl,
  },
  newEntryButton: {
    marginTop: theme.spacing.md,
  },
  fab: {
    position: 'absolute',
    margin: theme.spacing.md,
    right: 0,
    bottom: 0,
    backgroundColor: theme.colors.primary,
  },
});
