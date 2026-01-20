/**
 * Discover Screen
 * Semantic search and connection discovery
 */
import React, { useState } from 'react';
import {
  View,
  TextInput,
  ScrollView,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { Card, Title, Paragraph, Button, Searchbar } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme';
import { searchVerses, discoverConnections } from '../services/api';

export default function DiscoverScreen({ navigation }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState([]);
  const [selectedVerse, setSelectedVerse] = useState(null);
  const [connections, setConnections] = useState([]);
  const [loadingConnections, setLoadingConnections] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      setSearching(true);
      setResults([]);
      const data = await searchVerses(searchQuery);
      setResults(data.connections || []);
    } catch (error) {
      console.error('Error searching:', error);
    } finally {
      setSearching(false);
    }
  };

  const handleVerseSelect = async (verse) => {
    setSelectedVerse(verse);
    try {
      setLoadingConnections(true);
      const data = await discoverConnections(
        verse.reference,
        verse.text || verse.verse_text,
        10
      );
      setConnections(data.connections || []);
    } catch (error) {
      console.error('Error loading connections:', error);
    } finally {
      setLoadingConnections(false);
    }
  };

  const handleViewUnderstanding = (verse) => {
    navigation.navigate('Understanding', {
      verseReference: verse.reference || verse.verse_reference,
      verseText: verse.text || verse.verse_text,
    });
  };

  return (
    <View style={styles.container}>
      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search by meaning, not just keywords..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          onSubmitEditing={handleSearch}
          style={styles.searchbar}
        />
        <Button
          mode="contained"
          onPress={handleSearch}
          loading={searching}
          style={styles.searchButton}
          icon="search"
        >
          Search
        </Button>
      </View>

      <ScrollView style={styles.scrollView}>
        {/* Search Results */}
        {results.length > 0 && (
          <View style={styles.resultsSection}>
            <Title style={styles.sectionTitle}>
              Search Results ({results.length})
            </Title>
            {results.map((verse, index) => (
              <Card
                key={index}
                style={styles.verseCard}
                onPress={() => handleVerseSelect(verse)}
              >
                <Card.Content>
                  <View style={styles.verseCardHeader}>
                    <Text style={styles.verseReference}>
                      {verse.reference || verse.verse_reference}
                    </Text>
                    <Ionicons
                      name="arrow-forward-circle"
                      size={24}
                      color={theme.colors.primary}
                    />
                  </View>
                  <Paragraph style={styles.verseText} numberOfLines={2}>
                    {verse.text || verse.verse_text}
                  </Paragraph>
                  <Text style={styles.similarityText}>
                    Relevance: {(verse.similarity * 100).toFixed(0)}%
                  </Text>
                </Card.Content>
              </Card>
            ))}
          </View>
        )}

        {/* Connections */}
        {selectedVerse && (
          <View style={styles.connectionsSection}>
            <Title style={styles.sectionTitle}>
              Connections for {selectedVerse.reference || selectedVerse.verse_reference}
            </Title>
            {loadingConnections ? (
              <ActivityIndicator size="large" color={theme.colors.primary} />
            ) : connections.length > 0 ? (
              connections.map((verse, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.connectionItem}
                  onPress={() => handleViewUnderstanding(verse)}
                >
                  <View style={styles.connectionHeader}>
                    <Ionicons
                      name="link"
                      size={20}
                      color={theme.colors.secondary}
                    />
                    <Text style={styles.connectionReference}>
                      {verse.verse_reference || verse.reference}
                    </Text>
                  </View>
                  <Text style={styles.connectionText} numberOfLines={2}>
                    {verse.verse_text || verse.text}
                  </Text>
                  {verse.connection_explanation && (
                    <Text style={styles.connectionExplanation}>
                      {verse.connection_explanation}
                    </Text>
                  )}
                  <Text style={styles.connectionSimilarity}>
                    Similarity: {(verse.similarity * 100).toFixed(0)}%
                  </Text>
                </TouchableOpacity>
              ))
            ) : (
              <Paragraph style={styles.emptyText}>
                No connections found. Try selecting a different verse.
              </Paragraph>
            )}
          </View>
        )}

        {/* Info Card */}
        {results.length === 0 && !searching && (
          <Card style={styles.infoCard}>
            <Card.Content>
              <Ionicons
                name="compass"
                size={48}
                color={theme.colors.primary}
                style={styles.infoIcon}
              />
              <Title style={styles.infoTitle}>Discover Connections</Title>
              <Paragraph style={styles.infoText}>
                Search for Bible verses by meaning, not just keywords. The AI will
                find verses that are related in meaning, even if they don't share
                the same words.
              </Paragraph>
              <Paragraph style={styles.infoText}>
                Try searching for concepts like:
              </Paragraph>
              <View style={styles.exampleContainer}>
                {['forgiveness', 'God\'s love', 'trust in trials', 'hope'].map(
                  (example, index) => (
                    <TouchableOpacity
                      key={index}
                      style={styles.exampleChip}
                      onPress={() => {
                        setSearchQuery(example);
                        handleSearch();
                      }}
                    >
                      <Text style={styles.exampleText}>{example}</Text>
                    </TouchableOpacity>
                  )
                )}
              </View>
            </Card.Content>
          </Card>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  searchContainer: {
    padding: theme.spacing.md,
    backgroundColor: theme.colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border,
  },
  searchbar: {
    marginBottom: theme.spacing.sm,
  },
  searchButton: {
    marginTop: theme.spacing.xs,
  },
  scrollView: {
    flex: 1,
  },
  resultsSection: {
    padding: theme.spacing.md,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: theme.spacing.md,
  },
  verseCard: {
    marginBottom: theme.spacing.sm,
    elevation: 2,
  },
  verseCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  verseReference: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
  },
  verseText: {
    fontSize: 15,
    color: theme.colors.text,
    marginBottom: theme.spacing.xs,
  },
  similarityText: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    marginTop: theme.spacing.xs,
  },
  connectionsSection: {
    padding: theme.spacing.md,
    paddingTop: 0,
  },
  connectionItem: {
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius.md,
    borderLeftWidth: 3,
    borderLeftColor: theme.colors.secondary,
    elevation: 1,
  },
  connectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  connectionReference: {
    marginLeft: theme.spacing.xs,
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.secondary,
  },
  connectionText: {
    fontSize: 15,
    color: theme.colors.text,
    marginBottom: theme.spacing.xs,
  },
  connectionExplanation: {
    fontSize: 13,
    color: theme.colors.textSecondary,
    fontStyle: 'italic',
    marginBottom: theme.spacing.xs,
  },
  connectionSimilarity: {
    fontSize: 12,
    color: theme.colors.textSecondary,
  },
  emptyText: {
    color: theme.colors.textSecondary,
    fontStyle: 'italic',
    textAlign: 'center',
    padding: theme.spacing.lg,
  },
  infoCard: {
    margin: theme.spacing.md,
    elevation: 2,
  },
  infoIcon: {
    alignSelf: 'center',
    marginBottom: theme.spacing.md,
  },
  infoTitle: {
    textAlign: 'center',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: theme.spacing.md,
  },
  infoText: {
    fontSize: 16,
    lineHeight: 24,
    color: theme.colors.text,
    marginBottom: theme.spacing.md,
    textAlign: 'center',
  },
  exampleContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: theme.spacing.sm,
  },
  exampleChip: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    backgroundColor: theme.colors.primary + '20',
    borderRadius: theme.borderRadius.lg,
  },
  exampleText: {
    color: theme.colors.primary,
    fontWeight: '600',
  },
});
