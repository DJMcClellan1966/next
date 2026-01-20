/**
 * Search Screen
 * Verse search and selection
 */
import React, { useState } from 'react';
import {
  View,
  FlatList,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { Searchbar, Card, Paragraph } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme';
import { searchVerses } from '../services/api';

export default function SearchScreen({ navigation, route }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      setSearching(true);
      const data = await searchVerses(searchQuery);
      setResults(data.connections || []);
    } catch (error) {
      console.error('Error searching:', error);
    } finally {
      setSearching(false);
    }
  };

  const handleVerseSelect = (verse) => {
    const onSelect = route.params?.onSelect;
    if (onSelect) {
      onSelect(verse);
      navigation.goBack();
    } else {
      navigation.navigate('Understanding', {
        verseReference: verse.reference || verse.verse_reference,
        verseText: verse.text || verse.verse_text,
      });
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search verses by meaning..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          onSubmitEditing={handleSearch}
          style={styles.searchbar}
        />
      </View>

      {searching ? (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>Searching...</Text>
        </View>
      ) : results.length > 0 ? (
        <FlatList
          data={results}
          keyExtractor={(item, index) => index.toString()}
          renderItem={({ item }) => (
            <TouchableOpacity onPress={() => handleVerseSelect(item)}>
              <Card style={styles.resultCard}>
                <Card.Content>
                  <View style={styles.resultHeader}>
                    <Text style={styles.resultReference}>
                      {item.reference || item.verse_reference}
                    </Text>
                    <Ionicons
                      name="chevron-forward"
                      size={20}
                      color={theme.colors.primary}
                    />
                  </View>
                  <Paragraph style={styles.resultText} numberOfLines={3}>
                    {item.text || item.verse_text}
                  </Paragraph>
                  {item.similarity && (
                    <Text style={styles.similarityText}>
                      Relevance: {(item.similarity * 100).toFixed(0)}%
                    </Text>
                  )}
                </Card.Content>
              </Card>
            </TouchableOpacity>
          )}
          contentContainerStyle={styles.listContent}
        />
      ) : (
        <View style={styles.centerContainer}>
          <Ionicons name="search" size={64} color={theme.colors.inactive} />
          <Text style={styles.emptyText}>
            Search for verses by meaning or keywords
          </Text>
        </View>
      )}
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
  },
  searchbar: {
    elevation: 2,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.xl,
  },
  loadingText: {
    marginTop: theme.spacing.md,
    color: theme.colors.textSecondary,
  },
  emptyText: {
    marginTop: theme.spacing.md,
    color: theme.colors.textSecondary,
    textAlign: 'center',
    fontSize: 16,
  },
  listContent: {
    padding: theme.spacing.md,
  },
  resultCard: {
    marginBottom: theme.spacing.sm,
    elevation: 2,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  resultReference: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
  },
  resultText: {
    fontSize: 15,
    color: theme.colors.text,
    marginBottom: theme.spacing.xs,
  },
  similarityText: {
    fontSize: 12,
    color: theme.colors.textSecondary,
  },
});
