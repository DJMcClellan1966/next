/**
 * Scholar Screen
 * Generate explanations in different scholar voices
 */
import React, { useState } from 'react';
import {
  View,
  ScrollView,
  Text,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { Card, Title, Paragraph, Button, Chip, Divider } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme';
import { generateScholarVoice } from '../services/api';

const SCHOLAR_STYLES = [
  { id: 'academic', name: 'Academic Scholar', icon: 'school', color: theme.colors.secondary },
  { id: 'devotional', name: 'Devotional Writer', icon: 'heart', color: theme.colors.accent },
  { id: 'pastor', name: 'Pastor/Teacher', icon: 'person', color: theme.colors.primary },
  { id: 'theologian', name: 'Theologian', icon: 'library', color: theme.colors.success },
];

const LENGTHS = [
  { id: 'short', name: 'Short (500 words)', words: 500 },
  { id: 'medium', name: 'Medium (1500 words)', words: 1500 },
  { id: 'long', name: 'Long (3000 words)', words: 3000 },
  { id: 'book', name: 'Book Length (8000+ words)', words: 8000 },
];

export default function ScholarScreen({ route, navigation }) {
  const { verseReference: initialVerseRef, verseText: initialVerseText } = route.params || {};
  
  const [verseReference, setVerseReference] = useState(initialVerseRef || '');
  const [verseText, setVerseText] = useState(initialVerseText || '');
  const [selectedStyle, setSelectedStyle] = useState('devotional');
  const [selectedLength, setSelectedLength] = useState('medium');
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);

  const handleGenerate = async () => {
    if (!verseReference || !verseText) {
      Alert.alert('Error', 'Please enter a verse reference and text');
      return;
    }

    try {
      setGenerating(true);
      setResult(null);
      
      const data = await generateScholarVoice(
        verseReference,
        verseText,
        selectedStyle,
        selectedLength
      );
      
      setResult(data);
    } catch (error) {
      Alert.alert('Error', 'Failed to generate scholar explanation. Make sure the backend is running.');
      console.error(error);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Input Section */}
      <Card style={styles.inputCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>Verse to Understand</Title>
          <Paragraph style={styles.inputLabel}>Reference (e.g., Psalm 23:1)</Paragraph>
          <View style={styles.inputContainer}>
            <Text style={styles.inputText}>{verseReference || 'Enter verse reference...'}</Text>
          </View>
          {initialVerseRef && (
            <Text style={styles.verseText}>{initialVerseText}</Text>
          )}
          {!initialVerseRef && (
            <TouchableOpacity
              style={styles.searchButton}
              onPress={() => navigation.navigate('Search')}
            >
              <Ionicons name="search" size={20} color={theme.colors.primary} />
              <Text style={styles.searchButtonText}>Search for a verse</Text>
            </TouchableOpacity>
          )}
        </Card.Content>
      </Card>

      {/* Scholar Style Selection */}
      <Card style={styles.selectionCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>Scholar Style</Title>
          <View style={styles.chipContainer}>
            {SCHOLAR_STYLES.map((style) => (
              <Chip
                key={style.id}
                selected={selectedStyle === style.id}
                onPress={() => setSelectedStyle(style.id)}
                style={[
                  styles.chip,
                  selectedStyle === style.id && { backgroundColor: style.color + '20' },
                ]}
                icon={({ size, color }) => (
                  <Ionicons name={style.icon} size={size} color={selectedStyle === style.id ? style.color : color} />
                )}
              >
                {style.name}
              </Chip>
            ))}
          </View>
        </Card.Content>
      </Card>

      {/* Length Selection */}
      <Card style={styles.selectionCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>Explanation Length</Title>
          <View style={styles.chipContainer}>
            {LENGTHS.map((length) => (
              <Chip
                key={length.id}
                selected={selectedLength === length.id}
                onPress={() => setSelectedLength(length.id)}
                style={[
                  styles.chip,
                  selectedLength === length.id && { backgroundColor: theme.colors.primary + '20' },
                ]}
              >
                {length.name}
              </Chip>
            ))}
          </View>
        </Card.Content>
      </Card>

      {/* Generate Button */}
      <View style={styles.generateSection}>
        <Button
          mode="contained"
          onPress={handleGenerate}
          loading={generating}
          disabled={generating || !verseReference || !verseText}
          style={styles.generateButton}
          icon="sparkles"
        >
          Generate Scholar Explanation
        </Button>
        {selectedLength === 'book' && (
          <Text style={styles.warningText}>
            ⚠️ Book-length generation may take several minutes
          </Text>
        )}
      </View>

      {/* Result */}
      {result && (
        <Card style={styles.resultCard}>
          <Card.Content>
            <View style={styles.resultHeader}>
              <Ionicons name="checkmark-circle" size={24} color={theme.colors.success} />
              <Title style={styles.resultTitle}>
                {SCHOLAR_STYLES.find(s => s.id === selectedStyle)?.name} Explanation
              </Title>
            </View>
            <Text style={styles.resultMeta}>
              {verseReference} • {result.length} characters
            </Text>
            <Divider style={styles.divider} />
            <ScrollView style={styles.resultTextContainer}>
              <Paragraph style={styles.resultText}>
                {result.explanation}
              </Paragraph>
            </ScrollView>
          </Card.Content>
        </Card>
      )}

      {/* Generating Indicator */}
      {generating && (
        <View style={styles.generatingContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.generatingText}>
            Generating {selectedLength} explanation in {SCHOLAR_STYLES.find(s => s.id === selectedStyle)?.name} style...
          </Text>
          <Text style={styles.generatingSubtext}>
            This may take a minute for longer explanations
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  inputCard: {
    margin: theme.spacing.md,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: theme.spacing.sm,
  },
  inputLabel: {
    fontSize: 14,
    color: theme.colors.textSecondary,
    marginBottom: theme.spacing.xs,
  },
  inputContainer: {
    padding: theme.spacing.md,
    backgroundColor: theme.colors.background,
    borderRadius: theme.borderRadius.md,
    marginBottom: theme.spacing.sm,
  },
  inputText: {
    fontSize: 16,
    color: theme.colors.text,
  },
  verseText: {
    fontSize: 16,
    fontStyle: 'italic',
    color: theme.colors.text,
    marginTop: theme.spacing.sm,
    padding: theme.spacing.sm,
    backgroundColor: theme.colors.background,
    borderRadius: theme.borderRadius.md,
  },
  searchButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: theme.spacing.sm,
    marginTop: theme.spacing.sm,
  },
  searchButtonText: {
    marginLeft: theme.spacing.xs,
    color: theme.colors.primary,
    fontWeight: '600',
  },
  selectionCard: {
    margin: theme.spacing.md,
    marginTop: 0,
    elevation: 2,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
  },
  chip: {
    marginBottom: theme.spacing.xs,
  },
  generateSection: {
    padding: theme.spacing.md,
  },
  generateButton: {
    marginBottom: theme.spacing.sm,
  },
  warningText: {
    fontSize: 12,
    color: theme.colors.warning,
    textAlign: 'center',
    marginTop: theme.spacing.xs,
  },
  resultCard: {
    margin: theme.spacing.md,
    marginTop: 0,
    elevation: 2,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  resultTitle: {
    marginLeft: theme.spacing.sm,
    fontSize: 18,
    fontWeight: '600',
  },
  resultMeta: {
    fontSize: 14,
    color: theme.colors.textSecondary,
    marginBottom: theme.spacing.sm,
  },
  divider: {
    marginVertical: theme.spacing.sm,
  },
  resultTextContainer: {
    maxHeight: 800,
  },
  resultText: {
    fontSize: 16,
    lineHeight: 26,
    color: theme.colors.text,
  },
  generatingContainer: {
    padding: theme.spacing.xl,
    alignItems: 'center',
  },
  generatingText: {
    marginTop: theme.spacing.md,
    fontSize: 16,
    color: theme.colors.text,
    textAlign: 'center',
  },
  generatingSubtext: {
    marginTop: theme.spacing.sm,
    fontSize: 14,
    color: theme.colors.textSecondary,
    textAlign: 'center',
  },
});
