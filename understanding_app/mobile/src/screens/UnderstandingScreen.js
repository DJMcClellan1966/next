/**
 * Understanding Screen
 * Deep understanding view with full explanation
 */
import React, { useState } from 'react';
import {
  View,
  ScrollView,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Share,
} from 'react-native';
import { Card, Title, Paragraph, Button, Divider } from 'react-native-paper';
import { View as RNView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme';
import { generateUnderstanding, discoverConnections } from '../services/api';

export default function UnderstandingScreen({ route, navigation }) {
  const { verseReference, verseText, understanding: initialUnderstanding, relatedVerses: initialRelatedVerses } = route.params || {};
  
  const [understanding, setUnderstanding] = useState(initialUnderstanding);
  const [relatedVerses, setRelatedVerses] = useState(initialRelatedVerses || []);
  const [loading, setLoading] = useState(!initialUnderstanding);
  const [loadingConnections, setLoadingConnections] = useState(false);

  React.useEffect(() => {
    if (!initialUnderstanding && verseReference && verseText) {
      loadUnderstanding();
    }
  }, [verseReference, verseText]);

  const loadUnderstanding = async () => {
    try {
      setLoading(true);
      const data = await generateUnderstanding(verseReference, verseText, 'deep');
      setUnderstanding(data.understanding);
      setRelatedVerses(data.related_verses || []);
    } catch (error) {
      console.error('Error loading understanding:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadConnections = async () => {
    try {
      setLoadingConnections(true);
      const data = await discoverConnections(verseReference, verseText);
      setRelatedVerses(data.connections || []);
    } catch (error) {
      console.error('Error loading connections:', error);
    } finally {
      setLoadingConnections(false);
    }
  };

  const handleShare = async () => {
    try {
      const shareText = `${verseReference}\n"${verseText}"\n\n${understanding?.explanation?.substring(0, 200)}...`;
      await Share.share({
        message: shareText,
        title: `Understanding: ${verseReference}`,
      });
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Generating deep understanding...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Verse Header */}
      <Card style={styles.verseCard}>
        <Card.Content>
          <View style={styles.verseHeader}>
            <Ionicons name="book" size={28} color={theme.colors.primary} />
            <Title style={styles.verseReference}>{verseReference}</Title>
          </View>
          <Paragraph style={styles.verseText}>"{verseText}"</Paragraph>
          <TouchableOpacity onPress={handleShare} style={styles.shareButton}>
            <Ionicons name="share-outline" size={20} color={theme.colors.primary} />
            <Text style={styles.shareText}>Share</Text>
          </TouchableOpacity>
        </Card.Content>
      </Card>

      {/* Deep Explanation */}
      {understanding?.explanation && (
        <Card style={styles.sectionCard}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <Ionicons name="bulb" size={24} color={theme.colors.accent} />
              <Title style={styles.sectionTitle}>Deep Explanation</Title>
            </View>
            <Paragraph style={styles.sectionText}>
              {understanding.explanation}
            </Paragraph>
          </Card.Content>
        </Card>
      )}

      {/* Historical Context */}
      {understanding?.context && (
        <Card style={styles.sectionCard}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <Ionicons name="time" size={24} color={theme.colors.secondary} />
              <Title style={styles.sectionTitle}>Historical Context</Title>
            </View>
            <Paragraph style={styles.sectionText}>
              {understanding.context}
            </Paragraph>
          </Card.Content>
        </Card>
      )}

      {/* Application */}
      {understanding?.application && (
        <Card style={styles.sectionCard}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <Ionicons name="heart" size={24} color={theme.colors.success} />
              <Title style={styles.sectionTitle}>Application</Title>
            </View>
            <Paragraph style={styles.sectionText}>
              {understanding.application}
            </Paragraph>
          </Card.Content>
        </Card>
      )}

      {/* Related Verses */}
      <Card style={styles.sectionCard}>
        <Card.Content>
          <View style={styles.sectionHeader}>
            <Ionicons name="link" size={24} color={theme.colors.secondary} />
            <Title style={styles.sectionTitle}>Related Verses</Title>
            <TouchableOpacity
              onPress={loadConnections}
              style={styles.refreshButton}
            >
              {loadingConnections ? (
                <ActivityIndicator size="small" color={theme.colors.primary} />
              ) : (
                <Ionicons name="refresh" size={20} color={theme.colors.primary} />
              )}
            </TouchableOpacity>
          </View>
          {relatedVerses.length > 0 ? (
            relatedVerses.map((verse, index) => (
              <TouchableOpacity
                key={index}
                style={styles.relatedVerseItem}
                onPress={() => {
                  navigation.push('Understanding', {
                    verseReference: verse.verse_reference || verse.reference,
                    verseText: verse.verse_text || verse.text,
                  });
                }}
              >
                <Text style={styles.relatedReference}>
                  {verse.verse_reference || verse.reference}
                </Text>
                <Text style={styles.relatedText} numberOfLines={2}>
                  {verse.verse_text || verse.text}
                </Text>
                {verse.connection_explanation && (
                  <Text style={styles.connectionText}>
                    {verse.connection_explanation}
                  </Text>
                )}
              </TouchableOpacity>
            ))
          ) : (
            <Paragraph style={styles.emptyText}>
              No related verses found. Tap refresh to discover connections.
            </Paragraph>
          )}
        </Card.Content>
      </Card>

      {/* Action Buttons */}
      <View style={styles.actions}>
        <Button
          mode="contained"
          onPress={() => {
            navigation.navigate('Scholar', {
              verseReference,
              verseText,
            });
          }}
          style={styles.actionButton}
          icon="school"
        >
          Scholar Voice
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
    fontSize: 22,
    fontWeight: 'bold',
    color: theme.colors.primary,
  },
  verseText: {
    fontSize: 20,
    lineHeight: 30,
    color: theme.colors.text,
    fontStyle: 'italic',
    marginBottom: theme.spacing.sm,
  },
  shareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: theme.spacing.sm,
    padding: theme.spacing.xs,
  },
  shareText: {
    marginLeft: theme.spacing.xs,
    color: theme.colors.primary,
    fontWeight: '600',
  },
  sectionCard: {
    margin: theme.spacing.md,
    marginTop: 0,
    elevation: 2,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  sectionTitle: {
    marginLeft: theme.spacing.sm,
    fontSize: 20,
    fontWeight: '600',
  },
  sectionText: {
    fontSize: 16,
    lineHeight: 26,
    color: theme.colors.text,
  },
  relatedVerseItem: {
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    backgroundColor: theme.colors.background,
    borderRadius: theme.borderRadius.md,
    borderLeftWidth: 3,
    borderLeftColor: theme.colors.primary,
  },
  relatedReference: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.primary,
    marginBottom: 4,
  },
  relatedText: {
    fontSize: 15,
    color: theme.colors.text,
    marginBottom: 4,
  },
  connectionText: {
    fontSize: 13,
    color: theme.colors.textSecondary,
    fontStyle: 'italic',
  },
  refreshButton: {
    marginLeft: 'auto',
    padding: theme.spacing.xs,
  },
  emptyText: {
    color: theme.colors.textSecondary,
    fontStyle: 'italic',
  },
  actions: {
    padding: theme.spacing.md,
  },
  actionButton: {
    marginBottom: theme.spacing.sm,
  },
});
