/**
 * Understanding Bible App
 * Deep understanding and scholar-level insights for Bible study
 */
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { Provider as PaperProvider } from 'react-native-paper';

// Screens
import TodayScreen from './src/screens/TodayScreen';
import DiscoverScreen from './src/screens/DiscoverScreen';
import JournalScreen from './src/screens/JournalScreen';
import ScholarScreen from './src/screens/ScholarScreen';
import UnderstandingScreen from './src/screens/UnderstandingScreen';
import SearchScreen from './src/screens/SearchScreen';

// Theme
import { theme } from './src/theme';

// Tab Navigator
const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Main Tab Navigator
function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Today') {
            iconName = focused ? 'book' : 'book-outline';
          } else if (route.name === 'Discover') {
            iconName = focused ? 'compass' : 'compass-outline';
          } else if (route.name === 'Journal') {
            iconName = focused ? 'journal' : 'journal-outline';
          } else if (route.name === 'Scholar') {
            iconName = focused ? 'school' : 'school-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.inactive,
        headerStyle: {
          backgroundColor: theme.colors.primary,
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      })}
    >
      <Tab.Screen 
        name="Today" 
        component={TodayScreen}
        options={{ title: "Today's Understanding" }}
      />
      <Tab.Screen 
        name="Discover" 
        component={DiscoverScreen}
        options={{ title: "Discover Connections" }}
      />
      <Tab.Screen 
        name="Scholar" 
        component={ScholarScreen}
        options={{ title: "Scholar Voice" }}
      />
      <Tab.Screen 
        name="Journal" 
        component={JournalScreen}
        options={{ title: "My Journal" }}
      />
    </Tab.Navigator>
  );
}

// Root Stack Navigator
function RootNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: theme.colors.primary,
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      }}
    >
      <Stack.Screen 
        name="Main" 
        component={MainTabs}
        options={{ headerShown: false }}
      />
      <Stack.Screen 
        name="Understanding" 
        component={UnderstandingScreen}
        options={{ title: "Deep Understanding" }}
      />
      <Stack.Screen 
        name="Search" 
        component={SearchScreen}
        options={{ title: "Search Verses" }}
      />
    </Stack.Navigator>
  );
}

// Main App Component
export default function App() {
  return (
    <PaperProvider theme={theme}>
      <NavigationContainer>
        <StatusBar style="light" />
        <RootNavigator />
      </NavigationContainer>
    </PaperProvider>
  );
}
