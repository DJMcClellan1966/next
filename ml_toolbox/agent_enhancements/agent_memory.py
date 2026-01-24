"""
Agent Memory - Short-term and Long-term Memory

Critical for production agents
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Memory item"""
    content: Any
    timestamp: float
    importance: float = 1.0
    tags: List[str] = field(default_factory=list)


class ShortTermMemory:
    """
    Short-term Memory
    
    Recent context, conversation history
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize short-term memory
        
        Parameters
        ----------
        max_size : int
            Maximum number of items
        """
        self.memory: deque = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, content: Any, importance: float = 1.0, tags: Optional[List[str]] = None):
        """Add item to memory"""
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            tags=tags or []
        )
        self.memory.append(item)
        logger.debug(f"[ShortTermMemory] Added item (size: {len(self.memory)})")
    
    def get_recent(self, n: int = 10) -> List[Any]:
        """Get recent items"""
        return [item.content for item in list(self.memory)[-n:]]
    
    def search(self, query: str) -> List[Any]:
        """Search memory by query"""
        results = []
        query_lower = query.lower()
        
        for item in self.memory:
            if isinstance(item.content, str) and query_lower in item.content.lower():
                results.append(item.content)
        
        return results
    
    def clear(self):
        """Clear memory"""
        self.memory.clear()


class LongTermMemory:
    """
    Long-term Memory
    
    Persistent knowledge, learned patterns
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize long-term memory
        
        Parameters
        ----------
        storage_path : str, optional
            Path to persistent storage
        """
        self.memory: Dict[str, MemoryItem] = {}
        self.storage_path = storage_path
        self._load()
    
    def add(self, key: str, content: Any, importance: float = 1.0, tags: Optional[List[str]] = None):
        """Add item to long-term memory"""
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            importance=importance,
            tags=tags or []
        )
        self.memory[key] = item
        self._save()
        logger.debug(f"[LongTermMemory] Added: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item by key"""
        item = self.memory.get(key)
        return item.content if item else None
    
    def search(self, query: str) -> List[tuple]:
        """Search memory"""
        results = []
        query_lower = query.lower()
        
        for key, item in self.memory.items():
            if query_lower in key.lower():
                results.append((key, item.content))
            elif isinstance(item.content, str) and query_lower in item.content.lower():
                results.append((key, item.content))
        
        return results
    
    def _save(self):
        """Save to persistent storage"""
        if self.storage_path:
            try:
                import json
                data = {
                    key: {
                        'content': item.content,
                        'timestamp': item.timestamp,
                        'importance': item.importance,
                        'tags': item.tags
                    }
                    for key, item in self.memory.items()
                }
                with open(self.storage_path, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"[LongTermMemory] Save failed: {e}")
    
    def _load(self):
        """Load from persistent storage"""
        if self.storage_path:
            try:
                import json
                import os
                if os.path.exists(self.storage_path):
                    with open(self.storage_path, 'r') as f:
                        data = json.load(f)
                    for key, item_data in data.items():
                        self.memory[key] = MemoryItem(
                            content=item_data['content'],
                            timestamp=item_data.get('timestamp', time.time()),
                            importance=item_data.get('importance', 1.0),
                            tags=item_data.get('tags', [])
                        )
            except Exception as e:
                logger.warning(f"[LongTermMemory] Load failed: {e}")


class AgentMemory:
    """
    Complete Agent Memory System
    
    Combines short-term and long-term memory
    """
    
    def __init__(self, short_term_size: int = 100, long_term_path: Optional[str] = None):
        """
        Initialize agent memory
        
        Parameters
        ----------
        short_term_size : int
            Short-term memory size
        long_term_path : str, optional
            Long-term memory storage path
        """
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.long_term = LongTermMemory(storage_path=long_term_path)
    
    def remember(self, content: Any, persistent: bool = False, key: Optional[str] = None,
                importance: float = 1.0, tags: Optional[List[str]] = None):
        """
        Remember content
        
        Parameters
        ----------
        content : any
            Content to remember
        persistent : bool
            Store in long-term memory
        key : str, optional
            Key for long-term storage
        importance : float
            Importance score
        tags : list, optional
            Tags for search
        """
        self.short_term.add(content, importance, tags)
        
        if persistent and key:
            self.long_term.add(key, content, importance, tags)
    
    def recall(self, query: str, use_long_term: bool = True) -> List[Any]:
        """
        Recall from memory
        
        Parameters
        ----------
        query : str
            Search query
        use_long_term : bool
            Search long-term memory
            
        Returns
        -------
        results : list
            Recalled items
        """
        results = self.short_term.search(query)
        
        if use_long_term:
            long_term_results = self.long_term.search(query)
            results.extend([content for _, content in long_term_results])
        
        return results
    
    def get_recent_context(self, n: int = 10) -> List[Any]:
        """Get recent context"""
        return self.short_term.get_recent(n)
