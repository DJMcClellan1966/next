"""
Agent Communication - Inter-Agent Communication

Implements:
- Message passing
- Agent coordination
- Communication protocols
- Message queues
"""
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    ERROR = "error"


@dataclass
class Message:
    """Agent message"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    response_to: Optional[str] = None  # Original message ID if this is a response


class AgentCommunication:
    """
    Agent Communication System
    
    Handles inter-agent communication
    """
    
    def __init__(self):
        self.message_queues: Dict[str, deque] = {}  # agent_id -> message queue
        self.message_history: List[Message] = []
        self.registered_agents: Dict[str, Any] = {}  # agent_id -> agent reference
    
    def register_agent(self, agent_id: str, agent: Any):
        """
        Register agent for communication
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
        agent : any
            Agent instance
        """
        self.registered_agents[agent_id] = agent
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = deque()
        logger.info(f"[AgentCommunication] Registered agent: {agent_id}")
    
    def send_message(self, message: Message) -> bool:
        """
        Send message to agent
        
        Parameters
        ----------
        message : Message
            Message to send
            
        Returns
        -------
        success : bool
            Whether message was sent
        """
        # Store in history
        self.message_history.append(message)
        
        # Send to specific agent
        if message.receiver_id:
            if message.receiver_id in self.message_queues:
                self.message_queues[message.receiver_id].append(message)
                logger.info(f"[AgentCommunication] Sent message {message.message_id} to {message.receiver_id}")
                return True
            else:
                logger.warning(f"[AgentCommunication] Agent not found: {message.receiver_id}")
                return False
        
        # Broadcast to all agents
        elif message.message_type == MessageType.BROADCAST:
            for agent_id in self.message_queues.keys():
                if agent_id != message.sender_id:  # Don't send to self
                    self.message_queues[agent_id].append(message)
            logger.info(f"[AgentCommunication] Broadcast message {message.message_id}")
            return True
        
        return False
    
    def receive_messages(self, agent_id: str) -> List[Message]:
        """
        Receive messages for agent
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
            
        Returns
        -------
        messages : list of Message
            Received messages
        """
        if agent_id not in self.message_queues:
            return []
        
        messages = []
        queue = self.message_queues[agent_id]
        
        while queue:
            messages.append(queue.popleft())
        
        return messages
    
    def create_message(self, sender_id: str, receiver_id: Optional[str],
                      message_type: MessageType, content: Dict[str, Any],
                      requires_response: bool = False) -> Message:
        """
        Create a message
        
        Parameters
        ----------
        sender_id : str
            Sender identifier
        receiver_id : str, optional
            Receiver identifier (None for broadcast)
        message_type : MessageType
            Message type
        content : dict
            Message content
        requires_response : bool
            Whether response is required
            
        Returns
        -------
        message : Message
            Created message
        """
        message_id = f"msg_{len(self.message_history)}"
        return Message(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            requires_response=requires_response
        )
    
    def send_request(self, sender_id: str, receiver_id: str, 
                    request_content: Dict[str, Any]) -> Message:
        """
        Send request message
        
        Parameters
        ----------
        sender_id : str
            Sender identifier
        receiver_id : str
            Receiver identifier
        request_content : dict
            Request content
            
        Returns
        -------
        message : Message
            Sent message
        """
        message = self.create_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            content=request_content,
            requires_response=True
        )
        self.send_message(message)
        return message
    
    def send_response(self, sender_id: str, receiver_id: str,
                     original_message_id: str, response_content: Dict[str, Any]) -> Message:
        """
        Send response message
        
        Parameters
        ----------
        sender_id : str
            Sender identifier
        receiver_id : str
            Receiver identifier
        original_message_id : str
            Original message ID
        response_content : dict
            Response content
            
        Returns
        -------
        message : Message
            Sent message
        """
        message = self.create_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.RESPONSE,
            content=response_content,
            requires_response=False
        )
        message.response_to = original_message_id
        self.send_message(message)
        return message
    
    def get_communication_stats(self) -> Dict:
        """Get communication statistics"""
        return {
            'total_messages': len(self.message_history),
            'registered_agents': len(self.registered_agents),
            'pending_messages': sum(len(q) for q in self.message_queues.values())
        }
