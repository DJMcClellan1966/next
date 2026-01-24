"""
Agent Negotiation - Agents negotiate for task allocation

Implements:
- Negotiation protocols
- Bid submission
- Agreement formation
- Conflict resolution
"""
from typing import Dict, List, Optional, Any
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class NegotiationStatus(Enum):
    """Negotiation status"""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    AGREED = "agreed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class Proposal:
    """Negotiation proposal"""
    proposal_id: str
    proposer_id: str
    task_id: str
    terms: Dict[str, Any]  # cost, time, quality, etc.
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Agreement:
    """Negotiation agreement"""
    agreement_id: str
    task_id: str
    parties: List[str]
    terms: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class NegotiationProtocol(Enum):
    """Negotiation protocols"""
    CONTRACT_NET = "contract_net"  # Manager announces, agents bid
    AUCTION = "auction"  # Competitive bidding
    BARGAINING = "bargaining"  # Back-and-forth negotiation
    CONSENSUS = "consensus"  # All parties must agree


class AgentNegotiation:
    """
    Agent Negotiation System
    
    Handles negotiations between agents for task allocation
    """
    
    def __init__(self, protocol: NegotiationProtocol = NegotiationProtocol.CONTRACT_NET):
        self.protocol = protocol
        self.negotiations: Dict[str, Dict] = {}  # negotiation_id -> negotiation
        self.proposals: Dict[str, List[Proposal]] = {}  # negotiation_id -> proposals
        self.agreements: Dict[str, Agreement] = {}  # agreement_id -> agreement
        self.agents: Dict[str, Any] = {}  # agent_id -> agent
    
    def register_agent(self, agent_id: str, agent: Any):
        """Register agent for negotiation"""
        self.agents[agent_id] = agent
        logger.info(f"[AgentNegotiation] Registered agent: {agent_id}")
    
    def initiate_negotiation(self, negotiation_id: str, task: Dict[str, Any],
                           initiator_id: str, participants: List[str]) -> str:
        """
        Initiate negotiation
        
        Parameters
        ----------
        negotiation_id : str
            Negotiation identifier
        task : dict
            Task to negotiate
        initiator_id : str
            Initiating agent
        participants : list of str
            Participating agents
            
        Returns
        -------
        negotiation_id : str
            Negotiation identifier
        """
        self.negotiations[negotiation_id] = {
            'task': task,
            'initiator': initiator_id,
            'participants': participants,
            'status': NegotiationStatus.INITIATED.value,
            'created_at': datetime.now().isoformat()
        }
        self.proposals[negotiation_id] = []
        
        logger.info(f"[AgentNegotiation] Negotiation {negotiation_id} initiated by {initiator_id}")
        return negotiation_id
    
    def submit_proposal(self, negotiation_id: str, proposer_id: str, 
                       terms: Dict[str, Any]) -> Proposal:
        """
        Submit proposal in negotiation
        
        Parameters
        ----------
        negotiation_id : str
            Negotiation identifier
        proposer_id : str
            Proposing agent
        terms : dict
            Proposal terms
            
        Returns
        -------
        proposal : Proposal
            Created proposal
        """
        if negotiation_id not in self.negotiations:
            raise ValueError(f"Negotiation not found: {negotiation_id}")
        
        negotiation = self.negotiations[negotiation_id]
        task_id = negotiation['task'].get('task_id', negotiation_id)
        
        proposal = Proposal(
            proposal_id=f"prop_{len(self.proposals[negotiation_id])}",
            proposer_id=proposer_id,
            task_id=task_id,
            terms=terms
        )
        
        self.proposals[negotiation_id].append(proposal)
        negotiation['status'] = NegotiationStatus.IN_PROGRESS.value
        
        logger.info(f"[AgentNegotiation] Proposal submitted by {proposer_id} for {negotiation_id}")
        return proposal
    
    def evaluate_proposals(self, negotiation_id: str, 
                          criteria: str = 'best_value') -> Optional[Proposal]:
        """
        Evaluate proposals and select best
        
        Parameters
        ----------
        negotiation_id : str
            Negotiation identifier
        criteria : str
            Evaluation criteria
            
        Returns
        -------
        best_proposal : Proposal, optional
            Best proposal
        """
        if negotiation_id not in self.proposals or not self.proposals[negotiation_id]:
            return None
        
        proposals = self.proposals[negotiation_id]
        
        if criteria == 'best_value':
            # Best value = lowest cost / highest quality
            best = None
            best_score = 0.0
            
            for proposal in proposals:
                cost = proposal.terms.get('cost', float('inf'))
                quality = proposal.terms.get('quality', 0.0)
                score = quality / (cost + 1e-10)  # Avoid division by zero
                
                if score > best_score:
                    best_score = score
                    best = proposal
            
            return best
        
        elif criteria == 'lowest_cost':
            return min(proposals, key=lambda p: p.terms.get('cost', float('inf')))
        
        elif criteria == 'fastest':
            return min(proposals, key=lambda p: p.terms.get('estimated_time', float('inf')))
        
        else:
            return proposals[0] if proposals else None
    
    def form_agreement(self, negotiation_id: str, selected_proposal: Proposal) -> Agreement:
        """
        Form agreement from selected proposal
        
        Parameters
        ----------
        negotiation_id : str
            Negotiation identifier
        selected_proposal : Proposal
            Selected proposal
            
        Returns
        -------
        agreement : Agreement
            Formed agreement
        """
        negotiation = self.negotiations[negotiation_id]
        
        agreement = Agreement(
            agreement_id=f"agreement_{len(self.agreements)}",
            task_id=selected_proposal.task_id,
            parties=[negotiation['initiator'], selected_proposal.proposer_id],
            terms=selected_proposal.terms
        )
        
        self.agreements[agreement.agreement_id] = agreement
        negotiation['status'] = NegotiationStatus.AGREED.value
        negotiation['agreement_id'] = agreement.agreement_id
        
        logger.info(f"[AgentNegotiation] Agreement formed: {agreement.agreement_id}")
        return agreement
    
    def get_agreement(self, agreement_id: str) -> Optional[Agreement]:
        """Get agreement by ID"""
        return self.agreements.get(agreement_id)
    
    def get_negotiation_stats(self) -> Dict:
        """Get negotiation statistics"""
        return {
            'total_negotiations': len(self.negotiations),
            'total_agreements': len(self.agreements),
            'agreement_rate': len(self.agreements) / len(self.negotiations) if self.negotiations else 0.0
        }
