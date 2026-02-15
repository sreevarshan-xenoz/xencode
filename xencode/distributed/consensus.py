"""
Distributed Consensus System
Implements ConsensusEngine for multi-agent decisions, Byzantine fault tolerance mechanisms,
voting and agreement protocols, and conflict resolution algorithms.
"""

import asyncio
import logging
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from datetime import datetime
import secrets
import random
from collections import Counter


logger = logging.getLogger(__name__)


class ConsensusMessageType(Enum):
    """Types of messages in the consensus protocol."""
    PROPOSE = "propose"
    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    DECIDE = "decide"
    VOTE = "vote"
    REJECT = "reject"
    HEARTBEAT = "heartbeat"


class ConsensusPhase(Enum):
    """Phases of the consensus protocol."""
    PROPOSAL = "proposal"
    PREPARATION = "preparation"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    DECIDED = "decided"


class VoteValue(Enum):
    """Possible vote values."""
    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class ConsensusMessage:
    """A message in the consensus protocol."""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None if broadcast
    message_type: ConsensusMessageType
    proposal_id: str
    value: Any  # The proposed value
    round_number: int
    timestamp: datetime
    signature: str
    metadata: Dict[str, Any] = None


@dataclass
class Proposal:
    """A proposal in the consensus process."""
    proposal_id: str
    proposer_id: str
    value: Any
    round_number: int
    timestamp: datetime
    votes: Dict[str, VoteValue]  # voter_id -> vote_value
    status: ConsensusPhase
    quorum_count: int


@dataclass
class ConsensusDecision:
    """Result of a consensus decision."""
    proposal_id: str
    decided_value: Any
    participants: List[str]
    consensus_time: datetime
    agreement_percentage: float
    metadata: Dict[str, Any]


class ByzantineFaultToleranceManager:
    """Manages Byzantine fault tolerance in the consensus system."""
    
    def __init__(self, node_count: int):
        self.node_count = node_count
        self.max_faulty_nodes = (node_count - 1) // 3  # For Byzantine fault tolerance
        self.suspected_malicious_nodes: Set[str] = set()
        self.voting_history: Dict[str, List[Tuple[str, VoteValue, datetime]]] = {}
        
    def is_safe_to_proceed(self) -> bool:
        """Check if it's safe to proceed with consensus given potential faulty nodes."""
        active_nodes = self.node_count - len(self.suspected_malicious_nodes)
        return active_nodes > 2 * self.max_faulty_nodes
        
    def suspect_node(self, node_id: str, reason: str = ""):
        """Mark a node as suspected for malicious behavior."""
        if node_id not in self.suspected_malicious_nodes:
            self.suspected_malicious_nodes.add(node_id)
            logger.warning(f"Node {node_id} suspected of malicious behavior: {reason}")
            
    def record_vote_pattern(self, node_id: str, proposal_id: str, vote: VoteValue):
        """Record a node's voting pattern for analysis."""
        if node_id not in self.voting_history:
            self.voting_history[node_id] = []
        self.voting_history[node_id].append((proposal_id, vote, datetime.now()))
        
    def analyze_voting_patterns(self) -> List[str]:
        """Analyze voting patterns to detect potentially malicious nodes."""
        suspicious_nodes = []
        
        for node_id, votes in self.voting_history.items():
            # Check if the node frequently votes differently from the majority
            if len(votes) > 5:  # Need sufficient data points
                majority_opposed_count = 0
                
                for proposal_id, vote, _ in votes:
                    # Get all votes for this proposal
                    proposal_votes = [
                        v for nid, vote_list in self.voting_history.items()
                        for pid, v, _ in vote_list if pid == proposal_id
                    ]
                    
                    if proposal_votes:
                        majority_vote = Counter(proposal_votes).most_common(1)[0][0]
                        if vote != majority_vote:
                            majority_opposed_count += 1
                
                # If the node disagrees with majority more than 70% of the time, flag it
                disagreement_ratio = majority_opposed_count / len(votes)
                if disagreement_ratio > 0.7:
                    suspicious_nodes.append(node_id)
                    
        return suspicious_nodes


class VotingProtocol:
    """Implements the voting protocol for consensus."""
    
    def __init__(self, consensus_threshold: float = 0.67):
        self.consensus_threshold = consensus_threshold  # 67% for safety
        self.ballots: Dict[str, Dict[str, VoteValue]] = {}  # proposal_id -> {node_id -> vote}
        
    def submit_vote(self, proposal_id: str, node_id: str, vote: VoteValue) -> bool:
        """Submit a vote for a proposal."""
        if proposal_id not in self.ballots:
            self.ballots[proposal_id] = {}
            
        self.ballots[proposal_id][node_id] = vote
        return True
        
    def get_vote_count(self, proposal_id: str, vote_value: VoteValue) -> int:
        """Get the count of a specific vote value for a proposal."""
        if proposal_id not in self.ballots:
            return 0
            
        return sum(1 for vote in self.ballots[proposal_id].values() if vote == vote_value)
        
    def get_total_votes(self, proposal_id: str) -> int:
        """Get the total number of votes for a proposal."""
        if proposal_id not in self.ballots:
            return 0
        return len(self.ballots[proposal_id])
        
    def has_consensus(self, proposal_id: str, total_nodes: int) -> Tuple[bool, Optional[VoteValue]]:
        """Check if consensus has been reached for a proposal."""
        if proposal_id not in self.ballots:
            return False, None
            
        votes = self.ballots[proposal_id]
        vote_counts = Counter(votes.values())
        
        # Check for each possible vote value
        for vote_value in [VoteValue.ACCEPT, VoteValue.REJECT]:
            count = vote_counts[vote_value]
            threshold = int(total_nodes * self.consensus_threshold)
            
            if count >= threshold:
                return True, vote_value
                
        return False, None
        
    def get_current_votes(self, proposal_id: str) -> Dict[VoteValue, int]:
        """Get the current vote distribution for a proposal."""
        if proposal_id not in self.ballots:
            return {}
            
        vote_counts = Counter(self.ballots[proposal_id].values())
        return {vote: count for vote, count in vote_counts.items()}


class ConflictResolutionAlgorithm:
    """Algorithm for resolving conflicts in the consensus process."""
    
    def __init__(self):
        self.conflict_resolution_strategies = {
            "first_proposal_wins": self._resolve_first_proposal_wins,
            "highest_priority_wins": self._resolve_highest_priority_wins,
            "random_choice": self._resolve_random_choice,
            "majority_preference": self._resolve_majority_preference
        }
        
    def resolve_conflict(
        self, 
        proposals: List[Proposal], 
        strategy: str = "first_proposal_wins"
    ) -> Proposal:
        """Resolve a conflict between multiple proposals."""
        if not proposals:
            raise ValueError("No proposals to resolve")
            
        if len(proposals) == 1:
            return proposals[0]
            
        resolver = self.conflict_resolution_strategies.get(strategy)
        if not resolver:
            raise ValueError(f"Unknown conflict resolution strategy: {strategy}")
            
        return resolver(proposals)
        
    def _resolve_first_proposal_wins(self, proposals: List[Proposal]) -> Proposal:
        """Resolve conflict by choosing the first received proposal."""
        return min(proposals, key=lambda p: p.timestamp)
        
    def _resolve_highest_priority_wins(self, proposals: List[Proposal]) -> Proposal:
        """Resolve conflict by choosing the proposal with highest priority."""
        # Priority could be based on proposer reputation, proposal importance, etc.
        # For now, we'll use a simple heuristic based on proposer_id hash
        return max(proposals, key=lambda p: int(p.proposer_id[:8], 16))
        
    def _resolve_random_choice(self, proposals: List[Proposal]) -> Proposal:
        """Resolve conflict by randomly choosing a proposal."""
        return random.choice(proposals)
        
    def _resolve_majority_preference(self, proposals: List[Proposal]) -> Proposal:
        """Resolve conflict by choosing the proposal with most preliminary support."""
        # Count preliminary votes or preferences for each proposal
        proposal_support = {}
        for proposal in proposals:
            support_count = sum(1 for vote in proposal.votes.values() if vote == VoteValue.ACCEPT)
            proposal_support[proposal.proposal_id] = support_count
            
        # Choose the proposal with the most support
        winning_proposal_id = max(proposal_support, key=proposal_support.get)
        return next(p for p in proposals if p.proposal_id == winning_proposal_id)


class ConsensusEngine:
    """
    Distributed consensus engine for multi-agent decisions with Byzantine fault tolerance,
    voting protocols, and conflict resolution.
    """
    
    def __init__(self, node_id: str, all_node_ids: List[str], consensus_timeout: int = 30):
        self.node_id = node_id
        self.all_node_ids = set(all_node_ids)
        self.consensus_timeout = consensus_timeout
        self.bft_manager = ByzantineFaultToleranceManager(len(all_node_ids))
        self.voting_protocol = VotingProtocol()
        self.conflict_resolver = ConflictResolutionAlgorithm()
        self.active_proposals: Dict[str, Proposal] = {}
        self.decisions: Dict[str, ConsensusDecision] = {}
        self.message_queue: List[ConsensusMessage] = []
        self.network_interface = None  # To be set externally
        self.consensus_callbacks: Dict[str, callable] = {}
        
    def set_network_interface(self, network_interface):
        """Set the network interface for sending/receiving messages."""
        self.network_interface = network_interface
        
    async def propose_value(self, value: Any, proposal_metadata: Dict[str, Any] = None) -> str:
        """
        Propose a value for consensus.
        
        Args:
            value: The value to propose
            proposal_metadata: Additional metadata about the proposal
            
        Returns:
            The ID of the proposal
        """
        proposal_id = f"proposal_{secrets.token_hex(8)}"
        round_number = 1  # Starting round
        
        proposal = Proposal(
            proposal_id=proposal_id,
            proposer_id=self.node_id,
            value=value,
            round_number=round_number,
            timestamp=datetime.now(),
            votes={self.node_id: VoteValue.ACCEPT},  # Self-vote
            status=ConsensusPhase.PROPOSAL,
            quorum_count=int(len(self.all_node_ids) * 2 / 3) + 1  # 2f+1 quorum
        )
        
        self.active_proposals[proposal_id] = proposal
        
        # Broadcast proposal to all nodes
        await self._broadcast_message(
            ConsensusMessageType.PROPOSE,
            proposal_id,
            value,
            round_number,
            proposal_metadata or {}
        )
        
        logger.info(f"Node {self.node_id} proposed value for {proposal_id}")
        
        # Start consensus process
        asyncio.create_task(self._run_consensus_process(proposal_id))
        
        return proposal_id
        
    async def _run_consensus_process(self, proposal_id: str):
        """Run the consensus process for a proposal."""
        if proposal_id not in self.active_proposals:
            return
            
        proposal = self.active_proposals[proposal_id]
        
        # Phase 1: Preparation
        proposal.status = ConsensusPhase.PREPARATION
        await self._broadcast_message(
            ConsensusMessageType.PREPARE,
            proposal_id,
            proposal.value,
            proposal.round_number
        )
        
        # Wait for prepare acknowledgments or timeout
        await asyncio.sleep(min(self.consensus_timeout / 4, 5))  # Shorter wait for prepare phase
        
        # Phase 2: Pre-commit
        if self._has_sufficient_prepares(proposal_id):
            proposal.status = ConsensusPhase.PRE_COMMIT
            await self._broadcast_message(
                ConsensusMessageType.PRE_COMMIT,
                proposal_id,
                proposal.value,
                proposal.round_number
            )
            
            # Wait for pre-commit acknowledgments
            await asyncio.sleep(min(self.consensus_timeout / 4, 5))
            
            # Phase 3: Commit
            if self._has_sufficient_pre_commits(proposal_id):
                proposal.status = ConsensusPhase.COMMIT
                await self._broadcast_message(
                    ConsensusMessageType.COMMIT,
                    proposal_id,
                    proposal.value,
                    proposal.round_number
                )
                
                # Wait for commit acknowledgments
                await asyncio.sleep(min(self.consensus_timeout / 4, 5))
                
                # Phase 4: Decide
                if self._has_sufficient_commits(proposal_id):
                    await self._finalize_decision(proposal_id)
                    return
        
        # If consensus failed, initiate new round or abort
        await self._handle_consensus_failure(proposal_id)
        
    def _has_sufficient_prepares(self, proposal_id: str) -> bool:
        """Check if there are sufficient prepare messages for the proposal."""
        # In a real implementation, this would count prepare messages
        # For now, we'll use a simple heuristic
        return True  # Simplified for this implementation
        
    def _has_sufficient_pre_commits(self, proposal_id: str) -> bool:
        """Check if there are sufficient pre-commit messages for the proposal."""
        # In a real implementation, this would count pre-commit messages
        # For now, we'll use a simple heuristic
        return True  # Simplified for this implementation
        
    def _has_sufficient_commits(self, proposal_id: str) -> bool:
        """Check if there are sufficient commit messages for the proposal."""
        # Check if we have reached consensus threshold
        has_consensus, vote_value = self.voting_protocol.has_consensus(
            proposal_id, len(self.all_node_ids)
        )
        return has_consensus and vote_value == VoteValue.ACCEPT
        
    async def _finalize_decision(self, proposal_id: str):
        """Finalize the consensus decision."""
        if proposal_id not in self.active_proposals:
            return
            
        proposal = self.active_proposals[proposal_id]
        proposal.status = ConsensusPhase.DECIDED
        
        # Calculate agreement percentage
        total_votes = self.voting_protocol.get_total_votes(proposal_id)
        accept_votes = self.voting_protocol.get_vote_count(proposal_id, VoteValue.ACCEPT)
        agreement_percentage = accept_votes / total_votes if total_votes > 0 else 0
        
        decision = ConsensusDecision(
            proposal_id=proposal_id,
            decided_value=proposal.value,
            participants=list(self.all_node_ids),
            consensus_time=datetime.now(),
            agreement_percentage=agreement_percentage,
            metadata={"round_number": proposal.round_number}
        )
        
        self.decisions[proposal_id] = decision
        
        # Broadcast decision
        await self._broadcast_message(
            ConsensusMessageType.DECIDE,
            proposal_id,
            proposal.value,
            proposal.round_number,
            {"agreement_percentage": agreement_percentage}
        )
        
        logger.info(f"Consensus reached for {proposal_id}: {proposal.value}")
        
        # Call any registered callbacks
        callback = self.consensus_callbacks.get(proposal_id)
        if callback:
            try:
                await callback(decision)
            except Exception as e:
                logger.error(f"Error in consensus callback: {str(e)}")
        
    async def _handle_consensus_failure(self, proposal_id: str):
        """Handle consensus failure and potentially start a new round."""
        logger.warning(f"Consensus failed for {proposal_id}")
        
        # In a real implementation, this might start a new round with a different proposer
        # or use a conflict resolution mechanism
        pass
        
    async def _broadcast_message(
        self, 
        msg_type: ConsensusMessageType, 
        proposal_id: str, 
        value: Any, 
        round_num: int,
        metadata: Dict[str, Any] = None
    ):
        """Broadcast a consensus message to all nodes."""
        if not self.network_interface:
            logger.error("No network interface set for consensus engine")
            return
            
        message = ConsensusMessage(
            message_id=f"msg_{secrets.token_hex(8)}",
            sender_id=self.node_id,
            receiver_id=None,  # Broadcast
            message_type=msg_type,
            proposal_id=proposal_id,
            value=value,
            round_number=round_num,
            timestamp=datetime.now(),
            signature="",  # Would be signed in real implementation
            metadata=metadata or {}
        )
        
        # Add to local queue
        self.message_queue.append(message)
        
        # Send via network interface
        try:
            await self.network_interface.broadcast_message(message)
        except Exception as e:
            logger.error(f"Error broadcasting consensus message: {str(e)}")
            
    async def handle_consensus_message(self, message: ConsensusMessage):
        """Handle an incoming consensus message."""
        if message.proposal_id not in self.active_proposals:
            # If this is a new proposal, add it
            if message.message_type == ConsensusMessageType.PROPOSE:
                proposal = Proposal(
                    proposal_id=message.proposal_id,
                    proposer_id=message.sender_id,
                    value=message.value,
                    round_number=message.round_number,
                    timestamp=message.timestamp,
                    votes={},
                    status=ConsensusPhase.PROPOSAL,
                    quorum_count=int(len(self.all_node_ids) * 2 / 3) + 1
                )
                self.active_proposals[message.proposal_id] = proposal
            else:
                logger.warning(f"Received message for unknown proposal: {message.proposal_id}")
                return
        
        proposal = self.active_proposals[message.proposal_id]
        
        if message.message_type == ConsensusMessageType.PROPOSE:
            # Handle proposal message
            logger.info(f"Received proposal {message.proposal_id} from {message.sender_id}")
            
        elif message.message_type == ConsensusMessageType.VOTE:
            # Handle vote message
            vote_value = VoteValue(message.metadata.get("vote", "accept"))
            self.voting_protocol.submit_vote(message.proposal_id, message.sender_id, vote_value)
            self.bft_manager.record_vote_pattern(message.sender_id, message.proposal_id, vote_value)
            
            # Check if consensus is reached
            has_consensus, vote_result = self.voting_protocol.has_consensus(
                message.proposal_id, len(self.all_node_ids)
            )
            
            if has_consensus and vote_result == VoteValue.ACCEPT:
                await self._finalize_decision(message.proposal_id)
                
        elif message.message_type == ConsensusMessageType.HEARTBEAT:
            # Handle heartbeat - just acknowledge
            pass
            
        else:
            # Handle other message types
            logger.debug(f"Received {message.message_type.value} message for {message.proposal_id}")
            
    def register_consensus_callback(self, proposal_id: str, callback: callable):
        """Register a callback to be called when consensus is reached."""
        self.consensus_callbacks[proposal_id] = callback
        
    async def vote_on_proposal(self, proposal_id: str, vote: VoteValue) -> bool:
        """Vote on a proposal."""
        if proposal_id not in self.active_proposals:
            logger.warning(f"Attempted to vote on unknown proposal: {proposal_id}")
            return False
            
        # Submit vote through the voting protocol
        success = self.voting_protocol.submit_vote(proposal_id, self.node_id, vote)
        
        if success:
            # Record vote pattern for BFT analysis
            self.bft_manager.record_vote_pattern(self.node_id, proposal_id, vote)
            
            # Broadcast vote to other nodes
            await self._broadcast_message(
                ConsensusMessageType.VOTE,
                proposal_id,
                None,  # Value not needed for vote
                self.active_proposals[proposal_id].round_number,
                {"vote": vote.value}
            )
            
            logger.info(f"Node {self.node_id} voted {vote.value} on {proposal_id}")
            
        return success
        
    def get_consensus_status(self, proposal_id: str) -> Optional[ConsensusPhase]:
        """Get the current status of a proposal."""
        if proposal_id not in self.active_proposals:
            return None
        return self.active_proposals[proposal_id].status
        
    def get_decision(self, proposal_id: str) -> Optional[ConsensusDecision]:
        """Get the decision for a proposal if it's been decided."""
        return self.decisions.get(proposal_id)
        
    async def resolve_multiple_proposals(self, proposal_ids: List[str], strategy: str = "first_proposal_wins") -> str:
        """Resolve conflicts between multiple competing proposals."""
        proposals = [self.active_proposals[pid] for pid in proposal_ids if pid in self.active_proposals]
        
        if not proposals:
            raise ValueError("None of the provided proposal IDs are active")
            
        winning_proposal = self.conflict_resolver.resolve_conflict(proposals, strategy)
        
        # Mark other proposals as rejected
        for proposal in proposals:
            if proposal.proposal_id != winning_proposal.proposal_id:
                proposal.status = ConsensusPhase.DECIDED
                # Create a decision indicating rejection
                decision = ConsensusDecision(
                    proposal_id=proposal.proposal_id,
                    decided_value=None,
                    participants=list(self.all_node_ids),
                    consensus_time=datetime.now(),
                    agreement_percentage=0.0,
                    metadata={"rejected": True, "reason": f"Lost to proposal {winning_proposal.proposal_id}"}
                )
                self.decisions[proposal.proposal_id] = decision
        
        # Finalize the winning proposal
        await self._finalize_decision(winning_proposal.proposal_id)
        
        return winning_proposal.proposal_id


# Convenience function for easy use
async def reach_consensus(
    node_id: str,
    all_node_ids: List[str],
    value_to_propose: Any,
    consensus_engine: Optional[ConsensusEngine] = None
) -> str:
    """
    Convenience function to propose a value and reach consensus.
    
    Args:
        node_id: ID of the current node
        all_node_ids: List of all node IDs in the network
        value_to_propose: Value to propose for consensus
        consensus_engine: Optional consensus engine instance (creates one if not provided)
        
    Returns:
        The ID of the proposal that reached consensus
    """
    if consensus_engine is None:
        consensus_engine = ConsensusEngine(node_id, all_node_ids)
        
    return await consensus_engine.propose_value(value_to_propose)