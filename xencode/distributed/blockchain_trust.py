"""
Blockchain Trust System
Implements BlockchainTrustManager for agent reputation, smart contract integration,
decentralized identity management, and reputation scoring algorithms.
"""

import asyncio
import logging
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import secrets
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import eth_keys


logger = logging.getLogger(__name__)


class TrustEventType(Enum):
    """Types of trust events that affect agent reputation."""
    SUCCESSFUL_INTERACTION = "successful_interaction"
    FAILED_INTERACTION = "failed_interaction"
    MALICIOUS_BEHAVIOR = "malicious_behavior"
    REPUTATION_ENDORSEMENT = "reputation_endorsement"
    RESOURCE_CONTRIBUTION = "resource_contribution"
    SERVICE_PROVIDED = "service_provided"


class IdentityStatus(Enum):
    """Status of agent identities in the system."""
    VERIFIED = "verified"
    PENDING_VERIFICATION = "pending_verification"
    SUSPICIOUS = "suspicious"
    REVOKED = "revoked"


@dataclass
class TrustEvent:
    """Represents a trust-related event in the blockchain."""
    event_id: str
    agent_id: str
    event_type: TrustEventType
    timestamp: datetime
    details: Dict[str, Any]
    signature: str
    previous_hash: str
    nonce: int = 0


@dataclass
class AgentIdentity:
    """Represents an agent's identity in the decentralized system."""
    agent_id: str
    public_key: str
    private_key_encrypted: str
    creation_date: datetime
    status: IdentityStatus
    metadata: Dict[str, Any]
    reputation_score: float = 0.0


@dataclass
class ReputationScore:
    """Detailed reputation score for an agent."""
    agent_id: str
    global_score: float
    category_scores: Dict[str, float]  # e.g., { "reliability": 0.9, "quality": 0.85 }
    total_interactions: int
    positive_interactions: int
    last_updated: datetime
    endorsers: List[str]  # List of agent IDs that endorsed this agent


class Block:
    """A block in the trust blockchain."""
    
    def __init__(self, index: int, previous_hash: str, trust_events: List[TrustEvent]):
        self.index = index
        self.timestamp = datetime.now()
        self.trust_events = trust_events
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
        
    def calculate_hash(self) -> str:
        """Calculate the hash of the block."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "trust_events": [event.event_id for event in self.trust_events],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
        
    def mine_block(self, difficulty: int):
        """Mine the block to meet the difficulty requirement."""
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            
        logger.info(f"Block mined: {self.hash}")


class SmartContractInterface:
    """Interface for interacting with smart contracts for trust verification."""
    
    def __init__(self):
        self.contract_address = None
        self.web3_connection = None
        
    async def deploy_contract(self, contract_code: str) -> str:
        """Deploy a new smart contract for trust management."""
        # In a real implementation, this would deploy to a blockchain
        # For now, we'll simulate deployment
        contract_address = f"0x{secrets.token_hex(20)}"
        self.contract_address = contract_address
        logger.info(f"Smart contract deployed at: {contract_address}")
        return contract_address
        
    async def verify_reputation(self, agent_id: str) -> float:
        """Verify an agent's reputation through the smart contract."""
        # In a real implementation, this would call the smart contract
        # For now, we'll simulate the verification
        # This could involve querying a blockchain for the agent's reputation
        return 0.85  # Simulated reputation score
        
    async def record_interaction(self, agent1_id: str, agent2_id: str, outcome: str) -> str:
        """Record an interaction between agents in the smart contract."""
        # In a real implementation, this would call the smart contract
        # For now, we'll simulate the recording
        interaction_id = f"interaction_{secrets.token_hex(8)}"
        logger.info(f"Interaction recorded: {interaction_id} between {agent1_id} and {agent2_id}")
        return interaction_id


class CryptographicManager:
    """Manages cryptographic operations for the trust system."""
    
    def __init__(self):
        self.keys_cache: Dict[str, Tuple[rsa.PrivateKey, rsa.PublicKey]] = {}
        
    def generate_key_pair(self) -> Tuple[str, str]:
        """Generate a new RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        
        return private_pem, public_pem
        
    def sign_message(self, message: str, private_key_pem: str) -> str:
        """Sign a message with the private key."""
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None
        )
        
        signature = private_key.sign(
            message.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
        
    def verify_signature(self, message: str, signature: str, public_key_pem: str) -> bool:
        """Verify a message signature with the public key."""
        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        signature_bytes = base64.b64decode(signature.encode())
        
        try:
            public_key.verify(
                signature_bytes,
                message.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class ReputationScoringAlgorithm:
    """Algorithm for calculating agent reputation scores."""
    
    def __init__(self):
        self.decay_factor = 0.95  # Reputation decays over time
        self.interaction_weight = 0.7
        self.endorsement_weight = 0.2
        self.temporal_weight = 0.1
        
    def calculate_reputation(
        self, 
        historical_events: List[TrustEvent], 
        endorsements: List[str],
        time_decay: bool = True
    ) -> ReputationScore:
        """Calculate reputation based on historical events and endorsements."""
        if not historical_events:
            return ReputationScore(
                agent_id="unknown",
                global_score=0.5,
                category_scores={},
                total_interactions=0,
                positive_interactions=0,
                last_updated=datetime.now(),
                endorsers=endorsements
            )
        
        # Calculate base reputation from events
        positive_events = 0
        negative_events = 0
        total_events = len(historical_events)
        
        category_contributions: Dict[str, List[float]] = {}
        
        for event in historical_events:
            if event.agent_id != historical_events[0].agent_id:
                continue  # Only consider events for this agent
                
            if event.event_type in [
                TrustEventType.SUCCESSFUL_INTERACTION, 
                TrustEventType.REPUTATION_ENDORSEMENT,
                TrustEventType.RESOURCE_CONTRIBUTION,
                TrustEventType.SERVICE_PROVIDED
            ]:
                positive_events += 1
            else:
                negative_events += 1
                
            # Add to category contributions
            category = event.event_type.value
            if category not in category_contributions:
                category_contributions[category] = []
            category_contributions[category].append(1.0 if "positive" in event.event_type.value else 0.0)
        
        # Calculate base score
        if total_events > 0:
            base_score = positive_events / total_events
        else:
            base_score = 0.5  # Neutral score if no events
        
        # Apply time decay if requested
        if time_decay and historical_events:
            oldest_event = min(event.timestamp for event in historical_events)
            days_since_oldest = (datetime.now() - oldest_event).days
            decay_multiplier = self.decay_factor ** (days_since_oldest / 30)  # Monthly decay
            base_score = base_score * decay_multiplier + (1 - decay_multiplier) * 0.5  # Revert to neutral
        
        # Calculate endorsement influence
        endorsement_score = 0.0
        if endorsements:
            # In a real system, we'd verify the endorsers' reputations
            # For now, we'll use a simple calculation
            endorsement_score = min(len(endorsements) * 0.1, 0.3)  # Max 0.3 from endorsements
        
        # Calculate category scores
        category_scores = {}
        for category, values in category_contributions.items():
            category_scores[category] = sum(values) / len(values) if values else 0.5
        
        # Calculate final score with weighted components
        final_score = (
            self.interaction_weight * base_score +
            self.endorsement_weight * endorsement_score +
            self.temporal_weight * base_score  # Temporal component is based on recency
        )
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, final_score))
        
        return ReputationScore(
            agent_id=historical_events[0].agent_id,
            global_score=final_score,
            category_scores=category_scores,
            total_interactions=total_events,
            positive_interactions=positive_events,
            last_updated=datetime.now(),
            endorsers=endorsements
        )


class BlockchainTrustManager:
    """
    Blockchain-based trust and reputation system for managing agent reputation,
    integrating smart contracts, decentralized identity management, 
    and reputation scoring algorithms.
    """
    
    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = []
        self.pending_trust_events: List[TrustEvent] = []
        self.agent_identities: Dict[str, AgentIdentity] = {}
        self.reputation_scores: Dict[str, ReputationScore] = {}
        self.difficulty = difficulty
        self.smart_contract = SmartContractInterface()
        self.crypto_manager = CryptographicManager()
        self.scoring_algorithm = ReputationScoringAlgorithm()
        
        # Initialize the blockchain with a genesis block
        self.create_genesis_block()
        
    def create_genesis_block(self):
        """Create the initial block in the blockchain."""
        genesis_block = Block(0, "0", [])
        self.chain.append(genesis_block)
        
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain."""
        return self.chain[-1]
        
    def add_trust_event(
        self, 
        agent_id: str, 
        event_type: TrustEventType, 
        details: Dict[str, Any]
    ) -> str:
        """Add a trust event to the pending list."""
        # Verify agent identity exists
        if agent_id not in self.agent_identities:
            raise ValueError(f"Agent {agent_id} not registered in the system")
            
        # Create event ID
        event_id = f"event_{secrets.token_hex(8)}"
        
        # Get previous hash
        previous_hash = self.get_latest_block().hash
        
        # Create trust event
        trust_event = TrustEvent(
            event_id=event_id,
            agent_id=agent_id,
            event_type=event_type,
            timestamp=datetime.now(),
            details=details,
            signature="",  # Will be added after mining
            previous_hash=previous_hash
        )
        
        # Sign the event
        agent_identity = self.agent_identities[agent_id]
        event_string = json.dumps({
            "event_id": trust_event.event_id,
            "agent_id": trust_event.agent_id,
            "event_type": trust_event.event_type.value,
            "timestamp": trust_event.timestamp.isoformat(),
            "details": trust_event.details,
            "previous_hash": trust_event.previous_hash
        }, sort_keys=True)
        
        trust_event.signature = self.crypto_manager.sign_message(
            event_string, 
            self._decrypt_private_key(agent_identity.private_key_encrypted)
        )
        
        # Add to pending events
        self.pending_trust_events.append(trust_event)
        
        logger.info(f"Trust event added: {event_id} for agent {agent_id}")
        return event_id
        
    def mine_pending_events(self) -> Optional[Block]:
        """Mine pending trust events into a new block."""
        if not self.pending_trust_events:
            return None
            
        # Create a new block with pending events
        new_block = Block(
            index=len(self.chain),
            previous_hash=self.get_latest_block().hash,
            trust_events=self.pending_trust_events.copy()
        )
        
        # Mine the block
        new_block.mine_block(self.difficulty)
        
        # Add to chain
        self.chain.append(new_block)
        
        # Clear pending events
        self.pending_trust_events.clear()
        
        # Update reputation scores based on new events
        self._update_reputation_scores(new_block.trust_events)
        
        logger.info(f"Mined block {new_block.index} with {len(new_block.trust_events)} trust events")
        return new_block
        
    def register_agent(
        self, 
        agent_metadata: Dict[str, Any], 
        initial_reputation: float = 0.5
    ) -> AgentIdentity:
        """Register a new agent in the decentralized identity system."""
        # Generate key pair
        private_key, public_key = self.crypto_manager.generate_key_pair()
        
        # Encrypt private key (in a real system, this would use the agent's password)
        encrypted_private_key = base64.b64encode(private_key.encode()).decode()
        
        # Create agent ID from public key hash
        agent_id = hashlib.sha256(public_key.encode()).hexdigest()[:16]
        
        # Create agent identity
        agent_identity = AgentIdentity(
            agent_id=agent_id,
            public_key=public_key,
            private_key_encrypted=encrypted_private_key,
            creation_date=datetime.now(),
            status=IdentityStatus.PENDING_VERIFICATION,
            metadata=agent_metadata,
            reputation_score=initial_reputation
        )
        
        # Store the identity
        self.agent_identities[agent_id] = agent_identity
        
        # Add initial trust event for registration
        self.add_trust_event(
            agent_id, 
            TrustEventType.REPUTATION_ENDORSEMENT, 
            {"action": "registration", "initial_reputation": initial_reputation}
        )
        
        logger.info(f"Agent registered: {agent_id}")
        return agent_identity
        
    def verify_agent_identity(self, agent_id: str) -> bool:
        """Verify an agent's identity."""
        if agent_id not in self.agent_identities:
            return False
            
        agent = self.agent_identities[agent_id]
        if agent.status == IdentityStatus.VERIFIED:
            return True
            
        # In a real system, this would involve verification processes
        # For now, we'll just mark as verified
        agent.status = IdentityStatus.VERIFIED
        self.add_trust_event(
            agent_id, 
            TrustEventType.SUCCESSFUL_INTERACTION, 
            {"action": "identity_verification"}
        )
        
        return True
        
    def get_reputation_score(self, agent_id: str) -> Optional[ReputationScore]:
        """Get the reputation score for an agent."""
        if agent_id in self.reputation_scores:
            return self.reputation_scores[agent_id]
            
        # If not cached, calculate from blockchain history
        historical_events = self._get_agent_trust_events(agent_id)
        if not historical_events:
            return None
            
        # Get endorsements for this agent
        endorsements = self._get_endorsements_for_agent(agent_id)
        
        # Calculate reputation
        score = self.scoring_algorithm.calculate_reputation(historical_events, endorsements)
        self.reputation_scores[agent_id] = score
        
        return score
        
    def _get_agent_trust_events(self, agent_id: str) -> List[TrustEvent]:
        """Get all trust events for a specific agent."""
        events = []
        for block in self.chain:
            for event in block.trust_events:
                if event.agent_id == agent_id:
                    events.append(event)
        return events
        
    def _get_endorsements_for_agent(self, agent_id: str) -> List[str]:
        """Get all agents that have endorsed the specified agent."""
        endorsements = []
        for block in self.chain:
            for event in block.trust_events:
                if (event.event_type == TrustEventType.REPUTATION_ENDORSEMENT and 
                    event.details.get("endorsed_agent") == agent_id):
                    endorsements.append(event.agent_id)
        return endorsements
        
    def _update_reputation_scores(self, new_events: List[TrustEvent]):
        """Update reputation scores based on new trust events."""
        # Group events by agent
        agent_events: Dict[str, List[TrustEvent]] = {}
        for event in new_events:
            if event.agent_id not in agent_events:
                agent_events[event.agent_id] = []
            agent_events[event.agent_id].append(event)
            
        # Update scores for each affected agent
        for agent_id, events in agent_events.items():
            # Get all historical events for this agent
            historical_events = self._get_agent_trust_events(agent_id)
            
            # Get endorsements for this agent
            endorsements = self._get_endorsements_for_agent(agent_id)
            
            # Calculate new reputation
            new_score = self.scoring_algorithm.calculate_reputation(
                historical_events, 
                endorsements
            )
            
            # Update the score
            self.reputation_scores[agent_id] = new_score
            
            # Update the agent identity's reputation score
            if agent_id in self.agent_identities:
                self.agent_identities[agent_id].reputation_score = new_score.global_score
                
    def _decrypt_private_key(self, encrypted_key: str) -> str:
        """Decrypt an encrypted private key."""
        # In a real system, this would use proper decryption
        # For now, we'll just decode the base64
        return base64.b64decode(encrypted_key.encode()).decode()
        
    def validate_chain(self) -> bool:
        """Validate the integrity of the blockchain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the hash is correct
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Block {i} has invalid hash")
                return False
                
            # Check if the previous hash matches
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Block {i} has invalid previous hash")
                return False
                
        return True
        
    def get_agent_info(self, agent_id: str) -> Optional[AgentIdentity]:
        """Get information about an agent."""
        return self.agent_identities.get(agent_id)
        
    def endorse_agent(self, endorser_id: str, endorsee_id: str) -> str:
        """Create an endorsement event for an agent."""
        if endorser_id not in self.agent_identities:
            raise ValueError(f"Endorser {endorser_id} not registered")
        if endorsee_id not in self.agent_identities:
            raise ValueError(f"Endorsee {endorsee_id} not registered")
            
        # Add endorsement event
        event_id = self.add_trust_event(
            endorser_id,
            TrustEventType.REPUTATION_ENDORSEMENT,
            {
                "endorsed_agent": endorsee_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Agent {endorser_id} endorsed agent {endorsee_id}")
        return event_id


# Convenience function for easy use
async def register_and_verify_agent(
    manager: Optional[BlockchainTrustManager] = None,
    agent_metadata: Dict[str, Any] = {}
) -> AgentIdentity:
    """
    Convenience function to register and verify an agent.
    
    Args:
        manager: Optional trust manager instance (creates one if not provided)
        agent_metadata: Metadata about the agent
        
    Returns:
        AgentIdentity object representing the registered agent
    """
    if manager is None:
        manager = BlockchainTrustManager()
        
    identity = manager.register_agent(agent_metadata)
    manager.verify_agent_identity(identity.agent_id)
    
    return identity