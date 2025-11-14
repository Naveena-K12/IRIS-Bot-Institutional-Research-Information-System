"""
Research Domain Assigner
Automatically assigns the nearest/most appropriate research domain based on paper content analysis.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class DomainMatch:
    """Result of domain matching analysis."""
    domain: str
    confidence: float
    matched_keywords: List[str]
    score: float
    reason: str


class ResearchDomainAssigner:
    """Assigns research domains based on content analysis."""
    
    def __init__(self):
        """Initialize the domain assigner with predefined domains and keywords."""
        self.domains = self._load_research_domains()
        self.keyword_patterns = self._create_keyword_patterns()
    
    def _load_research_domains(self) -> Dict[str, Dict]:
        """Load research domains with their associated keywords and weights."""
        return {
            "Artificial Intelligence (AI)": {
                "keywords": [
                    "artificial intelligence", "ai", "machine intelligence", "intelligent systems",
                    "expert systems", "knowledge representation", "reasoning", "problem solving",
                    "automated planning", "intelligent agents", "cognitive computing"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\b(?:artificial intelligence|ai)\b",
                    r"\bintelligent (?:systems?|agents?|computing)\b",
                    r"\bexpert systems?\b",
                    r"\bknowledge representation\b"
                ]
            },
            "Machine Learning (ML)": {
                "keywords": [
                    "machine learning", "ml", "supervised learning", "unsupervised learning",
                    "reinforcement learning", "deep learning", "neural networks", "algorithms",
                    "classification", "regression", "clustering", "feature extraction",
                    "model training", "predictive modeling", "pattern recognition"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bmachine learning\b",
                    r"\b(?:supervised|unsupervised|reinforcement) learning\b",
                    r"\bdeep learning\b",
                    r"\bneural networks?\b",
                    r"\b(?:classification|regression|clustering)\b"
                ]
            },
            "Deep Learning (DL)": {
                "keywords": [
                    "deep learning", "deep neural networks", "cnn", "rnn", "lstm", "gru",
                    "convolutional neural networks", "recurrent neural networks", "transformer",
                    "attention mechanisms", "backpropagation", "gradient descent", "tensorflow",
                    "pytorch", "keras", "computer vision", "natural language processing"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bdeep learning\b",
                    r"\b(?:cnn|rnn|lstm|gru)\b",
                    r"\bconvolutional neural networks?\b",
                    r"\brecurrent neural networks?\b",
                    r"\btransformer\b",
                    r"\battention mechanisms?\b"
                ]
            },
            "Natural Language Processing (NLP)": {
                "keywords": [
                    "natural language processing", "nlp", "text processing", "language models",
                    "sentiment analysis", "text classification", "named entity recognition",
                    "part-of-speech tagging", "parsing", "language understanding", "chatbots",
                    "speech recognition", "text generation", "word embeddings", "bert", "gpt"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bnatural language processing\b",
                    r"\bnlp\b",
                    r"\btext (?:processing|analysis|classification)\b",
                    r"\bsentiment analysis\b",
                    r"\blanguage models?\b",
                    r"\b(?:bert|gpt|transformer)\b"
                ]
            },
            "Computer Vision": {
                "keywords": [
                    "computer vision", "image processing", "image recognition", "object detection",
                    "face recognition", "optical character recognition", "ocr", "image segmentation",
                    "feature extraction", "edge detection", "image classification", "visual recognition",
                    "video analysis", "pattern recognition", "image enhancement"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bcomputer vision\b",
                    r"\bimage (?:processing|recognition|classification|analysis)\b",
                    r"\bobject detection\b",
                    r"\bface recognition\b",
                    r"\boptical character recognition\b",
                    r"\bimage segmentation\b"
                ]
            },
            "Data Science": {
                "keywords": [
                    "data science", "data analysis", "data mining", "big data", "data visualization",
                    "statistical analysis", "predictive analytics", "data modeling", "data warehousing",
                    "etl", "business intelligence", "data engineering", "data pipelines",
                    "exploratory data analysis", "eda", "data preprocessing"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bdata science\b",
                    r"\bdata (?:analysis|mining|modeling|engineering)\b",
                    r"\bbig data\b",
                    r"\bdata visualization\b",
                    r"\bstatistical analysis\b",
                    r"\bpredictive analytics\b"
                ]
            },
            "Cybersecurity": {
                "keywords": [
                    "cybersecurity", "cyber security", "information security", "network security",
                    "cryptography", "encryption", "digital forensics", "malware analysis",
                    "penetration testing", "vulnerability assessment", "intrusion detection",
                    "firewall", "authentication", "authorization", "privacy protection"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\b(?:cyber|information|network) security\b",
                    r"\bcybersecurity\b",
                    r"\bcryptography\b",
                    r"\bdigital forensics\b",
                    r"\bmalware analysis\b",
                    r"\bintrusion detection\b"
                ]
            },
            "Software Engineering": {
                "keywords": [
                    "software engineering", "software development", "programming", "software architecture",
                    "software design", "software testing", "quality assurance", "software maintenance",
                    "version control", "agile development", "devops", "continuous integration",
                    "code review", "software metrics", "software project management"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bsoftware engineering\b",
                    r"\bsoftware (?:development|architecture|design|testing)\b",
                    r"\bprogramming\b",
                    r"\bagile development\b",
                    r"\bdevops\b",
                    r"\bcontinuous integration\b"
                ]
            },
            "Database Systems": {
                "keywords": [
                    "database", "database systems", "sql", "nosql", "data modeling", "database design",
                    "query optimization", "transaction processing", "data warehousing", "data mining",
                    "relational databases", "distributed databases", "database administration",
                    "data integrity", "data consistency", "acid properties"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bdatabase (?:systems?|design|administration)\b",
                    r"\b(?:sql|nosql)\b",
                    r"\bdata modeling\b",
                    r"\bquery optimization\b",
                    r"\btransaction processing\b",
                    r"\brelational databases?\b"
                ]
            },
            "Computer Networks": {
                "keywords": [
                    "computer networks", "network protocols", "tcp/ip", "wireless networks",
                    "network security", "network performance", "routing algorithms", "network topology",
                    "bandwidth", "latency", "network monitoring", "network management",
                    "internet protocols", "mobile networks", "sensor networks"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bcomputer networks?\b",
                    r"\bnetwork (?:protocols?|security|performance|monitoring)\b",
                    r"\btcp/ip\b",
                    r"\bwireless networks?\b",
                    r"\brouting algorithms?\b"
                ]
            },
            "Operating Systems": {
                "keywords": [
                    "operating systems", "os", "process management", "memory management",
                    "file systems", "device drivers", "system calls", "kernel", "scheduling",
                    "concurrency", "threading", "multiprocessing", "virtual memory",
                    "system administration", "performance optimization"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\boperating systems?\b",
                    r"\b(?:process|memory) management\b",
                    r"\bfile systems?\b",
                    r"\bdevice drivers?\b",
                    r"\bsystem calls?\b",
                    r"\bkernel\b"
                ]
            },
            "Algorithms": {
                "keywords": [
                    "algorithms", "algorithm design", "complexity analysis", "data structures",
                    "sorting algorithms", "search algorithms", "graph algorithms", "dynamic programming",
                    "greedy algorithms", "divide and conquer", "recursion", "optimization",
                    "computational complexity", "big o notation", "algorithm efficiency"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\balgorithms?\b",
                    r"\balgorithm (?:design|efficiency)\b",
                    r"\bcomplexity analysis\b",
                    r"\bdata structures?\b",
                    r"\b(?:sorting|search|graph) algorithms?\b",
                    r"\bdynamic programming\b"
                ]
            },
            "Computer Graphics": {
                "keywords": [
                    "computer graphics", "graphics programming", "3d graphics", "rendering",
                    "animation", "visualization", "graphics algorithms", "opengl", "directx",
                    "ray tracing", "shader programming", "texture mapping", "lighting models",
                    "geometric modeling", "computer animation"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bcomputer graphics\b",
                    r"\bgraphics (?:programming|algorithms)\b",
                    r"\b3d graphics\b",
                    r"\brendering\b",
                    r"\b(?:opengl|directx)\b",
                    r"\bray tracing\b"
                ]
            },
            "Human-Computer Interaction (HCI)": {
                "keywords": [
                    "human-computer interaction", "hci", "user interface", "user experience",
                    "usability", "user-centered design", "interface design", "interaction design",
                    "accessibility", "user testing", "user research", "information architecture",
                    "visual design", "interactive systems", "user behavior"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bhuman-computer interaction\b",
                    r"\bhci\b",
                    r"\buser (?:interface|experience|testing|research)\b",
                    r"\busability\b",
                    r"\binterface design\b",
                    r"\binteraction design\b"
                ]
            },
            "Information Systems": {
                "keywords": [
                    "information systems", "management information systems", "mis",
                    "enterprise systems", "business systems", "system analysis", "system design",
                    "requirements analysis", "system implementation", "system integration",
                    "business process modeling", "workflow management", "enterprise architecture"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\binformation systems?\b",
                    r"\bmanagement information systems?\b",
                    r"\bmis\b",
                    r"\benterprise systems?\b",
                    r"\bsystem (?:analysis|design|implementation)\b",
                    r"\bbusiness systems?\b"
                ]
            },
            "Web Technologies": {
                "keywords": [
                    "web technologies", "web development", "html", "css", "javascript",
                    "web applications", "web services", "rest api", "web frameworks",
                    "frontend development", "backend development", "full stack", "responsive design",
                    "web security", "web performance", "progressive web apps"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bweb (?:technologies|development|applications?|services?)\b",
                    r"\b(?:html|css|javascript)\b",
                    r"\bweb frameworks?\b",
                    r"\b(?:frontend|backend) development\b",
                    r"\brest api\b",
                    r"\bresponsive design\b"
                ]
            },
            "Mobile Computing": {
                "keywords": [
                    "mobile computing", "mobile applications", "mobile development", "android",
                    "ios", "mobile platforms", "mobile ui/ux", "mobile security", "mobile networks",
                    "wireless communication", "mobile cloud computing", "mobile data management",
                    "location-based services", "mobile sensors", "mobile optimization"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bmobile (?:computing|applications?|development|platforms?)\b",
                    r"\b(?:android|ios)\b",
                    r"\bmobile (?:ui/ux|security|networks?)\b",
                    r"\bwireless communication\b",
                    r"\bmobile cloud computing\b"
                ]
            },
            "Cloud Computing": {
                "keywords": [
                    "cloud computing", "cloud services", "aws", "azure", "google cloud",
                    "virtualization", "containerization", "docker", "kubernetes", "microservices",
                    "serverless computing", "cloud storage", "cloud security", "cloud migration",
                    "infrastructure as a service", "platform as a service", "software as a service"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bcloud (?:computing|services?|storage|security|migration)\b",
                    r"\b(?:aws|azure|google cloud)\b",
                    r"\bvirtualization\b",
                    r"\b(?:docker|kubernetes)\b",
                    r"\bmicroservices\b",
                    r"\bserverless computing\b"
                ]
            },
            "Big Data": {
                "keywords": [
                    "big data", "data analytics", "data processing", "hadoop", "spark",
                    "data lakes", "data warehouses", "stream processing", "batch processing",
                    "real-time analytics", "data ingestion", "data transformation",
                    "distributed computing", "scalability", "data governance"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bbig data\b",
                    r"\bdata (?:analytics|processing|lakes?|warehouses?)\b",
                    r"\b(?:hadoop|spark)\b",
                    r"\bstream processing\b",
                    r"\bbatch processing\b",
                    r"\breal-time analytics\b"
                ]
            },
            "Internet of Things (IoT)": {
                "keywords": [
                    "internet of things", "iot", "smart devices", "sensor networks", "embedded systems",
                    "wireless sensor networks", "smart cities", "smart homes", "industrial iot",
                    "iot security", "iot protocols", "edge computing", "fog computing",
                    "device connectivity", "iot platforms", "iot analytics"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\binternet of things\b",
                    r"\biot\b",
                    r"\bsmart (?:devices?|cities?|homes?)\b",
                    r"\bsensor networks?\b",
                    r"\bembedded systems?\b",
                    r"\bwireless sensor networks?\b"
                ]
            },
            "Blockchain": {
                "keywords": [
                    "blockchain", "distributed ledger", "cryptocurrency", "bitcoin", "ethereum",
                    "smart contracts", "consensus algorithms", "decentralized systems",
                    "cryptographic hash", "mining", "wallet", "defi", "nft", "web3",
                    "blockchain security", "blockchain applications"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bblockchain\b",
                    r"\bdistributed ledger\b",
                    r"\b(?:cryptocurrency|bitcoin|ethereum)\b",
                    r"\bsmart contracts?\b",
                    r"\bconsensus algorithms?\b",
                    r"\bdecentralized systems?\b"
                ]
            },
            "Digital Forensics": {
                "keywords": [
                    "digital forensics", "computer forensics", "cyber forensics", "digital evidence",
                    "incident response", "malware analysis", "network forensics", "mobile forensics",
                    "data recovery", "file system analysis", "memory analysis", "timeline analysis",
                    "forensic tools", "chain of custody", "digital investigation"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\b(?:digital|computer|cyber) forensics\b",
                    r"\bdigital evidence\b",
                    r"\bincident response\b",
                    r"\bmalware analysis\b",
                    r"\bnetwork forensics\b",
                    r"\bdata recovery\b"
                ]
            },
            "Robotics": {
                "keywords": [
                    "robotics", "robotic systems", "autonomous robots", "robot control",
                    "path planning", "robot navigation", "manipulation", "humanoid robots",
                    "industrial robots", "service robots", "robot learning", "robot perception",
                    "robot localization", "robot mapping", "swarm robotics"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\brobotics\b",
                    r"\brobotic systems?\b",
                    r"\bautonomous robots?\b",
                    r"\brobot (?:control|navigation|learning|perception)\b",
                    r"\bpath planning\b",
                    r"\bmanipulation\b"
                ]
            },
            "Quantum Computing": {
                "keywords": [
                    "quantum computing", "quantum algorithms", "quantum mechanics", "qubits",
                    "quantum gates", "quantum entanglement", "quantum cryptography", "quantum simulation",
                    "quantum machine learning", "quantum optimization", "quantum error correction",
                    "quantum supremacy", "quantum annealing", "adiabatic quantum computing"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bquantum computing\b",
                    r"\bquantum (?:algorithms?|mechanics|cryptography|simulation)\b",
                    r"\bqubits?\b",
                    r"\bquantum gates?\b",
                    r"\bquantum entanglement\b",
                    r"\bquantum machine learning\b"
                ]
            },
            "Materials Science": {
                "keywords": [
                    "materials science", "materials engineering", "nanomaterials", "composite materials",
                    "metallic materials", "ceramic materials", "polymeric materials", "biomaterials",
                    "smart materials", "materials characterization", "materials testing",
                    "materials processing", "surface engineering", "corrosion engineering"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bmaterials science\b",
                    r"\bmaterials (?:engineering|characterization|testing|processing)\b",
                    r"\bnanomaterials?\b",
                    r"\bcomposite materials?\b",
                    r"\b(?:metallic|ceramic|polymeric) materials?\b",
                    r"\bbio materials?\b"
                ]
            },
            "Renewable Energy": {
                "keywords": [
                    "renewable energy", "solar energy", "wind energy", "hydroelectric power",
                    "geothermal energy", "biomass energy", "energy storage", "battery technology",
                    "smart grids", "energy efficiency", "sustainable energy", "clean energy",
                    "photovoltaic", "solar panels", "wind turbines", "energy systems"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\brenewable energy\b",
                    r"\b(?:solar|wind|hydroelectric|geothermal|biomass) energy\b",
                    r"\benergy (?:storage|efficiency|systems?)\b",
                    r"\bbattery technology\b",
                    r"\bsmart grids?\b",
                    r"\bsustainable energy\b"
                ]
            },
            "Biomedical Engineering": {
                "keywords": [
                    "biomedical engineering", "medical devices", "biomedical instrumentation",
                    "medical imaging", "biomedical signal processing", "prosthetics", "implants",
                    "biomaterials", "tissue engineering", "medical robotics", "diagnostic systems",
                    "therapeutic systems", "biomedical sensors", "healthcare technology"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\bbiomedical engineering\b",
                    r"\bmedical (?:devices?|imaging|robotics)\b",
                    r"\bbiomedical (?:instrumentation|signal processing|sensors?)\b",
                    r"\b(?:prosthetics|implants)\b",
                    r"\btissue engineering\b",
                    r"\bdiagnostic systems?\b"
                ]
            },
            "Environmental Engineering": {
                "keywords": [
                    "environmental engineering", "environmental systems", "pollution control",
                    "waste management", "water treatment", "air quality", "environmental monitoring",
                    "sustainable development", "environmental impact assessment", "green technology",
                    "climate change", "carbon footprint", "environmental remediation"
                ],
                "weight": 3.0,
                "patterns": [
                    r"\benvironmental engineering\b",
                    r"\benvironmental (?:systems?|monitoring)\b",
                    r"\bpollution control\b",
                    r"\bwaste management\b",
                    r"\bwater treatment\b",
                    r"\bair quality\b"
                ]
            }
        }
    
    def _create_keyword_patterns(self) -> Dict[str, List[str]]:
        """Create compiled regex patterns for efficient matching."""
        patterns = {}
        for domain, info in self.domains.items():
            patterns[domain] = [re.compile(pattern, re.IGNORECASE) for pattern in info.get('patterns', [])]
        return patterns
    
    def assign_domain(self, title: str, abstract: str = "", keywords: List[str] = None) -> DomainMatch:
        """
        Assign the most appropriate research domain based on content analysis.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            keywords: List of keywords
            
        Returns:
            DomainMatch object with the best matching domain
        """
        if keywords is None:
            keywords = []
        
        # Combine all text for analysis
        text = f"{title} {abstract} {' '.join(keywords)}".lower()
        
        # Calculate scores for each domain
        domain_scores = {}
        
        for domain, info in self.domains.items():
            score = 0.0
            matched_keywords = []
            
            # Check keyword matches
            for keyword in info['keywords']:
                if keyword.lower() in text:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # Check pattern matches (higher weight)
            patterns = self.keyword_patterns.get(domain, [])
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    score += len(matches) * 2.0  # Higher weight for pattern matches
                    matched_keywords.extend(matches)
            
            # Apply domain weight
            weight = info.get('weight', 1.0)
            final_score = score * weight
            
            # Calculate confidence based on score and text length
            confidence = min(final_score / max(len(text.split()) / 10, 1), 1.0)
            
            domain_scores[domain] = {
                'score': final_score,
                'confidence': confidence,
                'matched_keywords': list(set(matched_keywords)),
                'weight': weight
            }
        
        # Find the best match
        if not domain_scores or max(domain_scores.values(), key=lambda x: x['score'])['score'] == 0:
            return DomainMatch(
                domain="Computer Science & Engineering",
                confidence=0.1,
                matched_keywords=[],
                score=0.0,
                reason="No specific domain keywords found, defaulting to general CS"
            )
        
        best_domain = max(domain_scores.items(), key=lambda x: x[1]['score'])
        domain_name, match_info = best_domain
        
        # Generate reason
        if match_info['matched_keywords']:
            reason = f"Matched keywords: {', '.join(match_info['matched_keywords'][:5])}"
            if len(match_info['matched_keywords']) > 5:
                reason += f" and {len(match_info['matched_keywords']) - 5} more"
        else:
            reason = "Pattern-based matching"
        
        return DomainMatch(
            domain=domain_name,
            confidence=match_info['confidence'],
            matched_keywords=match_info['matched_keywords'],
            score=match_info['score'],
            reason=reason
        )
    
    def get_top_domains(self, title: str, abstract: str = "", keywords: List[str] = None, top_n: int = 3) -> List[DomainMatch]:
        """
        Get top N matching domains.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            keywords: List of keywords
            top_n: Number of top domains to return
            
        Returns:
            List of DomainMatch objects sorted by score
        """
        if keywords is None:
            keywords = []
        
        text = f"{title} {abstract} {' '.join(keywords)}".lower()
        domain_matches = []
        
        for domain, info in self.domains.items():
            score = 0.0
            matched_keywords = []
            
            # Check keyword matches
            for keyword in info['keywords']:
                if keyword.lower() in text:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # Check pattern matches
            patterns = self.keyword_patterns.get(domain, [])
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    score += len(matches) * 2.0
                    matched_keywords.extend(matches)
            
            # Apply weight
            weight = info.get('weight', 1.0)
            final_score = score * weight
            confidence = min(final_score / max(len(text.split()) / 10, 1), 1.0)
            
            if final_score > 0:
                domain_matches.append(DomainMatch(
                    domain=domain,
                    confidence=confidence,
                    matched_keywords=list(set(matched_keywords)),
                    score=final_score,
                    reason=f"Score: {final_score:.1f}"
                ))
        
        # Sort by score and return top N
        domain_matches.sort(key=lambda x: x.score, reverse=True)
        return domain_matches[:top_n]
    
    def get_available_domains(self) -> List[str]:
        """Get list of all available research domains."""
        return list(self.domains.keys())
    
    def get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords for a specific domain."""
        return self.domains.get(domain, {}).get('keywords', [])
    
    def add_custom_domain(self, domain_name: str, keywords: List[str], patterns: List[str] = None, weight: float = 1.0):
        """Add a custom research domain."""
        self.domains[domain_name] = {
            'keywords': keywords,
            'patterns': patterns or [],
            'weight': weight
        }
        
        # Update patterns
        if patterns:
            self.keyword_patterns[domain_name] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        else:
            self.keyword_patterns[domain_name] = []
        
        logger.info(f"Added custom domain: {domain_name} with {len(keywords)} keywords")


# Global instance
domain_assigner = ResearchDomainAssigner()


def assign_research_domain(title: str, abstract: str = "", keywords: List[str] = None) -> str:
    """
    Assign research domain based on paper content.
    
    Args:
        title: Paper title
        abstract: Paper abstract  
        keywords: List of keywords
        
    Returns:
        Best matching research domain name
    """
    match = domain_assigner.assign_domain(title, abstract, keywords)
    return match.domain


def get_top_research_domains(title: str, abstract: str = "", keywords: List[str] = None, top_n: int = 3) -> List[str]:
    """
    Get top N research domains based on paper content.
    
    Args:
        title: Paper title
        abstract: Paper abstract
        keywords: List of keywords
        top_n: Number of top domains to return
        
    Returns:
        List of domain names sorted by relevance
    """
    matches = domain_assigner.get_top_domains(title, abstract, keywords, top_n)
    return [match.domain for match in matches]


def get_domain_confidence(title: str, abstract: str = "", keywords: List[str] = None) -> float:
    """
    Get confidence score for the best matching domain.
    
    Args:
        title: Paper title
        abstract: Paper abstract
        keywords: List of keywords
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    match = domain_assigner.assign_domain(title, abstract, keywords)
    return match.confidence





















