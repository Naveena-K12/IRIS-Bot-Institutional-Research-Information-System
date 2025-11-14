"""
Research Domain Classifier
Automatically assigns research domains based on paper content and metadata.
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DomainClassification:
    """Container for domain classification results."""
    primary_domain: str = ""
    confidence: float = 0.0
    alternative_domains: List[str] = None
    keywords_found: List[str] = None
    source: str = ""  # Where classification came from
    
    def __post_init__(self):
        if self.alternative_domains is None:
            self.alternative_domains = []
        if self.keywords_found is None:
            self.keywords_found = []


class ResearchDomainClassifier:
    """Classifies papers into research domains based on content analysis."""
    
    def __init__(self):
        """Initialize the classifier with domain definitions."""
        self.domains = self._load_domains()
        self.keyword_patterns = self._create_keyword_patterns()
    
    def _load_domains(self) -> Dict[str, Dict]:
        """Load all research domains with their keywords."""
        return {
            # AI/ML Domains
            "Artificial Intelligence (AI)": {
                "keywords": ["artificial intelligence", "ai", "intelligent systems", "cognitive computing", "expert systems"],
                "priority": 1
            },
            "Machine Learning (ML)": {
                "keywords": ["machine learning", "ml", "supervised learning", "unsupervised learning", "learning algorithms", "pattern recognition"],
                "priority": 1
            },
            "Deep Learning (DL)": {
                "keywords": ["deep learning", "dl", "neural networks", "cnn", "rnn", "lstm", "transformer", "autoencoder", "gan"],
                "priority": 1
            },
            "Natural Language Processing (NLP)": {
                "keywords": ["natural language processing", "nlp", "text mining", "sentiment analysis", "language models", "word embedding", "bert", "gpt"],
                "priority": 1
            },
            "Computer Vision": {
                "keywords": ["computer vision", "image processing", "object detection", "image segmentation", "face recognition", "optical flow", "stereo vision"],
                "priority": 1
            },
            "Reinforcement Learning": {
                "keywords": ["reinforcement learning", "rl", "q-learning", "policy gradient", "actor-critic", "multi-agent", "reward function"],
                "priority": 1
            },
            "Explainable AI (XAI)": {
                "keywords": ["explainable ai", "xai", "interpretability", "model explanation", "black box", "transparent ai"],
                "priority": 2
            },
            "Generative AI": {
                "keywords": ["generative ai", "generative models", "gans", "variational autoencoder", "diffusion models", "text generation", "image generation"],
                "priority": 1
            },
            
            # Data Science
            "Data Science & Analytics": {
                "keywords": ["data science", "data analytics", "data analysis", "business intelligence", "data-driven"],
                "priority": 1
            },
            "Data Mining": {
                "keywords": ["data mining", "knowledge discovery", "association rules", "clustering", "classification", "outlier detection"],
                "priority": 1
            },
            "Big Data": {
                "keywords": ["big data", "hadoop", "spark", "distributed computing", "data warehouse", "etl"],
                "priority": 1
            },
            "Statistical Learning": {
                "keywords": ["statistical learning", "statistical modeling", "regression", "bayesian", "hypothesis testing"],
                "priority": 2
            },
            "Information Retrieval": {
                "keywords": ["information retrieval", "search engines", "recommendation systems", "ranking", "tf-idf"],
                "priority": 2
            },
            
            # Software Engineering
            "Software Engineering": {
                "keywords": ["software engineering", "software development", "software lifecycle", "requirements engineering"],
                "priority": 1
            },
            "Software Design & Architecture": {
                "keywords": ["software architecture", "design patterns", "microservices", "component design", "system design"],
                "priority": 2
            },
            "Software Testing": {
                "keywords": ["software testing", "unit testing", "integration testing", "test automation", "quality assurance"],
                "priority": 2
            },
            "Agile / DevOps": {
                "keywords": ["agile", "devops", "continuous integration", "continuous deployment", "scrum", "kanban"],
                "priority": 2
            },
            
            # Systems
            "Operating Systems": {
                "keywords": ["operating systems", "kernel", "process management", "memory management", "file systems"],
                "priority": 1
            },
            "Distributed Systems": {
                "keywords": ["distributed systems", "distributed computing", "consensus", "fault tolerance", "load balancing"],
                "priority": 1
            },
            "Cloud Computing": {
                "keywords": ["cloud computing", "aws", "azure", "google cloud", "virtualization", "saas", "paas", "iaas"],
                "priority": 1
            },
            "Edge / Fog Computing": {
                "keywords": ["edge computing", "fog computing", "edge devices", "mobile edge", "iot edge"],
                "priority": 2
            },
            "Internet of Things (IoT)": {
                "keywords": ["internet of things", "iot", "smart devices", "sensors", "embedded systems", "wireless sensor networks"],
                "priority": 1
            },
            
            # Security & Privacy
            "Cybersecurity & Privacy": {
                "keywords": ["cybersecurity", "cyber security", "information security", "privacy", "encryption", "authentication", "firewall"],
                "priority": 1
            },
            
            # HCI & Design
            "Human-Computer Interaction (HCI)": {
                "keywords": ["human-computer interaction", "hci", "user interface", "usability", "user experience"],
                "priority": 1
            },
            "UX / UI Design": {
                "keywords": ["user experience", "ux", "user interface", "ui", "interface design", "interaction design"],
                "priority": 2
            },
            "Accessibility": {
                "keywords": ["accessibility", "assistive technology", "universal design", "inclusive design"],
                "priority": 3
            },
            
            # Theoretical CS
            "Algorithms & Complexity": {
                "keywords": ["algorithms", "computational complexity", "algorithm design", "optimization", "graph algorithms"],
                "priority": 1
            },
            "Computation Theory": {
                "keywords": ["computation theory", "automata", "formal languages", "computability", "turing machines"],
                "priority": 2
            },
            "Quantum Computing": {
                "keywords": ["quantum computing", "quantum algorithms", "quantum mechanics", "qubits", "quantum gates"],
                "priority": 1
            },
            
            # Robotics & Autonomous Systems
            "Robotics & Autonomous Systems": {
                "keywords": ["robotics", "autonomous systems", "autonomous vehicles", "robot control", "path planning"],
                "priority": 1
            },
            
            # Bioinformatics
            "Bioinformatics & Computational Biology": {
                "keywords": ["bioinformatics", "computational biology", "genomics", "proteomics", "sequence analysis"],
                "priority": 1
            },
            
            # Graphics & Visualization
            "Computer Graphics & Visualization": {
                "keywords": ["computer graphics", "visualization", "rendering", "3d modeling", "animation"],
                "priority": 1
            },
            
            # Databases
            "Databases & Information Systems": {
                "keywords": ["databases", "database systems", "sql", "nosql", "data management", "information systems"],
                "priority": 1
            },
            
            # Embedded Systems
            "Embedded Systems": {
                "keywords": ["embedded systems", "microcontrollers", "real-time systems", "firmware"],
                "priority": 1
            },
            
            # Signal Processing
            "Signal Processing": {
                "keywords": ["signal processing", "digital signal processing", "dsp", "fourier transform", "filtering"],
                "priority": 1
            },
            
            # Communication
            "Wireless Communication": {
                "keywords": ["wireless communication", "5g", "wifi", "bluetooth", "mobile communication", "telecommunications"],
                "priority": 1
            },
            
            # Hardware
            "VLSI Design": {
                "keywords": ["vlsi", "chip design", "integrated circuits", "semiconductor", "asic"],
                "priority": 1
            },
            "Circuit Design": {
                "keywords": ["circuit design", "electronic circuits", "analog circuits", "digital circuits"],
                "priority": 1
            },
            "Microprocessors & Microcontrollers": {
                "keywords": ["microprocessors", "microcontrollers", "cpu", "processor design", "arm"],
                "priority": 2
            },
            
            # Control Systems
            "Control Systems": {
                "keywords": ["control systems", "feedback control", "pid control", "system control", "automation"],
                "priority": 1
            },
            
            # Power & Energy
            "Power Electronics": {
                "keywords": ["power electronics", "power systems", "electrical power", "power conversion"],
                "priority": 1
            },
            "Renewable Energy Systems": {
                "keywords": ["renewable energy", "solar energy", "wind energy", "clean energy", "sustainable energy"],
                "priority": 1
            },
            "Energy Storage": {
                "keywords": ["energy storage", "batteries", "energy storage systems", "grid storage"],
                "priority": 2
            },
            
            # Sensors
            "Sensors & Instrumentation": {
                "keywords": ["sensors", "instrumentation", "measurement", "sensor networks", "actuators"],
                "priority": 1
            },
            
            # Physics
            "Theoretical Physics": {
                "keywords": ["theoretical physics", "quantum physics", "particle physics", "condensed matter"],
                "priority": 1
            },
            "Quantum Physics": {
                "keywords": ["quantum physics", "quantum mechanics", "quantum theory", "quantum states"],
                "priority": 1
            },
            "Condensed Matter Physics": {
                "keywords": ["condensed matter", "solid state physics", "materials physics", "crystallography"],
                "priority": 2
            },
            "Particle Physics": {
                "keywords": ["particle physics", "high energy physics", "accelerators", "elementary particles"],
                "priority": 2
            },
            "Astrophysics": {
                "keywords": ["astrophysics", "astronomy", "cosmology", "stellar physics", "galaxies"],
                "priority": 2
            },
            "Nuclear Physics": {
                "keywords": ["nuclear physics", "nuclear reactions", "radioactivity", "nuclear energy"],
                "priority": 2
            },
            "Optics & Photonics": {
                "keywords": ["optics", "photonics", "lasers", "optical systems", "photonic devices"],
                "priority": 1
            },
            "Plasma Physics": {
                "keywords": ["plasma physics", "plasma", "fusion", "plasma dynamics"],
                "priority": 3
            },
            
            # Mathematics
            "Pure Mathematics": {
                "keywords": ["pure mathematics", "abstract algebra", "topology", "number theory", "geometry"],
                "priority": 1
            },
            "Applied Mathematics": {
                "keywords": ["applied mathematics", "mathematical modeling", "differential equations", "linear algebra"],
                "priority": 1
            },
            "Computational Mathematics": {
                "keywords": ["computational mathematics", "numerical analysis", "scientific computing", "finite element"],
                "priority": 1
            },
            "Probability & Statistics": {
                "keywords": ["probability", "statistics", "statistical analysis", "random variables", "bayesian"],
                "priority": 1
            },
            "Operations Research": {
                "keywords": ["operations research", "optimization", "linear programming", "decision analysis"],
                "priority": 2
            },
            "Mathematical Modelling": {
                "keywords": ["mathematical modeling", "mathematical models", "simulation", "modeling"],
                "priority": 2
            },
            
            # Biology & Medicine
            "Molecular Biology": {
                "keywords": ["molecular biology", "dna", "rna", "proteins", "gene expression"],
                "priority": 1
            },
            "Genetics & Genomics": {
                "keywords": ["genetics", "genomics", "genome", "genetic variation", "heredity"],
                "priority": 1
            },
            "Microbiology": {
                "keywords": ["microbiology", "bacteria", "viruses", "microorganisms", "pathogens"],
                "priority": 1
            },
            "Biotechnology": {
                "keywords": ["biotechnology", "biotech", "genetic engineering", "bioprocessing"],
                "priority": 1
            },
            "Ecology & Evolution": {
                "keywords": ["ecology", "evolution", "ecosystems", "biodiversity", "conservation"],
                "priority": 1
            },
            "Neuroscience": {
                "keywords": ["neuroscience", "brain", "neural networks", "cognitive neuroscience"],
                "priority": 1
            },
            "Biophysics": {
                "keywords": ["biophysics", "biological physics", "molecular biophysics"],
                "priority": 2
            },
            "Clinical Medicine": {
                "keywords": ["clinical medicine", "medical diagnosis", "treatment", "patient care"],
                "priority": 1
            },
            "Public Health": {
                "keywords": ["public health", "epidemiology", "health policy", "preventive medicine"],
                "priority": 1
            },
            "Biomedical Engineering": {
                "keywords": ["biomedical engineering", "medical devices", "biomedical systems", "healthcare technology"],
                "priority": 1
            },
            "Medical Imaging": {
                "keywords": ["medical imaging", "mri", "ct scan", "ultrasound", "x-ray", "medical image analysis"],
                "priority": 1
            },
            "Pharmacology": {
                "keywords": ["pharmacology", "drug development", "pharmaceuticals", "drug interactions"],
                "priority": 1
            },
            "Epidemiology": {
                "keywords": ["epidemiology", "disease surveillance", "public health", "disease patterns"],
                "priority": 2
            },
            "Genomic Medicine": {
                "keywords": ["genomic medicine", "precision medicine", "personalized medicine", "pharmacogenomics"],
                "priority": 2
            },
            
            # Materials Science
            "Nanomaterials": {
                "keywords": ["nanomaterials", "nanotechnology", "nanoparticles", "nano", "nanostructures"],
                "priority": 1
            },
            "Smart Materials": {
                "keywords": ["smart materials", "shape memory", "responsive materials", "adaptive materials"],
                "priority": 2
            },
            "Composite Materials": {
                "keywords": ["composite materials", "composites", "fiber reinforced", "polymer composites"],
                "priority": 1
            },
            "Surface Engineering": {
                "keywords": ["surface engineering", "surface treatment", "coatings", "surface modification"],
                "priority": 2
            },
            "Semiconductor Materials": {
                "keywords": ["semiconductor materials", "semiconductors", "silicon", "gallium arsenide"],
                "priority": 1
            },
            
            # Energy & Environment
            "Renewable Energy": {
                "keywords": ["renewable energy", "solar", "wind", "hydroelectric", "geothermal"],
                "priority": 1
            },
            "Sustainable Technologies": {
                "keywords": ["sustainable technologies", "sustainability", "green technology", "eco-friendly"],
                "priority": 1
            },
            "Climate Change": {
                "keywords": ["climate change", "global warming", "carbon emissions", "environmental impact"],
                "priority": 1
            },
            "Environmental Monitoring": {
                "keywords": ["environmental monitoring", "pollution monitoring", "environmental sensors"],
                "priority": 2
            },
            "Green Technology": {
                "keywords": ["green technology", "clean technology", "environmental technology"],
                "priority": 2
            },
            
            # Social Sciences
            "Psychology": {
                "keywords": ["psychology", "cognitive psychology", "behavioral psychology", "mental health"],
                "priority": 1
            },
            "Sociology": {
                "keywords": ["sociology", "social behavior", "social systems", "social research"],
                "priority": 1
            },
            "Political Science": {
                "keywords": ["political science", "politics", "governance", "public policy"],
                "priority": 1
            },
            "Economics": {
                "keywords": ["economics", "economic analysis", "microeconomics", "macroeconomics"],
                "priority": 1
            },
            "Education Technology (EdTech)": {
                "keywords": ["education technology", "edtech", "e-learning", "educational software", "online learning"],
                "priority": 1
            },
            "Linguistics": {
                "keywords": ["linguistics", "language", "linguistic analysis", "computational linguistics"],
                "priority": 2
            },
            "Philosophy": {
                "keywords": ["philosophy", "ethics", "logic", "philosophical analysis"],
                "priority": 2
            },
            "Anthropology": {
                "keywords": ["anthropology", "cultural anthropology", "social anthropology"],
                "priority": 2
            },
            "Law & Policy": {
                "keywords": ["law", "legal", "policy", "regulation", "governance"],
                "priority": 2
            },
            
            # Business & Management
            "Business Analytics": {
                "keywords": ["business analytics", "business intelligence", "data analytics", "business intelligence"],
                "priority": 1
            },
            "Operations Management": {
                "keywords": ["operations management", "production management", "manufacturing", "logistics"],
                "priority": 1
            },
            "Supply Chain": {
                "keywords": ["supply chain", "supply chain management", "logistics", "procurement"],
                "priority": 1
            },
            "Marketing & Consumer Behavior": {
                "keywords": ["marketing", "consumer behavior", "market research", "advertising"],
                "priority": 1
            },
            "Finance & Investment": {
                "keywords": ["finance", "investment", "financial markets", "portfolio management"],
                "priority": 1
            },
            "Entrepreneurship": {
                "keywords": ["entrepreneurship", "startups", "business development", "innovation"],
                "priority": 2
            },
            "Organizational Behavior": {
                "keywords": ["organizational behavior", "organizational psychology", "leadership", "management"],
                "priority": 2
            },
            
            # Emerging Technologies
            "Cyber-Physical Systems": {
                "keywords": ["cyber-physical systems", "cps", "embedded systems", "real-time systems"],
                "priority": 2
            },
            "Digital Twin": {
                "keywords": ["digital twin", "virtual twin", "simulation modeling"],
                "priority": 2
            },
            "Metaverse": {
                "keywords": ["metaverse", "virtual reality", "augmented reality", "vr", "ar"],
                "priority": 2
            },
            "Blockchain & Cryptography": {
                "keywords": ["blockchain", "cryptography", "cryptocurrency", "distributed ledger", "smart contracts"],
                "priority": 1
            },
            "FinTech": {
                "keywords": ["fintech", "financial technology", "digital banking", "payment systems"],
                "priority": 2
            },
            "HealthTech": {
                "keywords": ["healthtech", "health technology", "digital health", "telemedicine"],
                "priority": 2
            },
            "AgriTech": {
                "keywords": ["agritech", "agricultural technology", "precision agriculture", "smart farming"],
                "priority": 2
            },
            "Space Technology": {
                "keywords": ["space technology", "aerospace", "satellites", "space exploration"],
                "priority": 2
            },
            
            # Engineering Domains
            "Thermodynamics": {
                "keywords": ["thermodynamics", "heat transfer", "energy systems", "thermal analysis"],
                "priority": 1
            },
            "Fluid Mechanics": {
                "keywords": ["fluid mechanics", "fluid dynamics", "hydrodynamics", "aerodynamics"],
                "priority": 1
            },
            "Heat Transfer": {
                "keywords": ["heat transfer", "thermal conductivity", "convection", "radiation"],
                "priority": 1
            },
            "Robotics & Automation": {
                "keywords": ["robotics", "automation", "industrial robots", "automated systems"],
                "priority": 1
            },
            "CAD/CAM/CAE": {
                "keywords": ["cad", "cam", "cae", "computer-aided design", "manufacturing"],
                "priority": 1
            },
            "Materials Science": {
                "keywords": ["materials science", "materials engineering", "material properties", "material characterization"],
                "priority": 1
            },
            "Additive Manufacturing (3D Printing)": {
                "keywords": ["3d printing", "additive manufacturing", "rapid prototyping", "selective laser sintering"],
                "priority": 1
            },
            "Mechatronics": {
                "keywords": ["mechatronics", "mechanical systems", "electromechanical", "control systems"],
                "priority": 1
            },
            "Industrial Engineering": {
                "keywords": ["industrial engineering", "manufacturing engineering", "production systems"],
                "priority": 1
            },
            "Structural Engineering": {
                "keywords": ["structural engineering", "structural analysis", "civil engineering", "buildings"],
                "priority": 1
            },
            "Geotechnical Engineering": {
                "keywords": ["geotechnical engineering", "soil mechanics", "foundation engineering"],
                "priority": 2
            },
            "Transportation Engineering": {
                "keywords": ["transportation engineering", "traffic engineering", "highway engineering"],
                "priority": 2
            },
            "Construction Management": {
                "keywords": ["construction management", "project management", "construction planning"],
                "priority": 2
            },
            "Environmental Engineering": {
                "keywords": ["environmental engineering", "waste management", "water treatment", "air quality"],
                "priority": 1
            },
            "Water Resources Engineering": {
                "keywords": ["water resources", "hydrology", "water management", "irrigation"],
                "priority": 2
            },
            "Smart Cities & Infrastructure": {
                "keywords": ["smart cities", "urban planning", "smart infrastructure", "urban technology"],
                "priority": 2
            },
            "Chemical Reaction Engineering": {
                "keywords": ["chemical reaction engineering", "reactor design", "chemical processes"],
                "priority": 2
            },
            "Process Design & Optimization": {
                "keywords": ["process design", "process optimization", "chemical processes"],
                "priority": 2
            },
            "Polymer Science": {
                "keywords": ["polymer science", "polymers", "polymer chemistry", "polymer materials"],
                "priority": 2
            },
            "Catalysis": {
                "keywords": ["catalysis", "catalysts", "catalytic processes", "enzyme catalysis"],
                "priority": 3
            },
            "Biochemical Engineering": {
                "keywords": ["biochemical engineering", "bioprocess engineering", "fermentation"],
                "priority": 2
            },
            "Energy & Fuels": {
                "keywords": ["energy", "fuels", "energy systems", "fuel cells", "combustion"],
                "priority": 1
            }
        }
    
    def _create_keyword_patterns(self) -> Dict[str, re.Pattern]:
        """Create compiled regex patterns for efficient matching."""
        patterns = {}
        for domain, info in self.domains.items():
            # Create a pattern that matches any of the keywords
            keywords = info["keywords"]
            pattern_str = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            patterns[domain] = re.compile(pattern_str, re.IGNORECASE)
        return patterns
    
    def classify_domain(self, text: str, title: str = "", abstract: str = "", keywords: List[str] = None) -> DomainClassification:
        """
        Classify research domain based on content analysis.
        
        Args:
            text: Full text content
            title: Paper title
            abstract: Paper abstract
            keywords: List of keywords
            
        Returns:
            DomainClassification object
        """
        classification = DomainClassification()
        
        try:
            # Combine all text sources
            search_text = f"{title} {abstract} {text}".lower()
            if keywords:
                search_text += " " + " ".join(keywords).lower()
            
            # Score each domain
            domain_scores = {}
            found_keywords = {}
            
            for domain, pattern in self.keyword_patterns.items():
                matches = pattern.findall(search_text)
                if matches:
                    # Count unique matches
                    unique_matches = set(match.lower() for match in matches)
                    score = len(unique_matches)
                    
                    # Apply priority weighting
                    priority = self.domains[domain]["priority"]
                    weighted_score = score / priority
                    
                    domain_scores[domain] = weighted_score
                    found_keywords[domain] = list(unique_matches)
            
            if not domain_scores:
                # No keywords found, return empty classification
                classification.primary_domain = ""
                classification.confidence = 0.0
                classification.source = "No keywords matched"
                return classification
            
            # Find best domain
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = domain_scores[best_domain]
            
            # Calculate confidence (normalized score)
            max_possible_score = len(self.domains[best_domain]["keywords"]) / self.domains[best_domain]["priority"]
            confidence = min(best_score / max_possible_score, 1.0)
            
            # Set primary domain
            classification.primary_domain = best_domain
            classification.confidence = confidence
            classification.keywords_found = found_keywords[best_domain]
            classification.source = "Keyword matching"
            
            # Find alternative domains (other high-scoring domains)
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            for domain, score in sorted_domains[1:4]:  # Top 3 alternatives
                if score >= best_score * 0.5:  # At least 50% of best score
                    classification.alternative_domains.append(domain)
            
            logger.info(f"Domain classified as: {best_domain} (confidence: {confidence:.1%})")
            
        except Exception as e:
            logger.error(f"Error classifying domain: {e}")
            classification.primary_domain = ""
            classification.confidence = 0.0
            classification.source = f"Error: {str(e)}"
        
        return classification
    
    def get_all_domains(self) -> List[str]:
        """Get list of all available domains."""
        return list(self.domains.keys())
    
    def validate_domain(self, domain: str) -> bool:
        """Check if a domain is valid."""
        return domain in self.domains
    
    def add_custom_domain(self, domain: str, keywords: List[str], priority: int = 2) -> bool:
        """Add a custom domain to the classifier."""
        try:
            if domain not in self.domains:
                self.domains[domain] = {
                    "keywords": keywords,
                    "priority": priority
                }
                
                # Create pattern for new domain
                pattern_str = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
                self.keyword_patterns[domain] = re.compile(pattern_str, re.IGNORECASE)
                
                logger.info(f"Added custom domain: {domain}")
                return True
            else:
                logger.warning(f"Domain {domain} already exists")
                return False
                
        except Exception as e:
            logger.error(f"Error adding custom domain: {e}")
            return False
    
    def get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords for a specific domain."""
        return self.domains.get(domain, {}).get("keywords", [])
    
    def search_domains_by_keyword(self, keyword: str) -> List[str]:
        """Find domains that contain a specific keyword."""
        matching_domains = []
        keyword_lower = keyword.lower()
        
        for domain, info in self.domains.items():
            if any(keyword_lower in kw.lower() for kw in info["keywords"]):
                matching_domains.append(domain)
        
        return matching_domains


# Global instance
research_domain_classifier = ResearchDomainClassifier()


def classify_research_domain(text: str, title: str = "", abstract: str = "", keywords: List[str] = None) -> DomainClassification:
    """
    Convenience function to classify research domain.
    
    Args:
        text: Full text content
        title: Paper title
        abstract: Paper abstract
        keywords: List of keywords
        
    Returns:
        DomainClassification object
    """
    return research_domain_classifier.classify_domain(text, title, abstract, keywords)


def get_all_domains() -> List[str]:
    """Get list of all available domains."""
    return research_domain_classifier.get_all_domains()


def add_custom_domain(domain: str, keywords: List[str], priority: int = 2) -> bool:
    """Add a custom domain."""
    return research_domain_classifier.add_custom_domain(domain, keywords, priority)





















