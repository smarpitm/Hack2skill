"""
config.py

This is the single source of truth for all tunable parameters and configuration settings
used across the AI Candidate Ranking System pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "src" / "cache"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Create directories if they do not exist
for path in [DATA_DIR, MODELS_DIR, CACHE_DIR, SUBMISSIONS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MODEL NAMES
# ==============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# ==============================================================================
# TOP-K VALUES
# ==============================================================================
RETRIEVAL_K = 200
RANKER_K = 50
LLM_K = 20

# ==============================================================================
# SYNTHETIC LABEL WEIGHTS (must sum to 1.0)
# ==============================================================================
WEIGHT_SKILL_MATCH = 0.35
WEIGHT_SEMANTIC = 0.30
WEIGHT_EXPERIENCE = 0.20
WEIGHT_ACTIVITY = 0.15

# Ensure weights sum to 1.0
assert abs((WEIGHT_SKILL_MATCH + WEIGHT_SEMANTIC + WEIGHT_EXPERIENCE + WEIGHT_ACTIVITY) - 1.0) < 1e-9, \
    "Synthetic label weights must sum to 1.0"

SYNTHETIC_LABEL_THRESHOLD = 0.65

# ==============================================================================
# GROQ SETTINGS
# ==============================================================================
GROQ_DELAY_SECONDS = 3
GROQ_MAX_RETRIES = 1
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 1024

# ==============================================================================
# DEFAULT VALUES for missing data
# ==============================================================================
DEFAULT_PLATFORM_ACTIVITY = 0.5
DEFAULT_EXPERIENCE_MATCH = 0.5
DEFAULT_EDUCATION_MATCH = 0.5
DEFAULT_LOCATION_MATCH = 0.5

# ==============================================================================
# SKILL DICTIONARY (100+ technical skills)
# ==============================================================================
SKILL_DICTIONARY = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C", "C++", "C#", 
    ".NET", "Ruby", "PHP", "Scala", "Kotlin", "Swift", "Objective-C", "Dart", 
    "Shell Scripting", "Bash", "PowerShell", "R", "Julia", "MATLAB", "SQL",
    
    # Frontend Frameworks & Libraries
    "React", "Angular", "Vue", "Next.js", "Nuxt.js", "Svelte", "Redux", "HTML", 
    "CSS", "Sass", "Tailwind", "Bootstrap", "jQuery", "Webpack", "Vite", "Babel",
    
    # Backend Frameworks
    "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot", "Laravel", 
    "Ruby on Rails", "NestJS", "Koa",
    
    # Databases & Caching
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra", 
    "DynamoDB", "SQLite", "Oracle", "MSSQL", "MariaDB", "Neo4j", "Firebase", 
    "Supabase",
    
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", 
    "Jenkins", "GitHub Actions", "CI/CD", "Helm", "Docker Compose", "Nginx", 
    "Apache", "IIS", "HAProxy",
    
    # Message Brokers & Data Pipelines
    "Kafka", "RabbitMQ", "Airflow", "Spark", "Hadoop", "Flink", "Celery",
    
    # Data Science, ML & AI
    "Pandas", "NumPy", "SciPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", 
    "NLP", "Computer Vision", "Deep Learning", "Machine Learning", "MLOps", 
    "Data Science", "Data Engineering", "Analytics", "Tableau", "PowerBI", "Excel", 
    "Hugging Face", "OpenCV", "NLTK", "Spacy", "LangChain", "LlamaIndex",
    
    # API Standards & Communication Protocols
    "REST API", "GraphQL", "gRPC", "WebSocket", "Microservices", "SOAP", 
    "XML", "JSON", "YAML",
    
    # Developer Tools, Methodologies & Misc
    "Linux", "Git", "Jira", "Confluence", "Agile", "Scrum", "Docker Compose", 
    "OAuth", "JWT", "Prometheus", "Grafana"
]

# ==============================================================================
# EDUCATION EQUIVALENCE MAP (level 1-5)
# ==============================================================================
EDUCATION_EQUIVALENCE_MAP = {
    "phd": 5, 
    "ph.d": 5, 
    "doctorate": 5,
    "पीएचडी": 5,
    "masters": 4, 
    "m.tech": 4, 
    "mca": 4, 
    "mba": 4, 
    "m.s": 4, 
    "m.sc": 4, 
    "m.e": 4,
    "pgdm": 4,
    "m.com": 4,
    "m.a": 4,
    "एमटेक": 4,
    "एमसीए": 4,
    "bachelors": 3, 
    "b.tech": 3, 
    "b.e": 3, 
    "bca": 3, 
    "b.sc": 3, 
    "b.com": 3, 
    "b.a": 3,
    "bba": 3,
    "बीटेक": 3,
    "बीई": 3,
    "diploma": 2, 
    "polytechnic": 2,
    "डिप्लोमा": 2,
    "high school": 1, 
    "12th": 1, 
    "hsc": 1,
    "ssc": 1,
    "10th": 1,
    "cbse": 1,
    "icse": 1,
    "intermediate": 1,
    "matriculation": 1
}

# ==============================================================================
# LOCATION NORMALIZATION MAP
# ==============================================================================
LOCATION_NORMALIZATION_MAP = {
    "bengaluru": "bangalore", 
    "bangalore": "bangalore",
    "mumbai": "mumbai", 
    "bombay": "mumbai",
    "delhi": "delhi", 
    "new delhi": "delhi", 
    "ncr": "delhi",
    "gurgaon": "gurgaon", 
    "gurugram": "gurgaon",
    "noida": "noida", 
    "greater noida": "noida",
    "hyderabad": "hyderabad", 
    "secunderabad": "hyderabad",
    "chennai": "chennai", 
    "madras": "chennai",
    "pune": "pune",
    "kolkata": "kolkata", 
    "calcutta": "kolkata",
    "ahmedabad": "ahmedabad",
    "jaipur": "jaipur",
    "indore": "indore",
    "kochi": "kochi", 
    "cochin": "kochi",
    "thiruvananthapuram": "thiruvananthapuram", 
    "trivandrum": "thiruvananthapuram",
    "coimbatore": "coimbatore",
    "visakhapatnam": "visakhapatnam",
    "vizag": "visakhapatnam",
    "chandigarh": "chandigarh",
    "mohali": "chandigarh",
    "gandhinagar": "gandhinagar",
    "vadodara": "vadodara",
    "baroda": "vadodara",
    "nagpur": "nagpur",
    "lucknow": "lucknow",
    "bhubaneswar": "bhubaneswar",
    "ghaziabad": "ghaziabad",
    "faridabad": "faridabad"
}

# ==============================================================================
# COLLEGE TIER INDICATORS
# ==============================================================================
TIER_1_KEYWORDS = [
    "iit", "indian institute of technology", 
    "nit", "national institute of technology", 
    "bits", "birla institute"
]

TIER_2_KEYWORDS = [
    "vit", "srm", "manipal", "amity", 
    "lovely professional", "lpu", 
    "chandigarh university", "symbiosis"
]
