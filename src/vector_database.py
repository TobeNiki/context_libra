import logging
import psycopg2
import json
import os 
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))