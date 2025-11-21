import sqlite3
import logging
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CorrectionEntry(BaseModel):
    wrong_term: str
    correct_term: str
    topic: str = "general"
    industry: str = "general"
    country: str = "general"
    tags: List[str] = []
    confidence: float = 1.0
    count: int = 1

class KnowledgeBaseService:
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with advanced schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # We'll drop the old table for this upgrade since it's dev
            # In prod, we'd migrate
            try:
                # Check if old schema exists (simple check: if 'domain' column exists)
                cursor.execute("SELECT domain FROM corrections LIMIT 1")
                # If successful and no exception, it's the old schema. Drop it.
                cursor.execute("DROP TABLE corrections")
            except sqlite3.OperationalError:
                # Table doesn't exist or column missing, proceed
                pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wrong_term TEXT NOT NULL,
                    correct_term TEXT NOT NULL,
                    topic TEXT DEFAULT 'general',
                    industry TEXT DEFAULT 'general',
                    country TEXT DEFAULT 'general',
                    tags TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 1.0,
                    count INTEGER DEFAULT 1,
                    UNIQUE(wrong_term, correct_term, topic, industry, country)
                )
            """)
            conn.commit()

    def get_best_correction(self, term: str, context: Dict[str, str]) -> Optional[str]:
        """
        Retrieve the best correction based on weighted context matching.
        Context keys: topic, industry, country
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT correct_term, topic, industry, country, count FROM corrections WHERE wrong_term = ?", (term,))
            rows = cursor.fetchall()
            
            if not rows:
                return None

            best_term = None
            best_score = -1

            # Weighted Scoring
            # Country match: High importance (local slang/entities)
            # Topic match: Medium importance
            # Industry match: Medium importance
            # Count: Tie-breaker
            
            target_topic = context.get("topic", "general").lower()
            target_industry = context.get("industry", "general").lower()
            target_country = context.get("country", "general").lower()

            for row in rows:
                correct_term, r_topic, r_industry, r_country, r_count = row
                score = 0
                
                if r_country.lower() == target_country and r_country.lower() != "general":
                    score += 10
                if r_topic.lower() == target_topic and r_topic.lower() != "general":
                    score += 5
                if r_industry.lower() == target_industry and r_industry.lower() != "general":
                    score += 5
                
                # Normalize count to 0-1 range contribution (log scale would be better but simple for now)
                score += min(r_count, 10) * 0.1

                if score > best_score:
                    best_score = score
                    best_term = correct_term
            
            return best_term

    def add_correction(self, wrong: str, correct: str, context: Dict[str, Any]):
        """Add or update a correction with full context."""
        topic = context.get("topic", "general")
        industry = context.get("industry", "general")
        country = context.get("country", "general")
        tags = json.dumps(context.get("tags", []))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Upsert logic
            try:
                cursor.execute("""
                    INSERT INTO corrections (wrong_term, correct_term, topic, industry, country, tags, count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(wrong_term, correct_term, topic, industry, country) 
                    DO UPDATE SET count = count + 1
                """, (wrong, correct, topic, industry, country, tags))
                conn.commit()
                logger.info(f"Learned: {wrong} -> {correct} (Ctx: {topic}/{country})")
            except Exception as e:
                logger.error(f"Failed to add correction: {e}")

    def get_all_corrections(self, context: Optional[Dict[str, str]] = None) -> List[CorrectionEntry]:
        """Get all corrections, optionally filtered/scored by context."""
        # For global consistency pass, we might want everything or just relevant ones.
        # For now, return all to be safe, but we could filter.
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM corrections")
            rows = cursor.fetchall()
            
            entries = []
            for r in rows:
                # id, wrong, correct, topic, industry, country, tags, conf, count
                entries.append(CorrectionEntry(
                    wrong_term=r[1],
                    correct_term=r[2],
                    topic=r[3],
                    industry=r[4],
                    country=r[5],
                    tags=json.loads(r[6]),
                    confidence=r[7],
                    count=r[8]
                ))
            return entries
