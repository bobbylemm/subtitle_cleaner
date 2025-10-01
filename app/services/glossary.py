import re
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.domain.constants import Language
from app.api.schemas.glossaries import GlossaryTerm


class GlossaryStore:
    """In-memory glossary store (can be replaced with DB implementation)"""
    
    def __init__(self):
        self.glossaries: Dict[str, Dict[str, Any]] = {}
    
    def create_glossary(
        self,
        name: str,
        language: Language,
        terms: List[GlossaryTerm],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new glossary"""
        glossary_id = self._generate_id(name, language)
        
        self.glossaries[glossary_id] = {
            "id": glossary_id,
            "name": name,
            "description": description,
            "language": language,
            "terms": terms,
            "tags": tags or [],
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        return glossary_id
    
    def get_glossary(self, glossary_id: str) -> Optional[Dict[str, Any]]:
        """Get a glossary by ID"""
        return self.glossaries.get(glossary_id)
    
    def update_glossary(
        self,
        glossary_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        terms: Optional[List[GlossaryTerm]] = None,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """Update an existing glossary"""
        if glossary_id not in self.glossaries:
            return False
        
        glossary = self.glossaries[glossary_id]
        
        if name is not None:
            glossary["name"] = name
        if description is not None:
            glossary["description"] = description
        if terms is not None:
            glossary["terms"] = terms
        if tags is not None:
            glossary["tags"] = tags
        if is_active is not None:
            glossary["is_active"] = is_active
        
        glossary["updated_at"] = datetime.utcnow().isoformat()
        
        return True
    
    def delete_glossary(self, glossary_id: str) -> bool:
        """Delete a glossary"""
        if glossary_id in self.glossaries:
            del self.glossaries[glossary_id]
            return True
        return False
    
    def list_glossaries(
        self,
        language: Optional[Language] = None,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """List glossaries with filtering"""
        filtered = []
        
        for glossary in self.glossaries.values():
            # Apply filters
            if language and glossary["language"] != language:
                continue
            if tags and not any(tag in glossary["tags"] for tag in tags):
                continue
            if is_active is not None and glossary["is_active"] != is_active:
                continue
            
            filtered.append(glossary)
        
        # Sort by updated_at
        filtered.sort(key=lambda x: x["updated_at"], reverse=True)
        
        # Paginate
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        
        return filtered[start:end], total
    
    def search_glossaries(self, query: str) -> List[Dict[str, Any]]:
        """Search glossaries by name or description"""
        query_lower = query.lower()
        results = []
        
        for glossary in self.glossaries.values():
            if (
                query_lower in glossary["name"].lower() or
                (glossary["description"] and query_lower in glossary["description"].lower())
            ):
                results.append(glossary)
        
        return results
    
    def _generate_id(self, name: str, language: Language) -> str:
        """Generate a unique ID for a glossary"""
        content = f"{name}_{language}_{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class GlossaryEnforcer:
    """Apply glossary terms to text"""
    
    def __init__(self, glossary_store: GlossaryStore):
        self.store = glossary_store
    
    def apply_glossary(
        self,
        text: str,
        glossary_id: str,
        preview: bool = False
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Apply glossary terms to text"""
        glossary = self.store.get_glossary(glossary_id)
        if not glossary or not glossary["is_active"]:
            return text, []
        
        changes = []
        modified_text = text
        
        for term in glossary["terms"]:
            modified_text, term_changes = self._apply_term(
                modified_text,
                term,
                preview
            )
            changes.extend(term_changes)
        
        return modified_text, changes
    
    def _apply_term(
        self,
        text: str,
        term: GlossaryTerm,
        preview: bool
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Apply a single glossary term"""
        changes = []
        
        # Build regex pattern
        pattern = self._build_pattern(term)
        
        if not pattern:
            return text, changes
        
        # Find all matches
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return text, changes
        
        # Apply replacements (backwards to maintain positions)
        if not preview:
            offset = 0
            for match in matches:
                start = match.start() + offset
                end = match.end() + offset
                original = text[start:end]
                
                # Preserve case if needed
                replacement = self._preserve_case(original, term.replacement)
                
                text = text[:start] + replacement + text[end:]
                offset += len(replacement) - len(original)
                
                changes.append({
                    "type": "glossary_replacement",
                    "original": original,
                    "replacement": replacement,
                    "position": start,
                })
        else:
            # Just record what would change
            for match in matches:
                changes.append({
                    "type": "glossary_replacement",
                    "original": match.group(),
                    "replacement": term.replacement,
                    "position": match.start(),
                })
        
        return text, changes
    
    def _build_pattern(self, term: GlossaryTerm) -> Optional[re.Pattern]:
        """Build regex pattern for term matching"""
        try:
            # If context regex is provided, use it
            if term.context_regex:
                return re.compile(term.context_regex, 
                                flags=0 if term.case_sensitive else re.IGNORECASE)
            
            # Build pattern from original term
            pattern = re.escape(term.original)
            
            # Add word boundaries if whole word matching
            if term.whole_word:
                pattern = r'\b' + pattern + r'\b'
            
            flags = 0 if term.case_sensitive else re.IGNORECASE
            
            return re.compile(pattern, flags=flags)
        
        except re.error:
            return None
    
    def _preserve_case(self, original: str, replacement: str) -> str:
        """Preserve case of original in replacement"""
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        else:
            return replacement


# Global instances
glossary_store = GlossaryStore()
glossary_enforcer = GlossaryEnforcer(glossary_store)