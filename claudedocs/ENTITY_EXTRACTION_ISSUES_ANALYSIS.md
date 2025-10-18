# Entity Extraction Issues Analysis

## Investigation Results

### 1. Missing Entities Problem

#### Issue: "AC Milan" Not Extracted
**Root Cause**: The compound_name pattern expects `[A-Z][a-z]+` format (capital followed by lowercase).
- "AC" is all uppercase, doesn't match the pattern
- Pattern: `r'\b[A-Z][a-z]+(?:\s+...)`
- "AC Milan" has "AC" which fails the `[A-Z][a-z]+` requirement

**Solution**: Add pattern for acronym + name combinations

#### Issue: "Uruguay" Not Extracted  
**Root Cause**: Single-word country names are filtered out
- The extraction focuses on compound names
- Single words need minimum 4 characters AND pass validation
- Countries like "Uruguay", "Spain", "Italy" are being treated as common words

**Solution**: Add specific pattern for known country/location names

### 2. Duplicate Corrections Problem

#### Issue: "Romano Romano", "Hotspur Hotspur"
**Root Cause**: SmartEntityMatcher creates overlapping corrections
```
Corrections created:
- "Fabrizio" → "Fabrizio Romano"
- "Fabrizio Romano" → "Fabrizio Romano"
- "Tottenham" → "Tottenham Hotspur"  
- "Tottenham Hotspur" → "Tottenham Hotspur"
```

When applied sequentially:
1. "Fabrizio Romano" stays as is (already correct)
2. "Romano" alone gets replaced with "Fabrizio Romano" 
3. Result: "Fabrizio Romano Romano"

**Solution**: Check for partial matches already being part of complete entities

### 3. Pattern Detection Results

Testing showed:
- ✅ Extracted: Fabrizio Romano, Manchester United, Rodrigo Bentancur
- ❌ Missed: AC Milan (acronym issue)
- ❌ Missed: Uruguay, Spain, Italy (single words)
- ❌ Missed: Sporting CP (CP is acronym)

## Recommended Fixes

### Fix 1: Enhanced Entity Patterns in context_extraction_improved.py

```python
# Add to entity_patterns dict:
'acronym_plus_name': re.compile(
    r'\b[A-Z]{2,}\s+[A-Z][a-z]+\b'  # AC Milan, FC Barcelona, LA Galaxy
),

'country_names': re.compile(
    r'\b(?:Uruguay|Spain|Italy|France|England|Germany|Brazil|Argentina|Portugal|Netherlands|Belgium|Croatia|Nigeria|Ghana|Cameroon|Egypt|Morocco|Japan|Korea|China|India|USA|Mexico|Colombia|Chile|Peru|Ecuador)\b',
    re.IGNORECASE
),
```

### Fix 2: Prevent Duplicate Application in SmartEntityMatcher

```python
def find_smart_corrections(self, document_text, context_entities):
    corrections = {}
    
    # ... existing code ...
    
    # After building corrections, filter out overlapping ones
    filtered_corrections = {}
    
    # Sort by length (longest first) to prioritize complete matches
    sorted_items = sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, replacement in sorted_items:
        # Check if this is a substring of an already added correction
        is_substring = False
        for existing_orig in filtered_corrections:
            if original in existing_orig or existing_orig in original:
                # Skip if one is substring of another
                if len(existing_orig) > len(original):
                    is_substring = True
                    break
        
        if not is_substring:
            filtered_corrections[original] = replacement
    
    return filtered_corrections
```

### Fix 3: Improve Context Extraction Logic

```python
def _extract_entities_from_text(self, text, source_id, authority_score):
    entities = []
    seen = set()
    
    # Priority 0: Extract acronym + name combinations (NEW)
    acronym_name_pattern = re.compile(r'\b[A-Z]{2,}\s+[A-Z][a-z]+\b')
    for match in acronym_name_pattern.finditer(text):
        entity_text = match.group(0)
        if entity_text not in seen:
            seen.add(entity_text)
            entities.append(ExtractedEntity(
                text=entity_text,
                canonical_form=entity_text,
                entity_type=EntityType.ORGANIZATION,
                confidence=0.85 * authority_score,
                source_id=source_id,
                context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                position=match.start()
            ))
    
    # ... rest of existing extraction ...
```

## Testing Results

### Before Fixes:
- Extracted: 9 entities
- Missed: AC Milan, Uruguay, Sporting CP
- Duplicates: Romano Romano, Hotspur Hotspur

### Expected After Fixes:
- Extract: 12+ entities
- Include: AC Milan, Uruguay, all countries
- No duplicates in corrections

## Implementation Priority

1. **High**: Fix duplicate corrections (causes visible errors)
2. **High**: Add acronym+name pattern (common in sports)
3. **Medium**: Add country names pattern
4. **Low**: Optimize validation logic

## Files to Modify

1. `/app/services/context_extraction_improved.py`
   - Add new patterns
   - Update validation logic

2. `/app/services/smart_entity_matcher.py`
   - Add de-duplication logic
   - Improve matching algorithm

3. `/app/services/enhanced_cleaner.py`
   - Update correction application logic