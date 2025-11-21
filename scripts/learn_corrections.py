import asyncio
import sys
import os
from pathlib import Path
import difflib

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.knowledge_base import KnowledgeBaseService
from app.services.parser import SubtitleParser
from app.domain.constants import SubtitleFormat

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Learn corrections from diff.")
    parser.add_argument("original", help="Original subtitle file")
    parser.add_argument("corrected", help="Corrected subtitle file")
    parser.add_argument("--topic", default="general", help="Topic (e.g., Football)")
    parser.add_argument("--industry", default="general", help="Industry (e.g., Sports)")
    parser.add_argument("--country", default="general", help="Country (e.g., Germany)")
    
    args = parser.parse_args()

    print(f"🧠 Learning from {args.original} -> {args.corrected}")
    print(f"   Context: Topic={args.topic}, Industry={args.industry}, Country={args.country}")

    if not os.path.exists(args.original) or not os.path.exists(args.corrected):
        print("❌ Error: Files not found.")
        sys.exit(1)

    with open(args.original, "r", encoding="utf-8") as f:
        orig_content = f.read()
    with open(args.corrected, "r", encoding="utf-8") as f:
        corr_content = f.read()

    # Parse both
    orig_doc = SubtitleParser.parse(orig_content, SubtitleFormat.SRT)
    corr_doc = SubtitleParser.parse(corr_content, SubtitleFormat.SRT)

    if len(orig_doc.segments) != len(corr_doc.segments):
        print("❌ Error: Segment count mismatch. Cannot align files.")
        sys.exit(1)

    kb = KnowledgeBaseService()
    learned_count = 0
    
    context = {
        "topic": args.topic,
        "industry": args.industry,
        "country": args.country
    }

    for orig, corr in zip(orig_doc.segments, corr_doc.segments):
        if orig.text != corr.text:
            # Simple diff to find changed words
            s = difflib.SequenceMatcher(None, orig.text.split(), corr.text.split())
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag == 'replace':
                    wrong_phrase = " ".join(orig.text.split()[i1:i2])
                    correct_phrase = " ".join(corr.text.split()[j1:j2])
                    
                    # Filter out huge replacements
                    if len(wrong_phrase.split()) < 5:
                        kb.add_correction(wrong_phrase, correct_phrase, context)
                        print(f"  ✅ Learned: '{wrong_phrase}' -> '{correct_phrase}'")
                        learned_count += 1

    print(f"🎉 Done! Learned {learned_count} new corrections.")

if __name__ == "__main__":
    main()
