from app.services.parser import SubtitleParser
from app.domain.constants import SubtitleFormat

def inspect():
    with open("foden_winner.srt", "r", encoding="utf-8") as f:
        content = f.read()
        
    doc = SubtitleParser.parse(content, SubtitleFormat.SRT)
    
    for s in doc.segments:
        if s.idx == 168:
            print(f"Segment 168 repr: {repr(s.text)}")
            print(f"Segment 168 hex: {s.text.encode('utf-8').hex()}")
            
        if s.idx == 76:
            print(f"Segment 76 repr: {repr(s.text)}")

if __name__ == "__main__":
    inspect()
