import re
from typing import List, Tuple, Optional
from app.domain.models import Segment, SubtitleDocument
from app.domain.constants import (
    SubtitleFormat,
    TIMECODE_PATTERN_SRT,
    TIMECODE_PATTERN_WEBVTT,
)


class SubtitleParser:
    """Parse SRT and WebVTT subtitle formats"""
    
    @staticmethod
    def parse(content: str, format: SubtitleFormat) -> SubtitleDocument:
        """Parse subtitle content into document"""
        if format == SubtitleFormat.SRT:
            return SubtitleParser._parse_srt(content)
        elif format == SubtitleFormat.WEBVTT:
            return SubtitleParser._parse_webvtt(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _parse_srt(content: str) -> SubtitleDocument:
        """Parse SRT format"""
        segments = []
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # Parse index (first line)
            try:
                idx = int(lines[0].strip())
            except ValueError:
                continue
            
            # Parse timecodes (second line)
            timecode_line = lines[1].strip()
            times = SubtitleParser._parse_srt_timecode(timecode_line)
            if not times:
                continue
            
            start_ms, end_ms = times
            
            # Parse text (remaining lines)
            text = '\n'.join(lines[2:]).strip()
            if not text:
                continue
            
            segments.append(Segment(
                idx=idx,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text
            ))
        
        return SubtitleDocument(segments=segments)
    
    @staticmethod
    def _parse_webvtt(content: str) -> SubtitleDocument:
        """Parse WebVTT format"""
        segments = []
        
        # Remove WebVTT header
        content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
        
        # Split into blocks
        blocks = content.strip().split('\n\n')
        idx = 1
        
        for block in blocks:
            lines = block.strip().split('\n')
            if not lines:
                continue
            
            # Find timecode line
            timecode_line = None
            text_start = 0
            
            for i, line in enumerate(lines):
                if '-->' in line:
                    timecode_line = line
                    text_start = i + 1
                    break
            
            if not timecode_line:
                continue
            
            # Parse timecodes
            times = SubtitleParser._parse_webvtt_timecode(timecode_line)
            if not times:
                continue
            
            start_ms, end_ms = times
            
            # Parse text
            if text_start < len(lines):
                text = '\n'.join(lines[text_start:]).strip()
                # Remove WebVTT tags
                text = re.sub(r'<v[^>]*>', '', text)
                text = re.sub(r'</v>', '', text)
                text = re.sub(r'<c[^>]*>', '', text)
                text = re.sub(r'</c>', '', text)
                
                if text:
                    segments.append(Segment(
                        idx=idx,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=text
                    ))
                    idx += 1
        
        return SubtitleDocument(segments=segments)
    
    @staticmethod
    def _parse_srt_timecode(line: str) -> Optional[Tuple[int, int]]:
        """Parse SRT timecode line (00:00:00,000 --> 00:00:00,000)"""
        match = re.match(
            r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
            line
        )
        if not match:
            return None
        
        start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
        end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
        
        start_total_ms = (start_h * 3600 + start_m * 60 + start_s) * 1000 + start_ms
        end_total_ms = (end_h * 3600 + end_m * 60 + end_s) * 1000 + end_ms
        
        return start_total_ms, end_total_ms
    
    @staticmethod
    def _parse_webvtt_timecode(line: str) -> Optional[Tuple[int, int]]:
        """Parse WebVTT timecode line (00:00:00.000 --> 00:00:00.000)"""
        # Clean up any position/align settings
        line = line.split(' position')[0].split(' align')[0].strip()
        
        match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})',
            line
        )
        if not match:
            # Try without hours
            match = re.match(
                r'(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2})\.(\d{3})',
                line
            )
            if match:
                start_m, start_s, start_ms = map(int, match.groups()[:3])
                end_m, end_s, end_ms = map(int, match.groups()[3:])
                start_total_ms = (start_m * 60 + start_s) * 1000 + start_ms
                end_total_ms = (end_m * 60 + end_s) * 1000 + end_ms
                return start_total_ms, end_total_ms
            return None
        
        start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
        end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
        
        start_total_ms = (start_h * 3600 + start_m * 60 + start_s) * 1000 + start_ms
        end_total_ms = (end_h * 3600 + end_m * 60 + end_s) * 1000 + end_ms
        
        return start_total_ms, end_total_ms


class SubtitleSerializer:
    """Serialize subtitle documents to SRT/WebVTT formats"""
    
    @staticmethod
    def serialize(document: SubtitleDocument, format: SubtitleFormat) -> str:
        """Serialize document to subtitle format"""
        if format == SubtitleFormat.SRT:
            return SubtitleSerializer._serialize_srt(document)
        elif format == SubtitleFormat.WEBVTT:
            return SubtitleSerializer._serialize_webvtt(document)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _serialize_srt(document: SubtitleDocument) -> str:
        """Serialize to SRT format"""
        blocks = []
        
        for i, segment in enumerate(document.segments, 1):
            start_time = SubtitleSerializer._ms_to_srt_time(segment.start_ms)
            end_time = SubtitleSerializer._ms_to_srt_time(segment.end_ms)
            
            block = f"{i}\n{start_time} --> {end_time}\n{segment.text}"
            blocks.append(block)
        
        return '\n\n'.join(blocks) + '\n'
    
    @staticmethod
    def _serialize_webvtt(document: SubtitleDocument) -> str:
        """Serialize to WebVTT format"""
        lines = ["WEBVTT", ""]
        
        for segment in document.segments:
            start_time = SubtitleSerializer._ms_to_webvtt_time(segment.start_ms)
            end_time = SubtitleSerializer._ms_to_webvtt_time(segment.end_ms)
            
            lines.append(f"{start_time} --> {end_time}")
            lines.append(segment.text)
            lines.append("")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        """Convert milliseconds to SRT time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    @staticmethod
    def _ms_to_webvtt_time(ms: int) -> str:
        """Convert milliseconds to WebVTT time format"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"