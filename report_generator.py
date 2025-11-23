#!/usr/bin/env python3
"""
Improved Report Generator - Two-Step Approach
1. Extract structured data from transcript (JSON)
2. Format structured data into final report
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Constants for null/empty value checking
NULL_VALUES = ['', 'null', 'None', 'Not mentioned', 'Not specified']

class ReportGenerator:
    """Generate reports using structured data extraction approach"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extracted_tickers = []
        self.validated_ticker_map = {}
        self.ticker_corrections = {}
    
    def set_extracted_tickers(self, tickers: List[str], validated_map: Dict[str, str], corrections: Dict[str, str]):
        """Set the extracted tickers and validation results"""
        self.extracted_tickers = tickers
        self.validated_ticker_map = validated_map
        self.ticker_corrections = corrections
    
    def extract_structured_data(self, transcript: str, video_title: str, channel_name: str, model) -> Dict[str, Any]:
        """
        Step 1: Extract structured data from transcript as JSON
        This is more reliable than generating formatted text directly
        """
        # Build ticker reference
        ticker_ref = self._build_ticker_reference()
        
        extraction_prompt = f"""You are a data extraction specialist. Extract structured trading information from this Turkish trading video transcript.

🚫 CRITICAL ANTI-HALLUCINATION RULES:
- Extract ONLY information explicitly mentioned in the transcript
- NEVER add tickers, prices, dates, or any information not in the transcript
- NEVER use external knowledge or current market data
- NEVER assume or infer information not directly stated
- If information is not in the transcript, use null or "Not mentioned in video"
- NEVER guess years, dates, or timeframes not explicitly mentioned
- NEVER add current date or time unless mentioned in video
- If a date is mentioned as only "16 Eylül" without year, extract exactly "16 Eylül" (NO year added)

VIDEO INFORMATION:
- Title: {video_title}
- Channel: {channel_name}

{ticker_ref}

TRANSCRIPT:
{transcript}

Extract ALL trading information into a structured JSON format. Respond ONLY with valid JSON, no additional text.

Required JSON structure:
{{
  "video_info": {{
    "title": "{video_title}",
    "channel": "{channel_name}",
    "date": "[extract date EXACTLY as mentioned in transcript, or null if not mentioned]"
  }},
  "summary": "[2-3 sentence summary in Turkish - ONLY based on transcript content]",
  "tickers": [
    {{
      "ticker": "TICKER_CODE",
      "company_name": "[from validated reference if available, else 'Unknown Company']",
      "timestamp": "[EXACT MM:SS or HH:MM:SS from transcript brackets when ticker is first mentioned]",
      "sentiment": "Bullish|Bearish|Neutral",
      "sentiment_reason": "[brief reason in Turkish - ONLY from transcript]",
      "resistance": "[price EXACTLY as mentioned in transcript, or null if not mentioned]",
      "support": "[price EXACTLY as mentioned in transcript, or null if not mentioned]",
      "target": "[price EXACTLY as mentioned in transcript, or null if not mentioned]",
      "notes": "[all relevant information in Turkish - ONLY from transcript]",
      "high_potential": true|false,
      "entry_price": "[price EXACTLY as mentioned in transcript, or null if not mentioned]",
      "stop_loss": "[price EXACTLY as mentioned in transcript, or null if not mentioned]",
      "risk": "[EXACTLY as mentioned in transcript, or null if not mentioned]",
      "risk_reward": "[EXACTLY as mentioned in transcript, or null if not mentioned]"
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. Include a ticker ONLY if it has technical analysis details (support, resistance, targets, sentiment, price levels, trading recommendations)
2. Do NOT include tickers that are only briefly mentioned without any trading analysis or context
3. For validated tickers (from reference), check transcript - if there's trading analysis, include it; if not, skip it
4. For unvalidated tickers, include them ONLY if they have clear trading analysis (prices, support/resistance, targets, sentiment)
5. DO NOT include obvious false positives like Turkish common words (BAKIN, BELKI, BIRAZ, BUNDA, DAHA, DOLAR, FIYAT, HATTA, KADAR, OLAN, ONDA, ONUN, ORADA, UZUN, YANI, YINE, ZAMAN, ZATEN, etc.)
6. Use ONLY information from transcript - NO assumptions, NO external knowledge, NO guessing
7. Extract timestamps EXACTLY from [MM:SS] or [HH:MM:SS] brackets in transcript when ticker is first mentioned
8. Use validated company names from ticker reference when available (never invent company names)
9. Mark high_potential=true ONLY for tickers with explicit BUY/SELL/HOLD recommendations AND technical analysis in transcript
10. For prices: Use EXACT values from transcript, or null if not mentioned - NEVER guess or estimate
11. For dates: Use EXACT format from transcript - if only day/month mentioned, do NOT add year
12. For summary: Base ONLY on transcript content - NO external interpretation
13. If any field is not mentioned in transcript, use null (not empty string, not placeholder text)
14. QUALITY OVER QUANTITY: Only include tickers with actionable trading information

VALIDATION CHECKLIST:
- Include validated tickers ONLY if they have technical analysis in transcript (support, resistance, targets, sentiment, price levels)
- Do NOT include tickers just because they're validated - they need trading context
- Consider including unvalidated tickers ONLY if they have clear trading analysis (prices, support/resistance, targets, sentiment)
- DO NOT include obvious Turkish common words (BAKIN, BELKI, BIRAZ, etc.)
- Every price must be traceable to exact transcript mention
- Every timestamp must match transcript brackets
- Every date must match transcript exactly (no additions)
- No information added that isn't in transcript
- Quality over quantity - only actionable trading information

Return ONLY the JSON object, no markdown formatting, no code blocks, no explanations."""

        try:
            response = model.generate_content(extraction_prompt)
            response_text = response.text.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            structured_data = json.loads(response_text)
            
            # CRITICAL VALIDATION: Verify all VALIDATED tickers are included
            # Only include tickers that have been validated (to avoid false positives like Turkish words)
            validated_ticker_set = set(self.validated_ticker_map.keys()) if self.validated_ticker_map else set()
            extracted_in_json = set(t.get('ticker', '') for t in structured_data.get('tickers', []))
            missing_validated_tickers = validated_ticker_set - extracted_in_json
            
            if missing_validated_tickers:
                # Log warning but don't fail - we'll add missing validated tickers below
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Extracted JSON missing {len(missing_validated_tickers)} validated tickers: {missing_validated_tickers}")
            
            # Add missing VALIDATED tickers with minimal data (only validated tickers to avoid false positives)
            for missing_ticker in missing_validated_tickers:
                structured_data['tickers'].append({
                    'ticker': missing_ticker,
                    'company_name': self.validated_ticker_map.get(missing_ticker, 'Unknown Company'),
                    'timestamp': 'Not mentioned',
                    'sentiment': 'Neutral',
                    'sentiment_reason': 'Ticker mentioned in transcript but specific details not extracted',
                    'resistance': None,
                    'support': None,
                    'target': None,
                    'notes': 'Ticker was mentioned in transcript but specific details were not found.',
                    'high_potential': False,
                    'entry_price': None,
                    'stop_loss': None,
                    'risk': None,
                    'risk_reward': None
                })
            
            return structured_data
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}\nResponse was: {response_text[:500]}")
        except Exception as e:
            raise Exception(f"Extraction failed: {e}")
    
    def format_report(self, structured_data: Dict[str, Any], video_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Step 2: Format structured data into final report
        This ensures consistent formatting and all tickers are included
        """
        report = []
        
        # Report Information
        report.append("## 📊 REPORT INFORMATION")
        report.append(f"- **Source**: {structured_data.get('video_info', {}).get('title', 'Unknown')} - {structured_data.get('video_info', {}).get('channel', 'Unknown')}")
        date = structured_data.get('video_info', {}).get('date')
        if date:
            report.append(f"- **Video Date**: {date}")
        report.append("")
        
        # Short Summary - Add video metadata here (only once)
        report.append("## 📝 SHORT SUMMARY")
        
        # Add video metadata to summary if available
        metadata_lines = []
        if video_metadata:
            if video_metadata.get('upload_date'):
                metadata_lines.append(f"**Video Upload Date**: {video_metadata.get('upload_date')}")
            if video_metadata.get('duration'):
                metadata_lines.append(f"**Video Duration**: {video_metadata.get('duration')}")
        
        summary = structured_data.get('summary')
        if summary and summary not in [None, 'null', 'None']:
            if metadata_lines:
                report.append("\n".join(metadata_lines))
                report.append("")
            report.append(summary)
        else:
            if metadata_lines:
                report.append("\n".join(metadata_lines))
                report.append("")
            report.append("*Summary not available from transcript*")
        report.append("")
        
        # Trading Opportunities
        report.append("## 📈 TRADING OPPORTUNITIES")
        report.append("")
        
        tickers = structured_data.get('tickers', [])
        if not tickers:
            report.append("*No tickers found in transcript*")
        else:
            # Consolidate duplicate tickers - merge all information for same ticker
            ticker_dict = {}
            for ticker_data in tickers:
                ticker = ticker_data.get('ticker', 'UNKNOWN').upper()
                
                if ticker not in ticker_dict:
                    # First occurrence - use as base
                    ticker_dict[ticker] = ticker_data.copy()
                else:
                    # Duplicate ticker - merge information
                    existing = ticker_dict[ticker]
                    
                    # Merge notes intelligently
                    existing_notes = existing.get('notes', '')
                    new_notes = ticker_data.get('notes', '')
                    existing['notes'] = self._merge_notes_intelligently(existing_notes, new_notes)
                    
                    # Use earliest timestamp
                    existing_timestamp = existing.get('timestamp', 'Not mentioned')
                    new_timestamp = ticker_data.get('timestamp', 'Not mentioned')
                    if not self._is_null_or_empty(new_timestamp):
                        existing_time = self._parse_timestamp(existing_timestamp)
                        new_time = self._parse_timestamp(new_timestamp)
                        if existing_time is None or (new_time is not None and new_time < existing_time):
                            existing['timestamp'] = new_timestamp
                    
                    # Merge sentiment if different (prefer more specific)
                    if ticker_data.get('sentiment') and ticker_data.get('sentiment') != 'Neutral':
                        existing['sentiment'] = ticker_data.get('sentiment', existing.get('sentiment'))
                        existing['sentiment_reason'] = ticker_data.get('sentiment_reason', existing.get('sentiment_reason'))
                    
                    # Merge prices (combine multiple values for support/resistance/target)
                    for field in ['resistance', 'support', 'target']:
                        new_value = ticker_data.get(field)
                        if not self._is_null_or_empty(new_value):
                            existing_value = existing.get(field)
                            if self._is_null_or_empty(existing_value):
                                existing[field] = new_value
                            else:
                                # Combine multiple price levels
                                if existing_value != new_value:
                                    existing[field] = f"{existing_value}, {new_value}"
                    
                    # For entry_price and stop_loss, use first non-null value
                    for field in ['entry_price', 'stop_loss']:
                        new_value = ticker_data.get(field)
                        if not self._is_null_or_empty(new_value):
                            existing_value = existing.get(field)
                            if self._is_null_or_empty(existing_value):
                                existing[field] = new_value
                    
                    # Merge high_potential (if any is True, set to True)
                    if ticker_data.get('high_potential', False):
                        existing['high_potential'] = True
            
            # Format consolidated tickers with validation
            for ticker_data in ticker_dict.values():
                ticker = ticker_data.get('ticker', 'UNKNOWN')
                company_name = ticker_data.get('company_name', 'Unknown Company')
                
                # Validate required fields
                if not ticker or ticker == 'UNKNOWN':
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Skipping ticker with invalid ticker code: {ticker_data}")
                    continue
                if not company_name or company_name == 'Unknown Company':
                    # Try to get from validated map
                    company_name = self.validated_ticker_map.get(ticker, 'Unknown Company')
                
                report.append(f"### {company_name} ({ticker})")
                report.append(f"- **Timestamp**: {ticker_data.get('timestamp', 'Not mentioned')}")
                report.append(f"- **Sentiment**: {ticker_data.get('sentiment', 'Neutral')} - {ticker_data.get('sentiment_reason', '')}")
                
                resistance = ticker_data.get('resistance')
                if not self._is_null_or_empty(resistance):
                    report.append(f"- **Resistance**: {resistance}")
                else:
                    report.append("- **Resistance**:")
                
                support = ticker_data.get('support')
                if not self._is_null_or_empty(support):
                    report.append(f"- **Support**: {support}")
                else:
                    report.append("- **Support**:")
                
                target = ticker_data.get('target')
                if not self._is_null_or_empty(target):
                    report.append(f"- **Target**: {target}")
                else:
                    report.append("- **Target**:")
                
                notes = ticker_data.get('notes', '')
                if not self._is_null_or_empty(notes):
                    report.append(f"- **Notes**: {notes}")
                else:
                    report.append("- **Notes**:")
                
                report.append("")
        
        # High Potential Trades - use consolidated tickers
        consolidated_tickers = list(ticker_dict.values()) if tickers else []
        high_potential = [t for t in consolidated_tickers if t.get('high_potential', False)]
        if high_potential:
            report.append("## 🎯 HIGH POTENTIAL TRADES")
            report.append("")
            
            for i, ticker_data in enumerate(high_potential, 1):
                ticker = ticker_data.get('ticker', 'UNKNOWN')
                company_name = ticker_data.get('company_name', 'Unknown Company')
                sentiment = ticker_data.get('sentiment', 'BUY').upper()
                
                entry = ticker_data.get('entry_price', '')
                stop = ticker_data.get('stop_loss', '')
                target = ticker_data.get('target', '')
                risk = ticker_data.get('risk', '')
                risk_reward = ticker_data.get('risk_reward', '')
                
                # Build action line
                action_parts = []
                if entry:
                    action_parts.append(f"Entry: **{entry}**")
                if stop:
                    action_parts.append(f"Stop: **{stop}**")
                if target:
                    action_parts.append(f"Target: **{target}**")
                if risk:
                    action_parts.append(f"Risk: **{risk}**")
                if risk_reward:
                    action_parts.append(f"Risk/Reward: **{risk_reward}**")
                
                action_str = " - ".join([f"[{p}]" for p in action_parts]) if action_parts else ""
                
                reason = ticker_data.get('sentiment_reason', ticker_data.get('notes', ''))
                
                report.append(f"**{i}.** **{company_name} ({ticker})**: {sentiment} {action_str}")
                if reason:
                    report.append(f"   *[Reason: {reason}]*")
                report.append("")
        
        return "\n".join(report)
    
    def _build_ticker_reference(self) -> str:
        """Build ticker reference string for prompt"""
        ref = []
        
        if self.validated_ticker_map:
            ref.append("**VALIDATED TICKER REFERENCE:**")
            for ticker, name in sorted(self.validated_ticker_map.items()):
                ref.append(f"- {ticker} = {name}")
            ref.append("")
        
        if self.extracted_tickers:
            ref.append("**ALL TICKERS EXTRACTED FROM TRANSCRIPT (MUST INCLUDE VALIDATED ONES):**")
            # Show all extracted tickers, marking which are validated
            all_extracted = sorted(set(self.extracted_tickers))
            for ticker in all_extracted:
                if ticker in self.validated_ticker_map:
                    ref.append(f"- {ticker} (✓ validated: {self.validated_ticker_map[ticker]})")
                else:
                    # Check if it's a common Turkish word (false positive)
                    common_turkish_words = {
                        'BAKIN', 'BELKI', 'BIRAZ', 'BUNDA', 'DAHA', 'DOLAR', 'FIYAT', 'HATTA', 
                        'KADAR', 'OLAN', 'ONDA', 'ONUN', 'ORADA', 'UZUN', 'YANI', 'YINE', 
                        'ZAMAN', 'ZATEN', 'VAR', 'BIR', 'BU', 'VE', 'YA', 'DA', 'DE', 'KI', 'ILK', 'SON'
                    }
                    if ticker not in common_turkish_words:
                        ref.append(f"- {ticker} (not validated - check transcript for company name and context)")
            ref.append("")
        
        if self.ticker_corrections:
            ref.append("**TICKER CORRECTIONS (AUTO-CORRECTED):**")
            for incorrect, correct in sorted(self.ticker_corrections.items()):
                ref.append(f"- {incorrect} → {correct}")
            ref.append("")
        
        return "\n".join(ref)
    
    def _is_null_or_empty(self, value: Any) -> bool:
        """Helper function to check if value is null or empty"""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() in NULL_VALUES
        return False
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[int]:
        """Robust timestamp parsing with validation"""
        if not timestamp_str or timestamp_str in NULL_VALUES:
            return None
        
        try:
            parts = timestamp_str.split(':')
            if len(parts) == 2:
                # MM:SS format
                minutes, seconds = int(parts[0]), int(parts[1])
                if 0 <= minutes < 60 and 0 <= seconds < 60:
                    return minutes * 60 + seconds
            elif len(parts) == 3:
                # HH:MM:SS format
                hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
                if 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60:
                    return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            pass
        return None
    
    def _merge_notes_intelligently(self, existing_notes: str, new_notes: str) -> str:
        """Intelligently merge notes avoiding duplicates"""
        if self._is_null_or_empty(new_notes):
            return existing_notes
        if self._is_null_or_empty(existing_notes):
            return new_notes
        
        # Check if new notes are substantially different (not just a substring)
        existing_lower = existing_notes.lower()
        new_lower = new_notes.lower()
        
        # If new notes are contained in existing, don't add
        if new_lower in existing_lower:
            return existing_notes
        
        # If existing notes are contained in new, replace
        if existing_lower in new_lower:
            return new_notes
        
        # Check for significant overlap (more than 50% similarity)
        words_existing = set(existing_lower.split())
        words_new = set(new_lower.split())
        if len(words_existing) > 0 and len(words_new) > 0:
            overlap = len(words_existing & words_new) / max(len(words_existing), len(words_new))
            if overlap > 0.5:
                # High overlap, prefer the longer/more detailed version
                return new_notes if len(new_notes) > len(existing_notes) else existing_notes
        
        # Different content, combine them
        return f"{existing_notes}. {new_notes}"

