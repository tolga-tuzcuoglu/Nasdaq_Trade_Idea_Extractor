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
1. Include EVERY ticker mentioned in transcript - check the ALL TICKERS EXTRACTED list above - NO EXCEPTIONS
2. Use ONLY information from transcript - NO assumptions, NO external knowledge, NO guessing
3. Extract timestamps EXACTLY from [MM:SS] or [HH:MM:SS] brackets in transcript when ticker is first mentioned
4. Use validated company names from ticker reference when available (never invent company names)
5. Mark high_potential=true ONLY for tickers with explicit BUY/SELL/HOLD recommendations in transcript
6. For prices: Use EXACT values from transcript, or null if not mentioned - NEVER guess or estimate
7. For dates: Use EXACT format from transcript - if only day/month mentioned, do NOT add year
8. For summary: Base ONLY on transcript content - NO external interpretation
9. If any field is not mentioned in transcript, use null (not empty string, not placeholder text)

VALIDATION CHECKLIST:
- Every ticker in ALL TICKERS EXTRACTED list must appear in the JSON
- Every price must be traceable to exact transcript mention
- Every timestamp must match transcript brackets
- Every date must match transcript exactly (no additions)
- No information added that isn't in transcript

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
            
            # CRITICAL VALIDATION: Verify all extracted tickers are included
            extracted_ticker_set = set(self.extracted_tickers) if self.extracted_tickers else set()
            extracted_in_json = set(t.get('ticker', '') for t in structured_data.get('tickers', []))
            missing_tickers = extracted_ticker_set - extracted_in_json
            
            if missing_tickers:
                # Log warning but don't fail - we'll add missing tickers below
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Extracted JSON missing {len(missing_tickers)} tickers: {missing_tickers}")
            
            # Add missing tickers with minimal data (to ensure ALL tickers are included)
            for missing_ticker in missing_tickers:
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
    
    def format_report(self, structured_data: Dict[str, Any]) -> str:
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
        
        # Short Summary
        report.append("## 📝 SHORT SUMMARY")
        summary = structured_data.get('summary')
        if summary and summary not in [None, 'null', 'None']:
            report.append(summary)
        else:
            report.append("*Summary not available from transcript*")
        report.append("")
        
        # Trading Opportunities
        report.append("## 📈 TRADING OPPORTUNITIES")
        report.append("")
        
        tickers = structured_data.get('tickers', [])
        if not tickers:
            report.append("*No tickers found in transcript*")
        else:
            for ticker_data in tickers:
                ticker = ticker_data.get('ticker', 'UNKNOWN')
                company_name = ticker_data.get('company_name', 'Unknown Company')
                
                report.append(f"### {company_name} ({ticker})")
                report.append(f"- **Timestamp**: {ticker_data.get('timestamp', 'Not mentioned')}")
                report.append(f"- **Sentiment**: {ticker_data.get('sentiment', 'Neutral')} - {ticker_data.get('sentiment_reason', '')}")
                
                resistance = ticker_data.get('resistance')
                if resistance and resistance not in [None, 'null', 'None', '']:
                    report.append(f"- **Resistance**: {resistance}")
                else:
                    report.append("- **Resistance**:")
                
                support = ticker_data.get('support')
                if support and support not in [None, 'null', 'None', '']:
                    report.append(f"- **Support**: {support}")
                else:
                    report.append("- **Support**:")
                
                target = ticker_data.get('target')
                if target and target not in [None, 'null', 'None', '']:
                    report.append(f"- **Target**: {target}")
                else:
                    report.append("- **Target**:")
                
                notes = ticker_data.get('notes', '')
                if notes and notes not in [None, 'null', 'None', '']:
                    report.append(f"- **Notes**: {notes}")
                else:
                    report.append("- **Notes**:")
                
                report.append("")
        
        # High Potential Trades
        high_potential = [t for t in tickers if t.get('high_potential', False)]
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
            ref.append("**ALL TICKERS EXTRACTED FROM TRANSCRIPT (MUST INCLUDE ALL):**")
            for ticker in sorted(set(self.extracted_tickers)):
                if ticker in self.validated_ticker_map:
                    ref.append(f"- {ticker} (validated: {self.validated_ticker_map[ticker]})")
                else:
                    ref.append(f"- {ticker} (not validated - check transcript for company name)")
            ref.append("")
        
        if self.ticker_corrections:
            ref.append("**TICKER CORRECTIONS (AUTO-CORRECTED):**")
            for incorrect, correct in sorted(self.ticker_corrections.items()):
                ref.append(f"- {incorrect} → {correct}")
            ref.append("")
        
        return "\n".join(ref)

