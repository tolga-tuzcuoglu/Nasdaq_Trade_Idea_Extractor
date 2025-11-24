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

# Required assets that must be included if mentioned in transcript
REQUIRED_ASSETS = {
    # Indices
    'US500', 'US100', 'RTY', 'VIX',
    # Crypto
    'BTCUSD', 'ETHUSD', 'SOLUSD',
    # Stocks - Tech
    'MSTR', 'COIN', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOG', 'TSLA', 'AMD', 'PLTR', 'CRWD', 'AVGO',
    # Stocks - Other
    'HOOD', 'HIMS', 'SOFI', 'APP', 'RKLB', 'EOSE', 'CEG',
    # Stocks - Trading Analysis List
    'IBKR', 'AXON', 'UNH', 'LLY', 'ANET', 'ALAB', 'CRM', 'SE', 'GRAB', 'CLS', 'CELH', 'ZETA', 'NBIS', 
    'CRDO', 'OSCR', 'TEM', 'LMND', 'MRVL', 'MU', 'IREN', 'DLO', 'CRWV'
}

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
        # Apply ticker corrections to transcript before extraction
        # This ensures ASTR becomes ALAB, etc.
        corrected_transcript = self._apply_ticker_corrections_to_transcript(transcript)
        
        # Build ticker reference
        ticker_ref = self._build_ticker_reference()
        
        # Build required assets list for prompt
        required_assets_str = ", ".join(sorted(REQUIRED_ASSETS))
        
        extraction_prompt = f"""You are a data extraction specialist. Extract structured trading information from this Turkish trading video transcript.

🎯 REQUIRED ASSETS LIST:
The following assets should be prioritized if mentioned in the transcript WITH trading analysis:
{required_assets_str}

These include indices (US500, US100, RTY, VIX), cryptocurrencies (BTCUSD, ETHUSD, SOLUSD), and specific stocks.
Only include these if they have trading analysis details (support, resistance, targets, sentiment, price levels).

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
{corrected_transcript}

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
1. REQUIRED ASSETS: If any asset from the REQUIRED ASSETS LIST above is mentioned in transcript WITH trading analysis, it MUST be included
2. For ALL tickers (including required assets): Include ONLY if it has technical analysis details (support, resistance, targets, sentiment, price levels, trading recommendations)
3. Do NOT include tickers that are only briefly mentioned without any trading analysis or context - this applies to ALL tickers including required assets
4. For validated tickers (from reference), check transcript - if there's trading analysis, include it; if not, skip it (even if it's a required asset)
5. For unvalidated tickers, include them ONLY if they have clear trading analysis (prices, support/resistance, targets, sentiment)
6. DO NOT include obvious false positives like Turkish common words (BAKIN, BELKI, BIRAZ, BUNDA, DAHA, DOLAR, FIYAT, HATTA, KADAR, OLAN, ONDA, ONUN, ORADA, UZUN, YANI, YINE, ZAMAN, ZATEN, etc.)
7. Use ONLY information from transcript - NO assumptions, NO external knowledge, NO guessing
8. Extract timestamps EXACTLY from [MM:SS] or [HH:MM:SS] brackets in transcript when ticker is first mentioned
9. Use validated company names from ticker reference when available (never invent company names)
10. Mark high_potential=true ONLY for tickers with explicit BUY/SELL/HOLD recommendations AND technical analysis in transcript
11. For prices: Use EXACT values from transcript, or null if not mentioned - NEVER guess or estimate
12. For dates: Use EXACT format from transcript - if only day/month mentioned, do NOT add year
13. For summary: Base ONLY on transcript content - NO external interpretation
14. If any field is not mentioned in transcript, use null (not empty string, not placeholder text)
15. TICKER CORRECTIONS: ASTR should be extracted as ALAB, CRIDO as CRDO, etc. (corrections are applied automatically)

VALIDATION CHECKLIST:
- REQUIRED ASSETS: Include ONLY if mentioned WITH trading analysis (support, resistance, targets, sentiment, price levels)
- All validated tickers: Include ONLY if they have technical analysis in transcript (support, resistance, targets, sentiment, price levels)
- Unvalidated tickers: Include ONLY if they have clear trading analysis (prices, support/resistance, targets, sentiment)
- DO NOT include obvious Turkish common words (BAKIN, BELKI, BIRAZ, etc.)
- Every price must be traceable to exact transcript mention
- Every timestamp must match transcript brackets
- Every date must match transcript exactly (no additions)
- No information added that isn't in transcript
- Apply ticker corrections (ASTR→ALAB, etc.)

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
            
            # Apply ticker corrections to extracted tickers (ASTR -> ALAB, etc.)
            structured_data = self._apply_ticker_corrections_to_data(structured_data)
            
            # Filter out tickers that don't have trading details (including required assets)
            # Only keep tickers with actual trading analysis (support, resistance, targets, sentiment, prices, etc.)
            filtered_tickers = []
            for ticker_data in structured_data.get('tickers', []):
                # Check if ticker has any trading details
                resistance = ticker_data.get('resistance')
                support = ticker_data.get('support')
                target = ticker_data.get('target')
                entry_price = ticker_data.get('entry_price')
                stop_loss = ticker_data.get('stop_loss')
                sentiment = ticker_data.get('sentiment')
                notes = ticker_data.get('notes')
                
                # Check if any trading detail exists (not null/empty)
                has_resistance = resistance and resistance not in [None, 'null', 'None', '']
                has_support = support and support not in [None, 'null', 'None', '']
                has_target = target and target not in [None, 'null', 'None', '']
                has_entry = entry_price and entry_price not in [None, 'null', 'None', '']
                has_stop = stop_loss and stop_loss not in [None, 'null', 'None', '']
                has_sentiment = sentiment and sentiment not in [None, 'null', 'None', 'Neutral', '']
                # Notes must have meaningful content (not placeholder messages)
                empty_notes = ['', 'Ticker was mentioned in transcript but specific details were not found.', 
                              'Required asset was mentioned in transcript but specific trading details were not found.']
                has_notes = notes and notes not in [None, 'null', 'None'] + empty_notes
                
                # Only include if it has at least one trading detail
                has_trading_details = (has_resistance or has_support or has_target or has_entry or 
                                     has_stop or has_sentiment or has_notes)
                
                # Only include if it has trading details
                if has_trading_details:
                    filtered_tickers.append(ticker_data)
            
            structured_data['tickers'] = filtered_tickers
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
                    # Try to get from validated map first
                    company_name = self.validated_ticker_map.get(ticker, None)
                    # If not in validated map, check if it's a required asset
                    if not company_name and ticker in REQUIRED_ASSETS:
                        company_name = self._get_asset_name(ticker)
                    # Final fallback
                    if not company_name:
                        company_name = 'Unknown Company'
                
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
    
    def _apply_ticker_corrections_to_transcript(self, transcript: str) -> str:
        """Apply ticker corrections to transcript before extraction"""
        corrected = transcript
        if self.ticker_corrections:
            # Apply corrections (e.g., ASTR -> ALAB in context)
            # We need to be careful to only replace ticker mentions, not words
            for incorrect, correct in self.ticker_corrections.items():
                # Replace ticker codes in brackets or standalone mentions
                # Pattern: [ASTR] or "ASTR" or ASTR (as ticker)
                pattern = r'\b' + re.escape(incorrect) + r'\b'
                corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)
        return corrected
    
    def _apply_ticker_corrections_to_data(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ticker corrections to extracted JSON data"""
        corrected_tickers = []
        for ticker_data in structured_data.get('tickers', []):
            ticker = ticker_data.get('ticker', '').upper()
            original_company_name = ticker_data.get('company_name', 'Unknown Company')
            
            # Apply correction if needed
            if self.ticker_corrections and ticker in self.ticker_corrections:
                corrected_ticker = self.ticker_corrections[ticker]
                ticker_data['ticker'] = corrected_ticker
                ticker = corrected_ticker  # Update ticker variable for subsequent checks
            
            # Ensure company name is set properly (priority: validated map > asset name > original)
            if ticker in self.validated_ticker_map:
                # Use validated company name if available
                ticker_data['company_name'] = self.validated_ticker_map[ticker]
            elif ticker in REQUIRED_ASSETS:
                # Use asset name for required assets
                ticker_data['company_name'] = self._get_asset_name(ticker)
            elif original_company_name and original_company_name != 'Unknown Company':
                # Keep original if it's not Unknown Company
                ticker_data['company_name'] = original_company_name
            else:
                # Final fallback
                ticker_data['company_name'] = 'Unknown Company'
            
            corrected_tickers.append(ticker_data)
        
        structured_data['tickers'] = corrected_tickers
        return structured_data
    
    def _find_mentioned_required_assets(self, transcript: str) -> set:
        """Find which required assets are mentioned in transcript"""
        mentioned = set()
        transcript_upper = transcript.upper()
        
        # Map of ticker to possible mentions in Turkish/English
        asset_mentions = {
            'US500': ['US500', 'S&P 500', 'SMP 500', 'S&P500', 'SMP500', 'SPX'],
            'US100': ['US100', 'NASDAQ', 'NDX', 'NASDAQ 100', 'NAS100'],
            'RTY': ['RTY', 'RUSSELL', 'RUT', 'RUSSELL 2000'],
            'VIX': ['VIX'],
            'BTCUSD': ['BTCUSD', 'BITCOIN', 'BTC', 'BITCOIN USD'],
            'ETHUSD': ['ETHUSD', 'ETHEREUM', 'ETH', 'ETHEREUM USD'],
            'SOLUSD': ['SOLUSD', 'SOLANA', 'SOL', 'SOLANA USD'],
            'MSTR': ['MSTR', 'MICROSTRATEGY', 'MICRO STRATEGY'],
            'COIN': ['COIN', 'COINBASE'],
            'AAPL': ['AAPL', 'APPLE'],
            'MSFT': ['MSFT', 'MICROSOFT'],
            'NVDA': ['NVDA', 'NVIDIA', 'NVIDIA'],
            'AMZN': ['AMZN', 'AMAZON'],
            'META': ['META', 'FACEBOOK'],
            'GOOG': ['GOOG', 'GOOGLE', 'ALPHABET'],
            'TSLA': ['TSLA', 'TESLA'],
            'AMD': ['AMD'],
            'PLTR': ['PLTR', 'PALANTIR'],
            'CRWD': ['CRWD', 'CROWDSTRIKE'],
            'AVGO': ['AVGO', 'BROADCOM'],
            'HOOD': ['HOOD', 'ROBINHOOD'],
            'HIMS': ['HIMS', 'HIMS & HERS'],
            'SOFI': ['SOFI', 'SOFI TECHNOLOGIES'],
            'APP': ['APP', 'APPLOVIN'],
            'RKLB': ['RKLB', 'ROCKET LAB'],
            'EOSE': ['EOSE', 'EOS ENERGY'],
            'CEG': ['CEG', 'CONSTELLATION ENERGY'],
            'IBKR': ['IBKR', 'INTERACTIVE BROKERS'],
            'AXON': ['AXON'],
            'UNH': ['UNH', 'UNITEDHEALTH', 'UNITED HEALTH'],
            'LLY': ['LLY', 'ELI LILLY', 'LILLY'],
            'ANET': ['ANET', 'ARISTA'],
            'ALAB': ['ALAB', 'ASTERA LABS', 'ASTRALABS', 'ASTERALABS'],
            'CRM': ['CRM', 'SALESFORCE'],
            'SE': ['SE', 'SEA LIMITED', 'SEA LIMIT'],
            'GRAB': ['GRAB', 'GRAB HOLDINGS'],
            'CLS': ['CLS', 'CELESTICA'],
            'CELH': ['CELH', 'CELSIUS'],
            'ZETA': ['ZETA'],
            'NBIS': ['NBIS', 'NEBIUS', 'EN MISLI'],
            'CRDO': ['CRDO', 'CREDO', 'CRIDO'],
            'OSCR': ['OSCR', 'OSCAR'],
            'TEM': ['TEM', 'TEMPUS'],
            'LMND': ['LMND', 'LEMONADE'],
            'MRVL': ['MRVL', 'MARVELL', 'MARVEL'],
            'MU': ['MU', 'MICRON'],
            'IREN': ['IREN', 'IRIS ENERGY', 'IRIS ENERJI'],
            'DLO': ['DLO', 'DLOCAL', 'DEAL OKUL LIMIT'],
            'CRWV': ['CRWV', 'COREWAVE', 'CORE WAVE']
        }
        
        for ticker in REQUIRED_ASSETS:
            if ticker in asset_mentions:
                for mention in asset_mentions[ticker]:
                    if mention in transcript_upper:
                        mentioned.add(ticker)
                        break
        
        return mentioned
    
    def _get_asset_name(self, ticker: str) -> str:
        """Get proper name for asset/ticker"""
        asset_names = {
            'US500': 'S&P 500 Index',
            'US100': 'NASDAQ 100 Index',
            'RTY': 'Russell 2000 Index',
            'VIX': 'CBOE Volatility Index',
            'BTCUSD': 'Bitcoin',
            'ETHUSD': 'Ethereum',
            'SOLUSD': 'Solana',
            'MSTR': 'MicroStrategy Inc.',
            'COIN': 'Coinbase Global, Inc.',
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'NVDA': 'NVIDIA Corporation',
            'AMZN': 'Amazon.com, Inc.',
            'META': 'Meta Platforms, Inc.',
            'GOOG': 'Alphabet Inc.',
            'TSLA': 'Tesla, Inc.',
            'AMD': 'Advanced Micro Devices, Inc.',
            'PLTR': 'Palantir Technologies Inc.',
            'CRWD': 'CrowdStrike Holdings, Inc.',
            'AVGO': 'Broadcom Inc.',
            'HOOD': 'Robinhood Markets, Inc.',
            'HIMS': 'Hims & Hers Health, Inc.',
            'SOFI': 'SoFi Technologies, Inc.',
            'APP': 'AppLovin Corporation',
            'RKLB': 'Rocket Lab Corporation',
            'EOSE': 'Eos Energy Enterprises, Inc.',
            'CEG': 'Constellation Energy Corporation',
            'IBKR': 'Interactive Brokers Group, Inc.',
            'AXON': 'Axon Enterprise, Inc.',
            'UNH': 'UnitedHealth Group Incorporated',
            'LLY': 'Eli Lilly and Company',
            'ANET': 'Arista Networks Inc',
            'ALAB': 'Astera Labs, Inc.',
            'CRM': 'Salesforce, Inc.',
            'SE': 'Sea Limited',
            'GRAB': 'Grab Holdings Limited',
            'CLS': 'Celestica Inc.',
            'CELH': 'Celsius Holdings, Inc.',
            'ZETA': 'Zeta Global Holdings Corp.',
            'NBIS': 'Nebius Group N.V.',
            'CRDO': 'Credo Technology Group Holding Ltd',
            'OSCR': 'Oscar Health, Inc.',
            'TEM': 'Tempus AI, Inc.',
            'LMND': 'Lemonade, Inc.',
            'MRVL': 'Marvell Technology, Inc.',
            'MU': 'Micron Technology, Inc.',
            'IREN': 'IREN Limited',
            'DLO': 'DLocal Limited',
            'CRWV': 'CoreWeave, Inc.'
        }
        return asset_names.get(ticker.upper(), 'Unknown Company')

