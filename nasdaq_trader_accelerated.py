#!/usr/bin/env python3
"""
Accelerated Nasdaq Trader - Local Version
Optimized for maximum performance with parallel processing
"""

import os
import sys
import time
import re
import multiprocessing
import concurrent.futures
import logging
from pathlib import Path
import psutil
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules directly
import whisper
import google.generativeai as genai
from yt_dlp import YoutubeDL
import yfinance as yf
from dotenv import load_dotenv
import warnings
from ticker_validator import get_ticker_validator, validate_ticker, validate_tickers

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found, using defaults")
        return {
            "ACCELERATION": {
                "parallel_videos": 2,
                "max_workers": 4,
                "use_gpu": False,
                "optimize_memory": True
            },
            "MODELS": {
                "whisper_model": "small",
                "gemini_model": "gemini-2.5-flash"
            }
        }
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        return {
            "ACCELERATION": {
                "parallel_videos": 2,
                "max_workers": 4,
                "use_gpu": False,
                "optimize_memory": True
            },
            "MODELS": {
                "whisper_model": "small",
                "gemini_model": "gemini-2.5-flash"
            }
        }

class AcceleratedNasdaqTrader:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.setup_logging()
        self.system_info = self.get_system_info()
        self.optimal_settings = self.calculate_optimal_settings()
        
        # Initialize ticker validator
        self.ticker_validator = get_ticker_validator()
        
        # Load ticker corrections from config (defaults to empty dict if not configured)
        ticker_corrections_config = self.config.get('TICKER_CORRECTIONS', {})
        self.ticker_corrections = {k.upper(): v.upper() for k, v in ticker_corrections_config.items()} if ticker_corrections_config else {}
        
        print(f"Accelerated Nasdaq Trader Initialized")
        print(f"   System: {self.system_info['cpu_cores']} cores, {self.system_info['ram_gb']:.1f}GB RAM")
        print(f"   Optimal: {self.optimal_settings['parallel_videos']} parallel videos")
        print(f"   Ticker Validation: Enabled with caching")
        if self.ticker_corrections:
            print(f"   Ticker Corrections: {len(self.ticker_corrections)} mappings configured")
    
    def get_system_info(self):
        """Get system information for optimization"""
        return {
            "cpu_cores": multiprocessing.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_usage": psutil.cpu_percent(interval=1)
        }
    
    def calculate_optimal_settings(self):
        """Calculate optimal processing settings based on system specs"""
        cpu_cores = self.system_info['cpu_cores']
        available_ram = self.system_info['available_ram_gb']
        
        # Calculate parallel processing based on CPU cores
        if cpu_cores >= 8:
            parallel_videos = min(4, cpu_cores // 2)
            max_workers = cpu_cores
        elif cpu_cores >= 4:
            parallel_videos = min(3, cpu_cores // 2)
            max_workers = cpu_cores
        else:
            parallel_videos = 1
            max_workers = max(2, cpu_cores)
        
        # Memory-based optimizations
        if available_ram >= 16:
            batch_size = 4
            quality_mode = "balanced"
        elif available_ram >= 8:
            batch_size = 3
            quality_mode = "fast"
        else:
            batch_size = 2
            quality_mode = "fast"
        
        return {
            "parallel_videos": parallel_videos,
            "max_workers": max_workers,
            "batch_size": batch_size,
            "quality_mode": quality_mode,
            "use_gpu": self.check_gpu_availability()
        }
    
    def check_gpu_availability(self):
        """Check if GPU is available for processing"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    # Constants for null/empty value checking
    NULL_VALUES = ['', 'null', 'None', 'Not mentioned', 'Not specified']
    
    def _is_null_or_empty(self, value: Any) -> bool:
        """Helper function to check if value is null or empty"""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() in self.NULL_VALUES
        return False
    
    def _parse_timestamp_robust(self, timestamp_str: str) -> Optional[int]:
        """Robust timestamp parsing with validation"""
        if not timestamp_str or timestamp_str in self.NULL_VALUES:
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
    
    def _consolidate_duplicate_tickers(self, analysis_text: str) -> str:
        """Post-process analysis to consolidate duplicate ticker mentions"""
        import re
        
        # Pattern to match ticker sections: ### Company Name (TICKER)
        ticker_section_pattern = r'###\s+([^*\n(]+?)\s*\(([A-Z0-9]+)\)'
        
        # Find all ticker sections
        sections = []
        matches = list(re.finditer(ticker_section_pattern, analysis_text, re.IGNORECASE))
        
        if len(matches) <= 1:
            # No duplicates to consolidate
            return analysis_text
        
        # Group sections by ticker (case-insensitive)
        ticker_groups = {}
        for match in matches:
            ticker = match.group(2).upper()
            start_pos = match.start()
            # Find the end of this section (next ### or end of text)
            next_match = None
            for m in matches:
                if m.start() > start_pos:
                    next_match = m
                    break
            if next_match:
                end_pos = next_match.start()
            else:
                # Find next section header or end of TRADING OPPORTUNITIES
                next_section = analysis_text.find('##', start_pos + 1)
                end_pos = next_section if next_section != -1 else len(analysis_text)
            
            section_text = analysis_text[start_pos:end_pos].strip()
            
            if ticker not in ticker_groups:
                ticker_groups[ticker] = []
            ticker_groups[ticker].append({
                'text': section_text,
                'start': start_pos,
                'end': end_pos,
                'company_name': match.group(1).strip()
            })
        
        # Consolidate duplicates
        consolidated_text = analysis_text
        offset = 0  # Track offset from deletions
        
        for ticker, sections in ticker_groups.items():
            if len(sections) <= 1:
                continue  # No duplicates for this ticker
            
            self.logger.info(f"Consolidating {len(sections)} duplicate sections for ticker {ticker}")
            
            # Sort by position (first occurrence first)
            sections.sort(key=lambda x: x['start'])
            
            # Use first section as base
            base_section = sections[0]
            base_text = base_section['text']
            base_company_name = base_section['company_name']
            
            # Extract fields from all sections
            all_timestamps = []
            all_sentiments = []
            all_resistances = []
            all_supports = []
            all_targets = []
            all_notes = []
            
            for section in sections:
                section_text = section['text']
                # Extract timestamp
                ts_match = re.search(r'- \*\*Timestamp\*\*:\s*([^\n]+)', section_text)
                if ts_match:
                    all_timestamps.append(ts_match.group(1).strip())
                # Extract sentiment
                sent_match = re.search(r'- \*\*Sentiment\*\*:\s*([^\n]+)', section_text)
                if sent_match:
                    all_sentiments.append(sent_match.group(1).strip())
                # Extract resistance
                res_match = re.search(r'- \*\*Resistance\*\*:\s*([^\n]+)', section_text)
                if res_match and res_match.group(1).strip() and not self._is_null_or_empty(res_match.group(1).strip()):
                    all_resistances.append(res_match.group(1).strip())
                # Extract support
                sup_match = re.search(r'- \*\*Support\*\*:\s*([^\n]+)', section_text)
                if sup_match and sup_match.group(1).strip() and not self._is_null_or_empty(sup_match.group(1).strip()):
                    all_supports.append(sup_match.group(1).strip())
                # Extract target
                tgt_match = re.search(r'- \*\*Target\*\*:\s*([^\n]+)', section_text)
                if tgt_match and tgt_match.group(1).strip() and not self._is_null_or_empty(tgt_match.group(1).strip()):
                    all_targets.append(tgt_match.group(1).strip())
                # Extract notes
                notes_match = re.search(r'- \*\*Notes\*\*:\s*([^\n]+(?:\n(?!- \*\*|###|##)[^\n]+)*)', section_text, re.MULTILINE)
                if notes_match:
                    notes_text = notes_match.group(1).strip()
                    if notes_text and not self._is_null_or_empty(notes_text):
                        all_notes.append(notes_text)
            
            # Build consolidated section
            consolidated_section = f"### {base_company_name} ({ticker})\n"
            
            # Use earliest valid timestamp
            valid_timestamps = [ts for ts in all_timestamps if ts and not self._is_null_or_empty(ts)]
            if valid_timestamps:
                # Sort by time value
                sorted_timestamps = sorted(valid_timestamps, key=lambda x: self._parse_timestamp_robust(x) or float('inf'))
                consolidated_section += f"- **Timestamp**: {sorted_timestamps[0]}\n"
            else:
                consolidated_section += "- **Timestamp**: Not mentioned\n"
            
            # Use most specific sentiment (prefer non-Neutral)
            non_neutral_sentiments = [s for s in all_sentiments if s and 'Neutral' not in s]
            if non_neutral_sentiments:
                consolidated_section += f"- **Sentiment**: {non_neutral_sentiments[0]}\n"
            elif all_sentiments:
                consolidated_section += f"- **Sentiment**: {all_sentiments[0]}\n"
            else:
                consolidated_section += "- **Sentiment**: Neutral\n"
            
            # Merge resistance (use first non-empty, or combine if multiple)
            if all_resistances:
                unique_resistances = list(dict.fromkeys(all_resistances))  # Preserve order, remove duplicates
                if len(unique_resistances) == 1:
                    consolidated_section += f"- **Resistance**: {unique_resistances[0]}\n"
                else:
                    consolidated_section += f"- **Resistance**: {', '.join(unique_resistances)}\n"
            else:
                consolidated_section += "- **Resistance**:\n"
            
            # Merge support (use first non-empty, or combine if multiple)
            if all_supports:
                unique_supports = list(dict.fromkeys(all_supports))
                if len(unique_supports) == 1:
                    consolidated_section += f"- **Support**: {unique_supports[0]}\n"
                else:
                    consolidated_section += f"- **Support**: {', '.join(unique_supports)}\n"
            else:
                consolidated_section += "- **Support**:\n"
            
            # Merge target (use first non-empty, or combine if multiple)
            if all_targets:
                unique_targets = list(dict.fromkeys(all_targets))
                if len(unique_targets) == 1:
                    consolidated_section += f"- **Target**: {unique_targets[0]}\n"
                else:
                    consolidated_section += f"- **Target**: {', '.join(unique_targets)}\n"
            else:
                consolidated_section += "- **Target**:\n"
            
            # Merge notes intelligently
            merged_notes = ""
            for note in all_notes:
                merged_notes = self._merge_notes_intelligently(merged_notes, note)
            if merged_notes:
                consolidated_section += f"- **Notes**: {merged_notes}\n"
            else:
                consolidated_section += "- **Notes**:\n"
            
            consolidated_section += "\n"
            
            # Replace all duplicate sections with consolidated one
            # Work backwards to maintain positions
            for i in range(len(sections) - 1, -1, -1):
                section = sections[i]
                if i == 0:
                    # Replace first occurrence with consolidated version
                    consolidated_text = (
                        consolidated_text[:section['start'] + offset] +
                        consolidated_section +
                        consolidated_text[section['end'] + offset:]
                    )
                else:
                    # Remove duplicate occurrences
                    consolidated_text = (
                        consolidated_text[:section['start'] + offset] +
                        consolidated_text[section['end'] + offset:]
                    )
                    offset -= (section['end'] - section['start'])
        
        return consolidated_text
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """Parse price string to float, handling various formats"""
        if not price_str or self._is_null_or_empty(price_str):
            return None
        
        # Remove common prefixes/suffixes
        price_str = price_str.strip()
        price_str = price_str.replace('$', '').replace(',', '').replace('~', '').strip()
        
        # Handle ranges (e.g., "32.00-35.00" -> use first value)
        if '-' in price_str and not price_str.startswith('-'):
            price_str = price_str.split('-')[0].strip()
        
        # Handle ">" prefix (e.g., ">100.00" -> "100.00")
        if price_str.startswith('>'):
            price_str = price_str[1:].strip()
        
        try:
            return float(price_str)
        except (ValueError, AttributeError):
            return None
    
    def _calculate_risk(self, entry: float, stop: float) -> Optional[str]:
        """Calculate risk percentage from entry and stop"""
        if entry is None or stop is None or entry <= 0:
            return None
        
        try:
            risk_pct = abs(entry - stop) / entry * 100
            return f"{risk_pct:.1f}%"
        except (ZeroDivisionError, ValueError):
            return None
    
    def _calculate_risk_reward(self, entry: float, stop: float, target: float) -> Optional[str]:
        """Calculate risk/reward ratio from entry, stop, and target"""
        if entry is None or stop is None or target is None:
            return None
        
        try:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            
            if risk == 0:
                return None
            
            ratio = reward / risk
            return f"1:{ratio:.1f}"
        except (ZeroDivisionError, ValueError):
            return None
    
    def _extract_support_from_trading_opportunities(self, analysis_text: str, ticker: str) -> Optional[float]:
        """Extract support level from TRADING OPPORTUNITIES section for a ticker"""
        import re
        
        # Find ticker section in TRADING OPPORTUNITIES
        ticker_pattern = rf'###\s+[^*\n(]+?\s*\({re.escape(ticker)}\)'
        match = re.search(ticker_pattern, analysis_text, re.IGNORECASE)
        
        if not match:
            return None
        
        # Find the section content (until next ### or ##)
        section_start = match.end()
        next_section = analysis_text.find('##', section_start + 1)
        section_end = next_section if next_section != -1 else len(analysis_text)
        section_text = analysis_text[section_start:section_end]
        
        # Extract support value
        support_match = re.search(r'- \*\*Support\*\*:\s*([^\n]+)', section_text, re.IGNORECASE)
        if support_match:
            support_str = support_match.group(1).strip()
            # Extract first numeric value from support string
            price_match = re.search(r'\$?([\d,]+\.?\d*)', support_str)
            if price_match:
                return self._parse_price(price_match.group(1))
        
        return None
    
    def _enhance_high_potential_trades(self, analysis_text: str) -> str:
        """Post-process HIGH POTENTIAL TRADES section to calculate missing values and ensure trades are shown"""
        import re
        
        # Find HIGH POTENTIAL TRADES section
        hpt_pattern = r'##\s*🎯\s*HIGH\s+POTENTIAL\s+TRADES'
        hpt_match = re.search(hpt_pattern, analysis_text, re.IGNORECASE)
        
        if not hpt_match:
            # No HIGH POTENTIAL TRADES section - create fallback from TRADING OPPORTUNITIES
            self.logger.info("No HIGH POTENTIAL TRADES section found, creating fallback from TRADING OPPORTUNITIES")
            fallback_section = self._create_fallback_high_potential_trades(analysis_text)
            if fallback_section:
                # Append fallback section before end of report
                return analysis_text.rstrip() + "\n\n" + fallback_section
            return analysis_text
        
        section_start = hpt_match.start()
        
        # Find end of section (next ## header or end of text)
        next_section = analysis_text.find('##', section_start + 1)
        section_end = next_section if next_section != -1 else len(analysis_text)
        
        section_text = analysis_text[section_start:section_end]
        
        # Check if section has generic message (no trades)
        if re.search(r'belirtilen işlem fikirleri için net giriş|net giriş.*stop.*risk.*verilmemiş|spesifik alım.*satım emirleri', section_text, re.IGNORECASE):
            # Generic message found - replace with fallback
            self.logger.info("HIGH POTENTIAL TRADES section has generic message, creating fallback from TRADING OPPORTUNITIES")
            enhanced_section = self._create_fallback_high_potential_trades(analysis_text)
            if enhanced_section:
                return analysis_text[:section_start] + enhanced_section + analysis_text[section_end:]
            return analysis_text
        
        # Parse existing trades - handle both numbered and unnumbered formats
        trade_pattern = r'\*\*(\d+)\.\*\*\s*\*\*([^*\n(]+?)\s*\(([A-Z0-9]+)\)\*\*:\s*([A-Z/\s]+)\s*-?\s*(.*?)(?=\*\*\d+\.\*\*|\*\*[A-Z]|$)'
        trades = list(re.finditer(trade_pattern, section_text, re.IGNORECASE | re.DOTALL))
        
        # Also try unnumbered format
        if not trades:
            trade_pattern = r'\*\*([^*\n(]+?)\s*\(([A-Z0-9]+)\)\*\*:\s*([A-Z/\s]+)\s*-?\s*(.*?)(?=\*\*[A-Z]|$)'
            trades = list(re.finditer(trade_pattern, section_text, re.IGNORECASE | re.DOTALL))
        
        enhanced_trades = []
        trade_count = 0
        
        for trade_match in trades:
            trade_count += 1
            
            # Handle both numbered and unnumbered formats
            if len(trade_match.groups()) == 5:
                number = trade_match.group(1)
                company_name = trade_match.group(2).strip()
                ticker = trade_match.group(3).upper()
                action = trade_match.group(4).strip()
                trade_content = trade_match.group(5)
            else:
                number = str(trade_count)
                company_name = trade_match.group(1).strip()
                ticker = trade_match.group(2).upper()
                action = trade_match.group(3).strip()
                trade_content = trade_match.group(4)
            
            # Extract Entry, Stop, Target, Risk, Risk/Reward
            entry_match = re.search(r'Entry:\s*\*\*([^*]+)\*\*', trade_content, re.IGNORECASE)
            stop_match = re.search(r'Stop:\s*\*\*([^*]+)\*\*', trade_content, re.IGNORECASE)
            target_match = re.search(r'Target:\s*\*\*([^*]+)\*\*', trade_content, re.IGNORECASE)
            risk_match = re.search(r'Risk:\s*\*\*([^*]+)\*\*', trade_content, re.IGNORECASE)
            risk_reward_match = re.search(r'Risk/Reward:\s*\*\*([^*]+)\*\*', trade_content, re.IGNORECASE)
            
            entry_str = entry_match.group(1).strip() if entry_match else None
            stop_str = stop_match.group(1).strip() if stop_match else None
            target_str = target_match.group(1).strip() if target_match else None
            risk_str = risk_match.group(1).strip() if risk_match else None
            risk_reward_str = risk_reward_match.group(1).strip() if risk_reward_match else None
            
            # Parse prices
            entry = self._parse_price(entry_str) if entry_str else None
            stop = self._parse_price(stop_str) if stop_str else None
            target = self._parse_price(target_str) if target_str else None
            
            # Enhance missing values
            # 1. If Entry missing but Support exists in TRADING OPPORTUNITIES, infer Entry
            if entry is None:
                support = self._extract_support_from_trading_opportunities(analysis_text, ticker)
                if support is not None:
                    entry = support
                    entry_str = f"${entry:.2f}"
                    self.logger.info(f"Inferred Entry ${entry:.2f} from Support for {ticker}")
            
            # 2. If Stop missing but Entry exists, calculate Stop (2% below Entry)
            if stop is None and entry is not None:
                stop = entry * 0.98  # 2% buffer
                stop_str = f"${stop:.2f}"
                self.logger.info(f"Calculated Stop ${stop:.2f} (2% below Entry) for {ticker}")
            
            # 3. If Risk missing but Entry/Stop exist, calculate Risk
            if risk_str is None and entry is not None and stop is not None:
                risk_str = self._calculate_risk(entry, stop)
                if risk_str:
                    self.logger.info(f"Calculated Risk {risk_str} for {ticker}")
            
            # 4. If Risk/Reward missing but Entry/Stop/Target exist, calculate Risk/Reward
            if risk_reward_str is None and entry is not None and stop is not None and target is not None:
                risk_reward_str = self._calculate_risk_reward(entry, stop, target)
                if risk_reward_str:
                    self.logger.info(f"Calculated Risk/Reward {risk_reward_str} for {ticker}")
            
            # Build enhanced trade line
            action_parts = []
            if entry_str:
                action_parts.append(f"Entry: **{entry_str}**")
            if stop_str:
                action_parts.append(f"Stop: **{stop_str}**")
            if target_str:
                action_parts.append(f"Target: **{target_str}**")
            if risk_str:
                action_parts.append(f"Risk: **{risk_str}**")
            if risk_reward_str:
                action_parts.append(f"Risk/Reward: **{risk_reward_str}**")
            
            action_line = " - ".join([f"[{p}]" for p in action_parts]) if action_parts else ""
            
            # Extract reason
            reason_match = re.search(r'\*\[Reason:\s*([^\]]+)\]', trade_content, re.IGNORECASE)
            reason = reason_match.group(1).strip() if reason_match else ""
            
            # Build enhanced trade
            enhanced_trade = f"**{number}.** **{company_name} ({ticker})**: {action} {action_line}"
            if reason:
                enhanced_trade += f"\n   *[Reason: {reason}]*"
            
            enhanced_trades.append(enhanced_trade)
        
        # If no trades found in section, create fallback
        if trade_count == 0:
            self.logger.info("No trades found in HIGH POTENTIAL TRADES section, creating fallback")
            enhanced_section = self._create_fallback_high_potential_trades(analysis_text)
            if enhanced_section:
                return analysis_text[:section_start] + enhanced_section + analysis_text[section_end:]
            return analysis_text
        
        # Filter and rank trades by quality - keep only truly high potential ones
        filtered_trades = self._filter_high_quality_trades(enhanced_trades)
        
        # Rebuild section with filtered trades
        enhanced_section = "## 🎯 HIGH POTENTIAL TRADES\n\n"
        enhanced_section += "\n\n".join(filtered_trades)
        enhanced_section += "\n"
        
        # Replace original section
        return analysis_text[:section_start] + enhanced_section + analysis_text[section_end:]
    
    def _filter_high_quality_trades(self, trades: list) -> list:
        """Filter and rank trades by quality metrics - keep only truly high potential ones (no limit)"""
        import re
        
        scored_trades = []
        
        for trade in trades:
            score = 0
            trade_data = {'trade': trade, 'score': 0}
            
            # EXCLUDE TAKE PROFIT/SELL trades - these are not "high potential" trades
            if re.search(r':\s*TAKE\s+PROFIT|:\s*SELL\s+', trade, re.IGNORECASE):
                continue  # Skip TAKE PROFIT/SELL trades entirely
            
            # Extract Risk/Reward ratio
            rr_match = re.search(r'Risk/Reward:\s*\*\*1:([\d.]+)\*\*', trade, re.IGNORECASE)
            if rr_match:
                try:
                    rr_ratio = float(rr_match.group(1))
                    trade_data['risk_reward'] = rr_ratio
                    # Higher Risk/Reward = higher score
                    score += min(rr_ratio * 2, 50)  # Cap at 50 points
                except ValueError:
                    pass
            
            # Extract Risk percentage
            risk_match = re.search(r'Risk:\s*\*\*([\d.]+)%\*\*', trade, re.IGNORECASE)
            if risk_match:
                try:
                    risk_pct = float(risk_match.group(1))
                    trade_data['risk'] = risk_pct
                    # Lower risk = higher score (prefer trades with risk < 5%)
                    if risk_pct < 5:
                        score += 10
                    elif risk_pct < 10:
                        score += 5
                except ValueError:
                    pass
            
            # Check for BUY action (prefer BUY)
            if re.search(r':\s*BUY\s+', trade, re.IGNORECASE):
                score += 20
            
            # Check for Target (trades with targets are more actionable)
            if re.search(r'Target:\s*\*\*', trade, re.IGNORECASE):
                score += 10
            
            # Check for complete data (Entry, Stop, Target, Risk, Risk/Reward all present)
            has_entry = bool(re.search(r'Entry:\s*\*\*', trade, re.IGNORECASE))
            has_stop = bool(re.search(r'Stop:\s*\*\*', trade, re.IGNORECASE))
            has_target = bool(re.search(r'Target:\s*\*\*', trade, re.IGNORECASE))
            has_risk = bool(re.search(r'Risk:\s*\*\*', trade, re.IGNORECASE))
            has_rr = bool(re.search(r'Risk/Reward:\s*\*\*', trade, re.IGNORECASE))
            
            complete_data_score = sum([has_entry, has_stop, has_target, has_risk, has_rr]) * 2
            score += complete_data_score
            
            # STRICT FILTERING: Exclude trades with very low Risk/Reward ratios
            # Only keep trades with Risk/Reward >= 15:1 (truly high potential)
            if 'risk_reward' in trade_data:
                if trade_data['risk_reward'] < 15:
                    # Low Risk/Reward - exclude entirely
                    continue  # Skip low Risk/Reward trades
            
            trade_data['score'] = score
            scored_trades.append(trade_data)
        
        # Sort by score (highest first)
        scored_trades.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter by minimum quality score - keep all trades that meet quality threshold
        # NO LIMIT on number of trades - show all high potential ones
        min_quality_score = 50  # Very high threshold for truly exceptional trades
        filtered = [t['trade'] for t in scored_trades if t['score'] >= min_quality_score]
        
        # If no trades meet the threshold, keep at least top 5 (fallback)
        if not filtered and scored_trades:
            filtered = [t['trade'] for t in scored_trades[:5]]
            self.logger.info(f"No trades met quality threshold ({min_quality_score}), keeping top 5")
        
        # Renumber the trades
        renumbered_trades = []
        for i, trade in enumerate(filtered, 1):
            # Replace the number at the start
            renumbered = re.sub(r'\*\*(\d+)\.\*\*', f'**{i}.**', trade, count=1)
            renumbered_trades.append(renumbered)
        
        self.logger.info(f"Filtered {len(trades)} trades to {len(renumbered_trades)} high-quality trades (no limit, quality threshold: {min_quality_score}, min Risk/Reward: 15:1)")
        return renumbered_trades
    
    def _create_fallback_high_potential_trades(self, analysis_text: str) -> Optional[str]:
        """Create HIGH POTENTIAL TRADES section from TRADING OPPORTUNITIES when LLM didn't generate any"""
        import re
        
        # Find TRADING OPPORTUNITIES section
        to_pattern = r'##\s*📈\s*TRADING\s+OPPORTUNITIES'
        to_match = re.search(to_pattern, analysis_text, re.IGNORECASE)
        
        if not to_match:
            return None
        
        # Find all ticker sections with BUY/Bullish sentiment and Support/Target
        ticker_pattern = r'###\s+([^*\n(]+?)\s*\(([A-Z0-9]+)\)'
        ticker_matches = list(re.finditer(ticker_pattern, analysis_text[to_match.end():], re.IGNORECASE))
        
        potential_trades = []
        
        for match in ticker_matches[:10]:  # Limit to top 10
            ticker_start = to_match.end() + match.start()
            ticker_end = to_match.end() + match.end()
            
            # Find section content
            next_ticker = analysis_text.find('###', ticker_end + 1)
            next_section = analysis_text.find('##', ticker_end + 1)
            section_end = min(next_ticker, next_section) if next_ticker != -1 and next_section != -1 else (next_ticker if next_ticker != -1 else (next_section if next_section != -1 else len(analysis_text)))
            section_text = analysis_text[ticker_end:section_end]
            
            company_name = match.group(1).strip()
            ticker = match.group(2).upper()
            
            # Extract sentiment
            sentiment_match = re.search(r'- \*\*Sentiment\*\*:\s*([^\n]+)', section_text, re.IGNORECASE)
            sentiment = sentiment_match.group(1).strip() if sentiment_match else ""
            
            # Only include BUY/Bullish trades
            if not re.search(r'BUY|Bullish|Boğa|buy|bullish', sentiment, re.IGNORECASE):
                continue
            
            # Extract Support and Target
            support_match = re.search(r'- \*\*Support\*\*:\s*([^\n]+)', section_text, re.IGNORECASE)
            target_match = re.search(r'- \*\*Target\*\*:\s*([^\n]+)', section_text, re.IGNORECASE)
            
            support_str = support_match.group(1).strip() if support_match else None
            target_str = target_match.group(1).strip() if target_match else None
            
            if not support_str or self._is_null_or_empty(support_str):
                continue
            
            # Parse prices
            support_price = self._parse_price(support_str)
            target_price = self._parse_price(target_str) if target_str else None
            
            if support_price is None:
                continue
            
            # Use Support as Entry
            entry = support_price
            stop = entry * 0.98  # 2% buffer
            risk_str = self._calculate_risk(entry, stop)
            risk_reward_str = self._calculate_risk_reward(entry, stop, target_price) if target_price else None
            
            # Build trade
            action_parts = [f"Entry: **${entry:.2f}**", f"Stop: **${stop:.2f}**"]
            if target_price:
                action_parts.append(f"Target: **${target_price:.2f}**")
            if risk_str:
                action_parts.append(f"Risk: **{risk_str}**")
            if risk_reward_str:
                action_parts.append(f"Risk/Reward: **{risk_reward_str}**")
            
            action_line = " - ".join([f"[{p}]" for p in action_parts])
            
            # Extract notes for reason
            notes_match = re.search(r'- \*\*Notes\*\*:\s*([^\n]+(?:\n(?!- \*\*|###|##)[^\n]+)*)', section_text, re.MULTILINE)
            reason = notes_match.group(1).strip()[:200] if notes_match else sentiment[:200]
            
            potential_trades.append({
                'number': len(potential_trades) + 1,
                'company_name': company_name,
                'ticker': ticker,
                'action': 'BUY',
                'action_line': action_line,
                'reason': reason
            })
        
        if not potential_trades:
            return None
        
        # Build section
        enhanced_section = "## 🎯 HIGH POTENTIAL TRADES\n\n"
        for trade in potential_trades:
            enhanced_section += f"**{trade['number']}.** **{trade['company_name']} ({trade['ticker']})**: {trade['action']} {trade['action_line']}\n"
            if trade['reason']:
                enhanced_section += f"   *[Reason: {trade['reason']}]*\n"
            enhanced_section += "\n"
        
        self.logger.info(f"Created fallback HIGH POTENTIAL TRADES section with {len(potential_trades)} trades")
        return enhanced_section
    
    def validate_tickers_in_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """
        Extract and validate tickers from analysis text
        
        Args:
            analysis_text: The generated analysis text
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Extract tickers from analysis text
            tickers = self.extract_tickers_from_text(analysis_text)
            
            if not tickers:
                self.logger.info("No tickers found in analysis")
                return {
                    'tickers_found': 0,
                    'valid_tickers': [],
                    'invalid_tickers': [],
                    'validation_summary': {}
                }
            
            self.logger.info(f"Found {len(tickers)} tickers to validate: {tickers}")
            
            # Validate tickers
            validation_results = self.ticker_validator.validate_multiple_tickers(tickers)
            
            # Separate valid and invalid tickers
            valid_tickers = []
            invalid_tickers = []
            
            for ticker, result in validation_results.items():
                if result['is_valid']:
                    valid_tickers.append({
                        'ticker': ticker,
                        'company_name': result['company_name'],
                        'info': result['ticker_info']
                    })
                else:
                    invalid_tickers.append({
                        'ticker': ticker,
                        'error': result['error_message']
                    })
            
            # Get validation summary
            summary = self.ticker_validator.get_validation_summary(tickers)
            
            self.logger.info(f"Ticker validation complete: {len(valid_tickers)} valid, {len(invalid_tickers)} invalid")
            
            return {
                'tickers_found': len(tickers),
                'valid_tickers': valid_tickers,
                'invalid_tickers': invalid_tickers,
                'validation_summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Ticker validation failed: {e}")
            return {
                'tickers_found': 0,
                'valid_tickers': [],
                'invalid_tickers': [],
                'validation_summary': {},
                'error': str(e)
            }
    
    def extract_tickers_from_text(self, text: str) -> list:
        """
        Extract ticker symbols from analysis text
        
        Args:
            text: Analysis text to extract tickers from
            
        Returns:
            List of potential ticker symbols
        """
        import re
        
        # Common ticker patterns - improved to catch more tickers including case variations
        ticker_patterns = [
            r'\b[A-Z]{2,5}\b',  # 2-5 uppercase letters (increased min to reduce false positives)
            r'\$[A-Z]{2,5}\b',  # $ followed by 2-5 uppercase letters
            r'\b[A-Z]{2,5}\.\w{1,2}\b',  # Ticker with exchange suffix (e.g., AAPL.NASDAQ)
            # Pattern for tickers mentioned with Turkish possessive suffix
            # Handles: "IREN'e", "IREN'i", "Iren'in", "İren'in" (case-insensitive for first letter)
            r'\b([A-Z][A-Za-z]{1,4})[\'’][a-zığüşöç]',  # Ticker with Turkish possessive suffix (handles "Iren'in" -> IREN)
            # Pattern for standalone capitalized ticker-like words (e.g., "Tem", "Axon", "Iron")
            # This catches tickers mentioned in natural language that might not be all uppercase
            # Look for capitalized words followed by Turkish context words that suggest ticker mentions
            r'\b([A-Z][a-z]{1,4})\b(?=\s+(?:bu|şu|bu seviyeye|seviyeye|bakalım|diyelim|için|ile|gibi|olarak|de|da|den|dan|e|a))',  # Capitalized word followed by Turkish context words
            # Also catch capitalized words that appear in ticker-like contexts (e.g., "Tem bu seviyeye bakalım")
            r'\b([A-Z][A-Za-z]{1,4})\b(?=\s+[a-zığüşöç]+.*(?:bakalım|seviye|direnç|destek|hedef|bilanço))',  # Capitalized word before trading-related Turkish words
        ]
        
        tickers = set()
        
        for pattern in ticker_patterns:
            matches = re.findall(pattern, text)  # Search in original text (case-sensitive for pattern)
            for match in matches:
                # Handle tuple result from capture group
                if isinstance(match, tuple):
                    ticker = match[0]
                else:
                    ticker = match
                # Clean up the match
                ticker = ticker.replace('$', '').strip()
                # Convert to uppercase for consistency (handles "Iren" -> "IREN", "Tem" -> "TEM")
                ticker = ticker.upper()
                # Apply ticker corrections if needed
                if ticker in self.ticker_corrections:
                    corrected_ticker = self.ticker_corrections[ticker]
                    self.logger.info(f"Correcting ticker: {ticker} -> {corrected_ticker}")
                    ticker = corrected_ticker
                
                # Additional corrections for common extraction errors
                # Fix AMBSTE -> NBIS (Turkish pronunciation "Ambiste" was incorrectly extracted as AMBSTE)
                if ticker == 'AMBSTE':
                    ticker = 'NBIS'
                    self.logger.info(f"Correcting extracted ticker: AMBSTE -> NBIS")
                
                # Fix LEMONY -> LMND (Turkish pronunciation "Lemony" was incorrectly extracted as LEMONY)
                if ticker == 'LEMONY':
                    ticker = 'LMND'
                    self.logger.info(f"Correcting extracted ticker: LEMONY -> LMND")
                # Only add if length is valid (2-5 chars for tickers)
                if len(ticker) >= 2 and len(ticker) <= 5:
                    tickers.add(ticker)
        
        # Filter out common false positives (English and Turkish common words)
        false_positives = {
            # English common words
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR',
            'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO',
            'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'AL', 'AS', 'AT', 'BE', 'BY',
            'DO', 'GO', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'WE',
            'AN', 'AM', 'AI', 'OK', 'TV', 'ID', 'OS', 'PC', 'FY', 'IQ', 'QA', 'PM', 'AM', 'IO', 'IE', 'EU',
            # Turkish common words (capitalized versions that might be extracted)
            'BAKIN', 'BELKI', 'BIRAZ', 'BUNDA', 'DAHA', 'DOLAR', 'FIYAT', 'HATTA', 'KADAR', 'OLAN', 'ONDA',
            'ONUN', 'ORADA', 'UZUN', 'YANI', 'YINE', 'ZAMAN', 'ZATEN', 'VAR', 'BIR', 'BU', 'VE', 'YA', 'DA',
            'DE', 'KI', 'ILK', 'SON', 'DAHA', 'BIR', 'BIRAZ', 'BUNDA', 'ONDA', 'ONUN', 'ORADA', 'HATTA',
            # False positive tickers that are not valid
            'DIP', 'IMSANHORSE', 'SOFAY', 'OVEN'  # These are invalid tickers and should be filtered
            # Note: ACPLE is NOT in false_positives - it will be corrected to AAPL via ticker_corrections
        }
        
        # Filter out tickers that are purely numeric (like "257513")
        tickers = {t for t in tickers if not t.isdigit()}
        
        # Apply ticker corrections BEFORE filtering false positives
        # This ensures corrected tickers (like ACPLE -> AAPL) are not filtered out
        corrected_tickers = set()
        for ticker in tickers:
            if ticker in self.ticker_corrections:
                corrected_ticker = self.ticker_corrections[ticker]
                self.logger.info(f"Applying ticker correction during extraction: {ticker} -> {corrected_ticker}")
                corrected_tickers.add(corrected_ticker)
            else:
                corrected_tickers.add(ticker)
        tickers = corrected_tickers
        
        # Additional pass: Look for company names mentioned in transcript that might map to tickers
        # Use generalizable patterns that work across different videos
        # Pattern: Company name (with variations) -> ticker mapping
        company_name_patterns = [
            # Technology/SaaS companies
            (r'\bApple\b', 'AAPL'),  # Apple Inc.
            (r'\bAxon\s+(?:Enterprise|Inc\.?)?\b', 'AXON'),
            (r'\bAxon\s+Enterprise\b', 'AXON'),
            (r'\bUnited\s+Health\b', 'UNH'),
            (r'\bUnitedHealth\b', 'UNH'),
            (r'\bEli\s+Lilly\b', 'LLY'),
            (r'\bEl\s+ayliliğe\b', 'LLY'),  # Turkish pronunciation
            (r'\bArista\s+(?:Networks|Inc\.?)?\b', 'ANET'),
            (r'\bArista\s+Networks\b', 'ANET'),
            (r'\bAstra\s+(?:Space|Inc\.?)?\b', 'ASTR'),  # Will be corrected to ALAB
            (r'\bAstralabs\b', 'ASIL'),
            (r'\bSalesforce\b', 'CRM'),
            (r'\bSea\s+(?:Limited|Limit|Inc\.?)?\b', 'SE'),  # Catches "Sea Limited" and "Sea Limit" (transcription variation)
            (r'\bSea\s+Limited\b', 'SE'),
            (r'\bSea\s+Limit\b', 'SE'),  # Transcription variation
            (r'\bGrab\s+(?:Holdings|Limited|Inc\.?)?\b', 'GRAB'),
            (r'\bGrab\s+Holdings\b', 'GRAB'),
            (r'\bZeta\b', 'ZETA'),
            (r'\bNVIDIA\b', 'NVDA'),
            (r'\bNvidia\b', 'NVDA'),
            (r'\bMicrosoft\b', 'MSFT'),  # Microsoft Corporation
            (r'\bAmazon\b', 'AMZN'),  # Amazon.com Inc.
            (r'\bGoogle\b', 'GOOG'),  # Alphabet Inc. (Google)
            (r'\bCrowdstrike\b', 'CRWD'),  # CrowdStrike Holdings, Inc.
            (r'\bCrowdStrike\b', 'CRWD'),  # CrowdStrike Holdings, Inc.
            (r'\bBroadcom\b', 'AVGO'),  # Broadcom Inc.
            (r'\bRocket\s+Lab\b', 'RKLB'),  # Rocket Lab USA, Inc.
            (r'\bRocket\s+Lab\'e\b', 'RKLB'),  # Turkish grammar variation
            (r'\bEOS\b', 'EOSE'),  # Eos Energy Enterprises, Inc.
            (r'\bConstellation\s+Energy\b', 'CEG'),  # Constellation Energy Corporation
            (r'\bConstellation\s+Energi\b', 'CEG'),  # Constellation Energy Corporation (Turkish spelling)
            (r'\bCoinbase\b', 'COIN'),  # Coinbase Global, Inc.
            (r'\bMeta\b', 'META'),  # Meta Platforms, Inc.
            (r'\bFacebook\b', 'META'),  # Meta Platforms, Inc. (formerly Facebook)
            (r'\bTesla\b', 'TSLA'),  # Tesla, Inc.
            (r'\bPalantir\b', 'PLTR'),  # Palantir Technologies Inc.
            (r'\bRobinhood\b', 'HOOD'),  # Robinhood Markets, Inc.
            (r'\bHims\b', 'HIMS'),  # Hims & Hers Health
            (r'\bApplovin\b', 'APP'),  # Applovin Corporation
            (r'\bApp\s+Lovin\b', 'APP'),  # Applovin Corporation (with space)
            (r'\bEplovin\b', 'APP'),  # Applovin Corporation (Turkish pronunciation)
            (r'\bHymsenhurst\b', 'HIMS'),  # Turkish mispronunciation of "Hims"
            (r'\bEn\s+misli\b', 'NBIS'),  # Transcription: "En misli" is how "NBIS" sounds in Turkish
            (r'\bNBIS\b', 'NBIS'),  # NBIS is a valid ticker (not NVIDIA)
            (r'\bAmbiste\b', 'NBIS'),  # NBIS (Turkish pronunciation - "Ambiste")
            (r'\bCrido\b', 'CRIDO'),  # Will be corrected to CRDO
            (r'\bOscar\s+(?:Health|Inc\.?)?\b', 'OSCR'),  # Oscar Health, Inc.
            (r'\bOscar\s+Health\b', 'OSCR'),
            (r'\bTempus\s+(?:AI|Inc\.?)?\b', 'TEM'),
            (r'\bTempus\s+AI\b', 'TEM'),
            (r'\bTempsey\b', 'TEM'),  # Turkish pronunciation of Tempus
            (r'\bLemonade\b', 'LMND'),
            (r'\bLemmon\s*8\b', 'LMND'),  # Transcription variation
            (r'\bLemmon8\b', 'LMND'),  # Transcription variation
            (r'\bLemony\b', 'LMND'),  # Lemonade, Inc. (Turkish pronunciation)
            (r'\bInteractive\s+Brokers\b', 'IBKR'),
            (r'\bIBEKARAY\b', 'IBKR'),  # Turkish pronunciation
            (r'\bibekaray\b', 'IBKR'),  # Turkish pronunciation (lowercase)
            (r'\bCambium\s+Learning\s+Group\b', 'IBKR'),  # Cambium Learning Group, Inc. (ABCD) is wrong, correct is IBKR
            (r'\bAstera\s+Labs\b', 'ALAB'),
            (r'\bASTRALAC\b', 'ALAB'),  # Turkish mispronunciation
            (r'\bastralac\b', 'ALAB'),  # Turkish mispronunciation (lowercase)
            (r'\bastralaç\b', 'ALAB'),  # Turkish mispronunciation with ç
            (r'\bAstralapse\b', 'ALAB'),  # Turkish mispronunciation of Astera Labs
            (r'\bastralapse\b', 'ALAB'),  # Turkish mispronunciation (lowercase)
            (r'\bDuolingo\b', 'DUOL'),
            (r'\bDualingo\b', 'DUOL'),  # Transcription variation
            (r'\bMarvell\b', 'MRVL'),
            (r'\bMarvel\b', 'MRVL'),  # Transcription variation (missing 'l')
            (r'\bMicron\b', 'MU'),
            (r'\bIron\s+(?:Limited|Inc\.?)?\b', 'IRON'),  # Will be corrected to IREN
            (r'\bIren\b', 'IREN'),
            (r'\bIREN\b', 'IREN'),
            (r'\bDLocal\b', 'DLO'),
            (r'\bD\s+Local\b', 'DLO'),
            (r'\bD[- ]Local\b', 'DLO'),  # DLocal Limited (with hyphen or space)
            (r'\bCorewave\b', 'CRWV'),  # CoreWave Inc.
            (r'\bCoreweave\b', 'CRWV'),  # CoreWave Inc. (common misspelling)
            (r'\bCoreWave\b', 'CRWV'),  # CoreWave Inc.
            (r'\bCoreWave\s+Inc\.?\b', 'CRWV'),  # CoreWave Inc.
            # Manufacturing/Electronics
            (r'\bCelestica\b', 'CLS'),
            (r'\bCELESTICA\b', 'CLS'),  # Uppercase variation
            (r'\bcelestica\b', 'CLS'),  # Lowercase variation
            (r'\bselestika\b', 'CLS'),  # Turkish pronunciation
            (r'\bSelestika\b', 'CLS'),  # Turkish pronunciation (capitalized)
            (r'\bCelestica\'ya\b', 'CLS'),  # Turkish grammar variation (first mention)
            (r'\bCelestica\'da\b', 'CLS'),  # Turkish grammar variation
            (r'\bCelsius\s+(?:Holdings|Inc\.?)?\b', 'CELH'),
            (r'\bCelsius\b', 'CELH'),
            # Special case: "Celestica'ya da düşüş" after CLS discussion is actually Celsius (CELH)
            # This is a transcription error - "Celsius'ya" was transcribed as "Celestica'ya"
            # Pattern: After CLS discussion (lines 71-74), if "Celestica'ya da düşüş" appears with different support/resistance, it's CELH
            (r'\bCelestica\'ya\s+da\s+düşüş\b', 'CELH'),  # Transcription error: actually "Celsius'ya"
            # Additional context: if support 53-54 or resistance 66 is mentioned after "Celestica'ya", it's CELH not CLS
        ]
        
        for pattern, ticker_code in company_name_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if ticker_code in self.ticker_corrections:
                    ticker_code = self.ticker_corrections[ticker_code]
                tickers.add(ticker_code)
                self.logger.info(f"Found company name pattern '{pattern}' -> {ticker_code}")
        
        # Additional pass: Look for ticker-like words that might have been missed
        # This catches standalone capitalized words that appear in trading contexts
        # Pattern: Capitalized word (2-5 chars) followed by trading-related Turkish words
        trading_context_pattern = r'\b([A-Z][a-z]{1,4})\s+(?:bu|şu|bakalım|seviye|direnç|destek|hedef|bilanço|geri|yükseliş|düşüş)'
        additional_matches = re.findall(trading_context_pattern, text, re.IGNORECASE)
        for match in additional_matches:
            ticker = match.upper()
            # Apply ticker corrections BEFORE checking false positives
            if ticker in self.ticker_corrections:
                corrected_ticker = self.ticker_corrections[ticker]
                self.logger.info(f"Correcting ticker during extraction: {ticker} -> {corrected_ticker}")
                ticker = corrected_ticker
            if len(ticker) >= 2 and len(ticker) <= 5 and ticker not in false_positives:
                tickers.add(ticker)
        
        # Filter out index names that are often mistaken for tickers
        # Check if "SMP" appears in context of "SMP 500" or "S&P 500" - it's an index, not a ticker
        index_patterns = {
            'SMP': r'(?i)(?:SMP\s*500|S&P\s*500)',  # S&P 500 index
            'NDX': r'(?i)(?:NASDAQ|NDX)',  # NASDAQ index
            'SPX': r'(?i)(?:S&P|SPX)',  # S&P 500 index
            'RUT': r'(?i)(?:RUSSELL|RUT)',  # Russell 2000 index
            'VIX': r'(?i)(?:VIX|VOLATILITY)',  # VIX volatility index
        }
        
        # Remove index names that appear in index context
        filtered_tickers = []
        for ticker in tickers:
            if ticker in false_positives:
                continue
            
            # Check if this ticker is actually an index mentioned in context
            is_index = False
            if ticker in index_patterns:
                pattern = index_patterns[ticker]
                if re.search(pattern, text):
                    self.logger.info(f"Filtering out '{ticker}' - detected as index (pattern: {pattern})")
                    is_index = True
            
            # Special case: SMP is often S&P 500, not the ticker
            if ticker == 'SMP':
                # Check if it appears near "500" or "S&P"
                if re.search(r'(?i)SMP\s*500|S&P\s*500', text):
                    self.logger.info(f"Filtering out 'SMP' - detected as S&P 500 index")
                    is_index = True
            
            if not is_index:
                filtered_tickers.append(ticker)
        
        return filtered_tickers
    
    def extract_tickers_with_llm(self, transcript: str, model) -> set:
        """
        Use LLM to identify tickers from transcript context
        This is a generalizable fallback that catches tickers mentioned in unusual ways
        
        Args:
            transcript: Transcript text
            model: LLM model instance
            
        Returns:
            Set of additional ticker symbols found by LLM
        """
        try:
            prompt = f"""You are a financial data extraction specialist. Extract ALL stock ticker symbols mentioned in this Turkish trading video transcript.

TRANSCRIPT:
{transcript[:5000]}  # Limit to first 5000 chars to avoid token limits

INSTRUCTIONS:
1. Identify ALL stock ticker symbols (NASDAQ, NYSE, etc.) mentioned in the transcript
2. Include tickers mentioned by:
   - Company names (e.g., "Axon Enterprise" = AXON, "United Health" = UNH)
   - Direct ticker codes (e.g., "CRM", "SE", "NVDA")
   - Mispronunciations (e.g., "El ayliliğe" = LLY, "En misli" = NVDA)
3. Return ONLY a JSON array of ticker symbols, no explanations
4. Format: ["TICKER1", "TICKER2", ...]

Return ONLY valid JSON array with ticker symbols in uppercase."""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            import json
            llm_tickers = json.loads(response_text)
            
            # Convert to set and uppercase
            return set(t.upper().strip() for t in llm_tickers if isinstance(t, str) and len(t.strip()) >= 2)
            
        except Exception as e:
            self.logger.warning(f"LLM ticker extraction failed: {e}")
            return set()
    
    def format_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """
        Format ticker validation results for display
        
        Args:
            validation_results: Results from validate_tickers_in_analysis
            
        Returns:
            Formatted validation summary string
        """
        try:
            summary_lines = []
            
            # Overall statistics
            total_tickers = validation_results['tickers_found']
            valid_count = len(validation_results['valid_tickers'])
            invalid_count = len(validation_results['invalid_tickers'])
            
            summary_lines.append(f"**Toplam Ticker Sayısı**: {total_tickers}")
            summary_lines.append(f"**Geçerli Ticker'lar**: {valid_count}")
            summary_lines.append(f"**Geçersiz Ticker'lar**: {invalid_count}")
            
            if total_tickers > 0:
                success_rate = (valid_count / total_tickers) * 100
                summary_lines.append(f"**Başarı Oranı**: {success_rate:.1f}%")
            
            # Valid tickers
            if validation_results['valid_tickers']:
                summary_lines.append("\n### ✅ Geçerli Ticker'lar")
                for ticker_info in validation_results['valid_tickers']:
                    ticker = ticker_info['ticker']
                    company_name = ticker_info.get('company_name', 'Bilinmeyen')
                    summary_lines.append(f"- **{ticker}**: {company_name}")
            
            # Invalid tickers
            if validation_results['invalid_tickers']:
                summary_lines.append("\n### ❌ Geçersiz Ticker'lar")
                for ticker_info in validation_results['invalid_tickers']:
                    ticker = ticker_info['ticker']
                    error = ticker_info.get('error', 'Bilinmeyen hata')
                    summary_lines.append(f"- **{ticker}**: {error}")
            
            # Cache information
            cache_stats = self.ticker_validator.get_cache_stats()
            summary_lines.append(f"\n### 📊 Önbellek İstatistikleri")
            summary_lines.append(f"- **Önbellekteki Ticker Sayısı**: {cache_stats['total_cached']}")
            summary_lines.append(f"- **Geçerli Önbellek Girişleri**: {cache_stats['valid_entries']}")
            summary_lines.append(f"- **Süresi Dolmuş Girişler**: {cache_stats['expired_entries']}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to format validation summary: {e}")
            return f"Validation summary formatting failed: {e}"
    
    def setup_logging(self):
        """Setup logging configuration with organized log folder"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/nasdaq_trader_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - Log file: {log_file}")
    
    def optimize_system(self):
        """Optimize system for better performance"""
        try:
            # Set process priority to high
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            self.logger.info("System optimized for high performance")
        except Exception as e:
            self.logger.warning(f"Could not optimize system: {e}")
    
    def load_video_urls(self):
        """Load video URLs from various sources"""
        urls = []
        
        # Try environment variables first
        env_url = os.getenv('VIDEO_URL')
        env_urls = os.getenv('VIDEO_URLS')
        
        if env_url:
            urls.append(env_url)
        elif env_urls:
            urls.extend([url.strip() for url in env_urls.split(',') if url.strip()])
        
        # Fall back to video_list.txt
        if not urls and os.path.exists('video_list.txt'):
            try:
                with open('video_list.txt', 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                self.logger.error(f"Error reading video_list.txt: {e}")
        
        return urls
    
    def process_videos_parallel(self, video_urls):
        """Process videos in parallel for maximum performance"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.optimal_settings['max_workers']) as executor:
            # Submit all video processing tasks
            future_to_url = {
                executor.submit(self.process_single_video, url): url 
                for url in video_urls
            }
            
            # Collect results as they complete with rate limiting
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    # Add delay between completions to avoid API rate limits
                    if completed_count < len(video_urls):
                        self.logger.info(f"Completed {completed_count}/{len(video_urls)} videos. Waiting 15 seconds to avoid rate limits...")
                        time.sleep(15)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {url}: {e}")
                    results.append({
                        'url': url,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0
                    })
                    completed_count += 1
        
        return results
    
    def process_single_video(self, url):
        """Process a single video with all steps"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing: {url}")
            
            # Download video
            download_result = self.download_video(url)
            if isinstance(download_result, tuple) and len(download_result) == 2:
                audio_path, video_metadata = download_result
            else:
                audio_path = download_result
                video_metadata = {
                    'title': "Unknown Title",
                    'channel': "Unknown Channel",
                    'video_id': 'unknown',
                    'upload_date': None,
                    'view_count': None,
                    'like_count': None,
                    'duration': None,
                    'duration_seconds': None,
                    'description': None,
                    'url': url
                }
            
            if not audio_path:
                raise Exception("Failed to download video")
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                raise Exception("Failed to transcribe audio")
            
            # Generate AI analysis with full metadata
            analysis = self.generate_analysis(transcript, video_metadata)
            if not analysis:
                raise Exception("Failed to generate analysis")
            
            processing_time = time.time() - start_time
            
            return {
                'url': url,
                'success': True,
                'result': analysis,
                'metadata': video_metadata,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {url}: {e}")
            return {
                'url': url,
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def download_video(self, url):
        """Download video and extract audio with proper caching"""
        try:
            # Create output directory
            os.makedirs('video_cache', exist_ok=True)
            
            # Extract video ID from URL first
            video_id = self.extract_video_id(url)
            if not video_id:
                raise Exception("Could not extract video ID from URL")
            
            # Check for existing audio files (model-agnostic)
            import glob
            existing_files = []
            for ext in ['m4a', 'wav', 'mp3', 'webm']:
                pattern = f'video_cache/{video_id}_*.{ext}'
                existing_files.extend(glob.glob(pattern))
            
            if existing_files:
                # Use the most recent existing file
                existing_file = max(existing_files, key=os.path.getctime)
                self.logger.info(f"Using cached audio: {existing_file}")
                
                # Still fetch metadata from YouTube even for cached files
                try:
                    from yt_dlp import YoutubeDL
                    # Get YouTube authentication settings from config
                    auth_config = self.config.get('YOUTUBE_AUTHENTICATION', {})
                    enable_browser_cookies = auth_config.get('ENABLE_BROWSER_COOKIES', False)
                    preferred_browsers = auth_config.get('PREFERRED_BROWSERS', ['chrome', 'firefox', 'edge', 'safari'])
                    
                    # Configure yt-dlp for metadata extraction only (no download)
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                        'skip_download': True,  # Don't download, just extract metadata
                    }
                    
                    # Add cookie authentication if enabled
                    if enable_browser_cookies and os.path.exists('cookies.txt'):
                        ydl_opts['cookiefile'] = 'cookies.txt'
                    
                    with YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        
                        # Extract metadata
                        video_title = info.get('title', '')
                        channel_name = info.get('uploader', '')
                        if not video_title:
                            video_title = info.get('fulltitle', '') or info.get('alt_title', '')
                        if not channel_name:
                            channel_name = info.get('uploader_id', '') or info.get('channel', '') or info.get('channel_id', '')
                        
                        upload_date = info.get('upload_date', '')
                        if upload_date:
                            try:
                                from datetime import datetime as dt
                                upload_date_formatted = dt.strptime(upload_date, '%Y%m%d').strftime('%Y-%m-%d')
                            except:
                                upload_date_formatted = upload_date
                        else:
                            upload_date_formatted = None
                        
                        view_count = info.get('view_count', None)
                        like_count = info.get('like_count', None)
                        duration = info.get('duration', None)
                        description = info.get('description', '') or info.get('descriptions', '')
                        if isinstance(description, list):
                            description = '\n'.join(description) if description else ''
                        
                        # Format duration
                        duration_formatted = None
                        if duration:
                            try:
                                hours = int(duration // 3600)
                                minutes = int((duration % 3600) // 60)
                                seconds = int(duration % 60)
                                if hours > 0:
                                    duration_formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
                                else:
                                    duration_formatted = f"{minutes}:{seconds:02d}"
                            except:
                                duration_formatted = str(duration)
                        
                        video_metadata = {
                            'video_id': video_id,
                            'title': video_title or 'Unknown Title',
                            'channel': channel_name or 'Unknown Channel',
                            'upload_date': upload_date_formatted,
                            'view_count': view_count,
                            'like_count': like_count,
                            'duration': duration_formatted,
                            'duration_seconds': duration,
                            'description': description[:500] if description else None,
                            'url': url
                        }
                        
                        self.logger.info(f"Fetched metadata for cached video - Title: '{video_metadata['title']}', Channel: '{video_metadata['channel']}'")
                except Exception as e:
                    # If metadata fetch fails, use minimal metadata
                    self.logger.warning(f"Failed to fetch metadata for cached video: {e}")
                    video_metadata = {
                        'video_id': video_id,
                        'title': 'Unknown Title',
                        'channel': 'Unknown Channel',
                        'upload_date': None,
                        'view_count': None,
                        'like_count': None,
                        'duration': None,
                        'duration_seconds': None,
                        'description': None,
                        'url': url
                    }
                
                return existing_file, video_metadata
            
            # Only download if no cached file exists
            self.logger.info(f"Downloading new video: {video_id}")
            
            # Get current date for cache filename (audio files are model-agnostic)
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
            
            # Get YouTube authentication settings from config
            auth_config = self.config.get('YOUTUBE_AUTHENTICATION', {})
            enable_browser_cookies = auth_config.get('ENABLE_BROWSER_COOKIES', False)
            preferred_browsers = auth_config.get('PREFERRED_BROWSERS', ['chrome', 'firefox', 'edge', 'safari'])
            max_retries_per_browser = auth_config.get('MAX_RETRIES_PER_BROWSER', 3)
            fallback_to_no_auth = auth_config.get('FALLBACK_TO_NO_AUTH', True)
            
            # Configure yt-dlp for audio-only download with anti-403 measures
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': f'video_cache/%(id)s_{date_str}.%(ext)s',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
                # Anti-403 measures
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web'],  # Try Android client first, fallback to web
                        'player_skip': ['webpage', 'configs'],
                    }
                },
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                },
                'retries': 10,
                'fragment_retries': 10,
                'ignoreerrors': False,
                'no_check_certificate': False,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                    'preferredquality': '192',
                }]
            }
            
            # Add browser cookie authentication if enabled
            info = None
            if enable_browser_cookies:
                # Try manual cookie file first (cookies.txt)
                cookie_file = 'cookies.txt'
                if os.path.exists(cookie_file):
                    try:
                        self.logger.info(f"Attempting authentication with manual cookie file: {cookie_file}")
                        ydl_opts_with_auth = ydl_opts.copy()
                        ydl_opts_with_auth['cookiefile'] = cookie_file
                        
                        with YoutubeDL(ydl_opts_with_auth) as ydl:
                            info = ydl.extract_info(url, download=True)
                            self.logger.info(f"Successfully authenticated using cookie file: {cookie_file}")
                    except Exception as e:
                        self.logger.warning(f"Cookie file authentication failed: {str(e)[:100]}")
                
                # If cookie file didn't work, try browsers
                if info is None:
                    last_error = None
                    for browser in preferred_browsers:
                        try:
                            self.logger.info(f"Attempting authentication with {browser} browser cookies")
                            # Note: Close browser before running to avoid database lock issues
                            ydl_opts_with_auth = ydl_opts.copy()
                            ydl_opts_with_auth['cookiesfrombrowser'] = (browser,)
                            
                            with YoutubeDL(ydl_opts_with_auth) as ydl:
                                info = ydl.extract_info(url, download=True)
                                self.logger.info(f"Successfully authenticated using {browser} browser cookies")
                                break  # Success, exit the browser loop
                        except Exception as e:
                            last_error = e
                            error_msg = str(e)
                            # Provide helpful error messages
                            if "Could not copy" in error_msg or "cookie database" in error_msg:
                                self.logger.warning(f"Authentication failed with {browser}: Browser may be open. Please close {browser} and try again.")
                            elif "DPAPI" in error_msg or "decrypt" in error_msg:
                                self.logger.warning(f"Authentication failed with {browser}: Encryption issue. Try using cookies.txt file instead.")
                            elif "members" in error_msg or "Join this channel" in error_msg:
                                self.logger.warning(f"Authentication failed with {browser}: Cookies don't have membership access. Make sure you're logged in and a member.")
                            else:
                                self.logger.warning(f"Authentication failed with {browser}: {error_msg[:100]}")
                            continue  # Try next browser
                    
                    if info is None:
                        # All browsers failed
                        if fallback_to_no_auth:
                            self.logger.warning("All browser authentication attempts failed, trying without authentication with different extraction methods")
                            # Try different extraction methods as fallback
                            extraction_methods = [
                                {'player_client': ['android', 'web']},  # Try Android first
                                {'player_client': ['ios', 'web']},       # Try iOS
                                {'player_client': ['web']},              # Try web only
                                {'player_client': ['mweb', 'web']},     # Try mobile web
                            ]
                            
                            for method in extraction_methods:
                                try:
                                    self.logger.info(f"Trying extraction method: {method}")
                                    ydl_opts_fallback = ydl_opts.copy()
                                    ydl_opts_fallback['extractor_args'] = {
                                        'youtube': {
                                            **method,
                                            'player_skip': ['webpage', 'configs'],
                                        }
                                    }
                                    with YoutubeDL(ydl_opts_fallback) as ydl:
                                        info = ydl.extract_info(url, download=True)
                                        self.logger.info(f"Successfully downloaded using method: {method}")
                                        break
                                except Exception as e:
                                    self.logger.warning(f"Extraction method {method} failed: {str(e)[:100]}")
                                    continue
                            
                            if info is None:
                                raise Exception("All extraction methods failed. YouTube may be blocking requests. Try: 1) Update yt-dlp: pip install -U yt-dlp, 2) Export fresh cookies from browser, 3) Check if videos are accessible in browser")
                        else:
                            # Re-raise the last error if fallback is disabled
                            if last_error:
                                raise last_error
                            else:
                                raise Exception("All browser authentication attempts failed. Tip: Close your browser, ensure you're logged into YouTube as a member, or export cookies to cookies.txt file.")
            else:
                # No authentication configured, proceed normally with fallback methods
                try:
                    with YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "403" in error_msg or "forbidden" in error_msg:
                        self.logger.warning("Initial download failed with 403, trying alternative extraction methods")
                        # Try different extraction methods
                        extraction_methods = [
                            {'player_client': ['android', 'web']},
                            {'player_client': ['ios', 'web']},
                            {'player_client': ['web']},
                            {'player_client': ['mweb', 'web']},
                        ]
                        
                        for method in extraction_methods:
                            try:
                                self.logger.info(f"Trying extraction method: {method}")
                                ydl_opts_fallback = ydl_opts.copy()
                                ydl_opts_fallback['extractor_args'] = {
                                    'youtube': {
                                        **method,
                                        'player_skip': ['webpage', 'configs'],
                                    }
                                }
                                with YoutubeDL(ydl_opts_fallback) as ydl:
                                    info = ydl.extract_info(url, download=True)
                                    self.logger.info(f"Successfully downloaded using method: {method}")
                                    break
                            except Exception as e2:
                                self.logger.warning(f"Extraction method {method} failed: {str(e2)[:100]}")
                                continue
                        
                        if info is None:
                            raise Exception(f"All extraction methods failed. Original error: {e}. Try: 1) Update yt-dlp: pip install -U yt-dlp, 2) Export fresh cookies from browser")
                    else:
                        raise e
            
            # Extract comprehensive metadata (common for all authentication methods)
            downloaded_video_id = info.get('id', 'unknown')
            video_title = info.get('title', '')
            channel_name = info.get('uploader', '')
            
            # If title is empty or None, try alternative fields
            if not video_title:
                video_title = info.get('fulltitle', '') or info.get('alt_title', '')
            
            # If channel is empty or None, try alternative fields
            if not channel_name:
                channel_name = info.get('uploader_id', '') or info.get('channel', '') or info.get('channel_id', '')
            
            # Extract additional metadata
            upload_date = info.get('upload_date', '')
            if upload_date:
                # Format upload date: YYYYMMDD -> YYYY-MM-DD
                try:
                    from datetime import datetime as dt
                    upload_date_formatted = dt.strptime(upload_date, '%Y%m%d').strftime('%Y-%m-%d')
                except:
                    upload_date_formatted = upload_date
            else:
                upload_date_formatted = None
            
            view_count = info.get('view_count', None)
            like_count = info.get('like_count', None)
            duration = info.get('duration', None)  # Duration in seconds
            description = info.get('description', '') or info.get('descriptions', '')
            if isinstance(description, list):
                description = '\n'.join(description) if description else ''
            
            # Format duration as HH:MM:SS or MM:SS
            duration_formatted = None
            if duration:
                try:
                    hours = int(duration // 3600)
                    minutes = int((duration % 3600) // 60)
                    seconds = int(duration % 60)
                    if hours > 0:
                        duration_formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
                    else:
                        duration_formatted = f"{minutes}:{seconds:02d}"
                except:
                    duration_formatted = str(duration)
            
            # Build metadata dictionary
            video_metadata = {
                'video_id': downloaded_video_id,
                'title': video_title or 'Unknown Title',
                'channel': channel_name or 'Unknown Channel',
                'upload_date': upload_date_formatted,
                'view_count': view_count,
                'like_count': like_count,
                'duration': duration_formatted,
                'duration_seconds': duration,
                'description': description[:500] if description else None,  # Limit description length
                'url': url
            }
            
            # Final fallback for title and channel
            if not video_metadata['title'] or video_metadata['title'] == '':
                video_metadata['title'] = 'Unknown Title'
            if not video_metadata['channel'] or video_metadata['channel'] == '':
                video_metadata['channel'] = 'Unknown Channel'
            
            # Log the extracted metadata for debugging
            self.logger.info(f"Extracted metadata - Title: '{video_metadata['title']}', Channel: '{video_metadata['channel']}', "
                           f"Upload Date: '{video_metadata['upload_date']}', Views: {video_metadata['view_count']}, "
                           f"Duration: {video_metadata['duration']}")
            
            # Find the downloaded file
            for ext in ['m4a', 'wav', 'mp3', 'webm']:
                audio_path = f'video_cache/{downloaded_video_id}_{date_str}.{ext}'
                if os.path.exists(audio_path):
                    self.logger.info(f"Downloaded and cached: {audio_path}")
                    return audio_path, video_metadata
            
            raise Exception("Audio file not found after download")
                
        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            return None
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        try:
            import re
            # Handle various YouTube URL formats including live streams
            patterns = [
                r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
                r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
                r'youtube\.com\/live\/([a-zA-Z0-9_-]{11})',  # Live stream URLs
                r'youtube\.com\/live\/([a-zA-Z0-9_-]+)'      # Live stream URLs with different ID format
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
            return None
        except Exception as e:
            self.logger.error(f"Error extracting video ID: {e}")
            return None
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper with caching"""
        try:
            # Create transcript cache directory
            os.makedirs('transcript_cache', exist_ok=True)
            
            # Generate cache filename with date
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
            video_id = os.path.basename(audio_path).split('.')[0].split('_')[0]  # Remove date suffix
            
            # Get whisper model for cache filename
            whisper_model = self.config.get('MODEL_PERFORMANCE', {}).get('WHISPER_MODEL', 'small')
            transcript_cache_path = f'transcript_cache/{video_id}_{date_str}_{whisper_model}.txt'
            
            # Check if transcript is already cached
            if os.path.exists(transcript_cache_path):
                self.logger.info(f"Using cached transcript for {video_id} (model: {whisper_model})")
                with open(transcript_cache_path, 'r', encoding='utf-8') as f:
                    cached_text = f.read()
                # Check if cached transcript has timestamps (new format)
                if '[' in cached_text and ']' in cached_text and re.search(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', cached_text):
                    return cached_text
                # If cached transcript doesn't have timestamps, we need to re-transcribe with timestamps
                # But for now, return the cached version (will be updated on next transcription)
                self.logger.info(f"Cached transcript found but lacks timestamps - will add timestamps on next transcription")
                return cached_text
            
            # Transcribe if not cached
            self.logger.info(f"Transcribing audio: {audio_path} (model: {whisper_model})")
            model = whisper.load_model(whisper_model)
            result = model.transcribe(audio_path, language='tr')
            
            # Format transcript with timestamps for better ticker and timestamp extraction
            transcript_text = result['text']
            segments = result.get('segments', [])
            
            # Create timestamped transcript if segments are available
            if segments:
                timestamped_transcript = []
                for segment in segments:
                    start_time = segment.get('start', 0)
                    text = segment.get('text', '').strip()
                    if text:
                        # Format timestamp as MM:SS or HH:MM:SS
                        hours = int(start_time // 3600)
                        minutes = int((start_time % 3600) // 60)
                        seconds = int(start_time % 60)
                        if hours > 0:
                            timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            timestamp_str = f"{minutes:02d}:{seconds:02d}"
                        timestamped_transcript.append(f"[{timestamp_str}] {text}")
                
                # Use timestamped version for better analysis
                transcript_text = "\n".join(timestamped_transcript)
            
            # Cache the transcript
            with open(transcript_cache_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            self.logger.info(f"Transcript cached: {transcript_cache_path}")
            
            return transcript_text
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return None
    
    def generate_analysis(self, transcript, video_metadata=None):
        """Generate AI analysis using Gemini with improved two-step approach"""
        try:
            # Handle backward compatibility - if metadata is not provided, create default
            if video_metadata is None or not isinstance(video_metadata, dict):
                # Backward compatibility: if old signature is used
                if isinstance(video_metadata, str):
                    video_title = video_metadata
                    channel_name = "Unknown Channel"
                else:
                    video_title = "Unknown Title"
                    channel_name = "Unknown Channel"
                video_metadata = {
                    'title': video_title,
                    'channel': channel_name,
                    'video_id': 'unknown',
                    'upload_date': None,
                    'view_count': None,
                    'like_count': None,
                    'duration': None,
                    'duration_seconds': None,
                    'description': None,
                    'url': None
                }
            
            # Extract metadata fields with defaults
            video_title = video_metadata.get('title', 'Unknown Title')
            channel_name = video_metadata.get('channel', 'Unknown Channel')
            upload_date = video_metadata.get('upload_date')
            view_count = video_metadata.get('view_count')
            like_count = video_metadata.get('like_count')
            duration = video_metadata.get('duration')
            description = video_metadata.get('description')
            
            # Setup Gemini
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.config.get('MODELS', {}).get('gemini_model', 'gemini-2.5-flash'))
            
            # CRITICAL: Extract and validate tickers from transcript BEFORE generating analysis
            # This prevents hallucination by providing validated company names
            self.logger.info("Extracting and validating tickers from transcript...")
            extracted_tickers_list = self.extract_tickers_from_text(transcript)
            extracted_tickers = set(extracted_tickers_list)  # Convert to set for uniqueness
            self.logger.info(f"Found {len(extracted_tickers)} potential tickers in transcript: {extracted_tickers}")
            
            # Use LLM to identify additional tickers from context (comprehensive fallback)
            # This catches tickers that might be mispronounced or mentioned in unusual ways
            self.logger.info("Using LLM to identify tickers from context...")
            llm_tickers = self.extract_tickers_with_llm(transcript, model)
            if llm_tickers:
                self.logger.info(f"LLM found {len(llm_tickers)} additional tickers: {llm_tickers}")
                extracted_tickers.update(llm_tickers)
                self.logger.info(f"Total unique tickers after LLM extraction: {len(extracted_tickers)}")
            
            # Validate all extracted tickers and build validated ticker mapping
            validated_ticker_map = {}
            all_extracted_tickers_list = list(extracted_tickers) if extracted_tickers else []
            
            if extracted_tickers:
                # Use fuzzy matching for validation - this will auto-correct close matches
                fuzzy_corrections = {}  # Track fuzzy corrections
                for ticker in extracted_tickers:
                    # Try fuzzy matching if direct validation fails
                    # Disable fuzzy for OSCAR to prevent incorrect match to ASCAR
                    enable_fuzzy = True
                    if ticker == 'OSCAR' or ticker == 'OSCR':
                        enable_fuzzy = False  # OSCAR should validate directly, not fuzzy match
                    is_valid, company_name, error_msg, ticker_info, corrected_ticker = \
                        self.ticker_validator.validate_ticker_with_fuzzy(ticker, enable_fuzzy=enable_fuzzy)
                    
                    if is_valid:
                        # Use corrected ticker if fuzzy match found a different one
                        actual_ticker = corrected_ticker if corrected_ticker else ticker
                        if actual_ticker != ticker:
                            fuzzy_corrections[ticker] = actual_ticker
                            self.logger.info(f"Fuzzy correction: {ticker} -> {actual_ticker}")
                        
                        validated_ticker_map[actual_ticker] = company_name
                        self.logger.info(f"Validated: {actual_ticker} -> {company_name}")
                    else:
                        self.logger.warning(f"Invalid ticker found in transcript: {ticker}")
                        # Only add to list if it's not a common Turkish word
                        common_turkish_words = {
                            'BAKIN', 'BELKI', 'BIRAZ', 'BUNDA', 'DAHA', 'DOLAR', 'FIYAT', 'HATTA', 
                            'KADAR', 'OLAN', 'ONDA', 'ONUN', 'ORADA', 'UZUN', 'YANI', 'YINE', 
                            'ZAMAN', 'ZATEN', 'VAR', 'BIR', 'BU', 'VE', 'YA', 'DA', 'DE', 'KI', 'ILK', 'SON'
                        }
                        if ticker not in common_turkish_words:
                            all_extracted_tickers_list.append(ticker)
                
                # Merge fuzzy corrections into ticker_corrections for downstream use
                if fuzzy_corrections:
                    self.ticker_corrections.update(fuzzy_corrections)
                    self.logger.info(f"Added {len(fuzzy_corrections)} fuzzy corrections to ticker_corrections")
            
            # Build validated ticker reference string for prompt
            validated_ticker_reference = ""
            if validated_ticker_map:
                validated_ticker_reference = "\n\n**VALIDATED TICKER REFERENCE (MUST USE EXACTLY):**\n"
                validated_ticker_reference += "The following tickers have been validated with Yahoo Finance. You MUST use these exact company names:\n"
                for ticker, company_name in sorted(validated_ticker_map.items()):
                    validated_ticker_reference += f"- {ticker} = {company_name}\n"
                validated_ticker_reference += "\n**CRITICAL**: When mentioning any ticker in the report, you MUST use the exact company name from this list above. NEVER invent or guess company names.\n"
            
            # Add list of ALL extracted tickers (including unvalidated ones) so AI knows what to look for
            all_tickers_reference = ""
            if all_extracted_tickers_list:
                unique_tickers = sorted(set(all_extracted_tickers_list))
                all_tickers_reference = "\n\n**ALL TICKERS EXTRACTED FROM TRANSCRIPT:**\n"
                all_tickers_reference += "The following ticker symbols were extracted from the transcript:\n"
                for ticker in unique_tickers:
                    if ticker in validated_ticker_map:
                        all_tickers_reference += f"- {ticker} (✓ validated: {validated_ticker_map[ticker]}) - MUST include\n"
                    else:
                        # Check if it's a common Turkish word (false positive)
                        common_turkish_words = {
                            'BAKIN', 'BELKI', 'BIRAZ', 'BUNDA', 'DAHA', 'DOLAR', 'FIYAT', 'HATTA', 
                            'KADAR', 'OLAN', 'ONDA', 'ONUN', 'ORADA', 'UZUN', 'YANI', 'YINE', 
                            'ZAMAN', 'ZATEN', 'VAR', 'BIR', 'BU', 'VE', 'YA', 'DA', 'DE', 'KI', 'ILK', 'SON'
                        }
                        if ticker not in common_turkish_words:
                            all_tickers_reference += f"- {ticker} (not validated - check transcript for company name and context, include if it's a real company)\n"
                all_tickers_reference += "\n**CRITICAL INSTRUCTIONS:**\n"
                all_tickers_reference += "1. Include a ticker if it has technical analysis details (support, resistance, targets, sentiment, price levels, trading recommendations)\n"
                all_tickers_reference += "2. If a ticker is mentioned with technical analysis, it MUST be included - do not skip it\n"
                all_tickers_reference += "3. Do NOT include tickers that are only mentioned briefly without any trading analysis or context\n"
                all_tickers_reference += "4. For validated tickers (marked with ✓), check the transcript - if there's trading analysis, you MUST include it\n"
                all_tickers_reference += "5. For unvalidated tickers, include them if they have clear trading analysis (prices, support/resistance, targets, sentiment)\n"
                all_tickers_reference += "6. DO NOT include obvious Turkish common words (BAKIN, BELKI, BIRAZ, etc.)\n"
                all_tickers_reference += "7. IMPORTANT: \"En misli\" is how \"NBIS\" sounds in Turkish transcription - if mentioned with support/resistance, MUST include NBIS in report (NBIS is a valid ticker, NOT NVIDIA/NVDA)\n"
                all_tickers_reference += "8. IMPORTANT: \"Celestica'ya da düşüş\" at timestamp 4:54 is a transcription error - it's actually \"Celsius'ya\" (CELH) - look for different support/resistance levels (53-54 support, 66 resistance) which are different from CLS levels (297, 264, 370)\n"
                all_tickers_reference += "9. Quality over quantity - include ALL tickers with actionable trading information\n"
            
            # Add ticker corrections reference if configured
            ticker_corrections_reference = ""
            if self.ticker_corrections:
                ticker_corrections_reference = "\n\n**TICKER CORRECTIONS (AUTO-CORRECTED):**\n"
                ticker_corrections_reference += "The following ticker corrections are applied automatically. If you see these in the transcript, use the corrected ticker:\n"
                for incorrect, correct in sorted(self.ticker_corrections.items()):
                    ticker_corrections_reference += f"- {incorrect} → {correct}\n"
                ticker_corrections_reference += "\n**NOTE**: These corrections are applied automatically during processing.\n"
            
            # Add index reference to prevent false positives
            index_reference = """
**IMPORTANT INDEX NAMES (NOT TICKERS):**
The following are market indices, NOT individual stock tickers. When mentioned in transcript, use these exact names:
- "SMP 500" or "S&P 500" = S&P 500 Index (SPX) - NOT "Standard Motor Products, Inc."
- "NASDAQ" or "NDX" = NASDAQ 100 Index (NDX)
- "RUSSELL" or "RUT" = Russell 2000 Index (RUT)
- "VIX" = CBOE Volatility Index (VIX)
- "SPX" = S&P 500 Index (SPX)

**CRITICAL**: If transcript mentions "SMP 500", "S&P 500", or similar, it refers to the S&P 500 INDEX, NOT any individual stock ticker. Always use the full index name like "S&P 500 Index (SPX)" in the report.
"""
            validated_ticker_reference += ticker_corrections_reference + all_tickers_reference + index_reference
            
            # Generate content-based title if metadata extraction failed
            if video_title == "Unknown Title" or channel_name == "Unknown Channel":
                self.logger.info("Metadata extraction failed, generating content-based title...")
                title_prompt = f"""
                Based on this Turkish trading video transcript, generate a meaningful title and identify the speaker/channel name.
                
                TRANSCRIPT:
                {transcript}
                
                Please provide:
                1. A descriptive title for the video (in Turkish)
                2. The speaker/channel name (if mentioned in the transcript)
                
                Format your response as:
                TITLE: [Generated title]
                CHANNEL: [Speaker/channel name or "Unknown Speaker"]
                """
                
                try:
                    title_response = model.generate_content(title_prompt)
                    title_text = title_response.text
                    
                    # Parse the response
                    lines = title_text.strip().split('\n')
                    for line in lines:
                        if line.startswith('TITLE:'):
                            video_title = line.replace('TITLE:', '').strip()
                        elif line.startswith('CHANNEL:'):
                            channel_name = line.replace('CHANNEL:', '').strip()
                except Exception as e:
                    self.logger.warning(f"Failed to generate content-based title: {e}")
                    # Keep the original fallback values
            
            # Use proven prompt-based approach with enhanced anti-hallucination safeguards
            # This approach generates comprehensive, well-formatted reports directly
            self.logger.info("Using enhanced prompt-based report generation...")
            
            # Build metadata strings before prompt to avoid f-string nesting issues
            video_info_lines = []
            if upload_date:
                video_info_lines.append(f"- Upload Date: {upload_date}")
            if duration:
                video_info_lines.append(f"- Duration: {duration}")
            if view_count is not None:
                video_info_lines.append(f"- Views: {view_count:,}")
            if like_count is not None:
                video_info_lines.append(f"- Likes: {like_count:,}")
            if description:
                if len(description) > 200:
                    video_info_lines.append(f"- Description: {description[:200]}...")
                else:
                    video_info_lines.append(f"- Description: {description}")
            video_info_str = "\n            ".join(video_info_lines) if video_info_lines else ""
            
            report_info_lines = []
            if upload_date:
                report_info_lines.append(f"- **Video Upload Date**: {upload_date}")
            if duration:
                report_info_lines.append(f"- **Video Duration**: {duration}")
            if view_count is not None:
                report_info_lines.append(f"- **Views**: {view_count:,}")
            if like_count is not None:
                report_info_lines.append(f"- **Likes**: {like_count:,}")
            report_info_str = "\n            ".join(report_info_lines) if report_info_lines else ""
            
            summary_metadata_lines = []
            if upload_date:
                summary_metadata_lines.append(f"- **Video Upload Date**: {upload_date}")
            if duration:
                summary_metadata_lines.append(f"- **Video Duration**: {duration}")
            if view_count is not None:
                summary_metadata_lines.append(f"- **Views**: {view_count:,}")
            if like_count is not None:
                summary_metadata_lines.append(f"- **Likes**: {like_count:,}")
            summary_metadata_str = "\n            ".join(summary_metadata_lines) if summary_metadata_lines else ""
            
            # Create professional trading analysis prompt
            prompt = f"""
            As an experienced Nasdaq portfolio manager, analyze this trading video transcript and create a professional trading report in English.
            
            **CRITICAL TEMPLATE REQUIREMENT**: 
            - Use standard English section headers: "SHORT SUMMARY", "TRADING OPPORTUNITIES", "HIGH POTENTIAL TRADES"
            - Keep template structure in English (headers, labels, format)
            - Content can be in Turkish (analysis, reasoning, descriptions, notes)
            - Use English field labels: "Timestamp:", "Sentiment:", "Resistance:", "Support:", "Target:", "Notes:"
            - Use English section headers but Turkish content for analysis
            
            VIDEO INFORMATION:
            - Title: {video_title}
            - Channel: {channel_name}
            {video_info_str}
            
            {validated_ticker_reference}
            
            TRANSCRIPT:
            {transcript}
            
            Create a comprehensive trading analysis report in this EXACT format:
            
            **IMPORTANT LANGUAGE REQUIREMENTS:**
            - Generate ALL content in Turkish for Turkish day/swing traders
            - Use Turkish trading terminology and expressions
            - Keep the report concise and action-oriented
            - Focus on practical trading information
            
            **CONCISE REPORT REQUIREMENTS:**
            - Generate SHORT, ACTIONABLE reports (maximum 2-3 pages)
            - Focus on HIGH-IMPACT trading opportunities only
            - Eliminate redundant information and verbose explanations
            - Use bullet points and clear formatting
            - Prioritize immediate executable actions over analysis
            - Keep each section focused and concise
            - Use direct, actionable language
            - Focus on specific price levels and trading signals
            
            ## 📝 SHORT SUMMARY
            {summary_metadata_str + "\n            " if summary_metadata_str else ""}[Brief summary of video content - 2-3 sentences covering main message and trading opportunities]
            
            ## 📈 TRADING OPPORTUNITIES
            [CREATE SECTIONS FOR ALL TICKERS - THIS IS CRITICAL]
            
            **MANDATORY TICKER REQUIREMENTS:**
            1. Include a ticker if it has technical analysis details (support, resistance, targets, sentiment, price levels, trading context)
            2. If a ticker is mentioned with technical analysis (support/resistance levels, price targets, sentiment, trading recommendations), it MUST be included
            3. Do NOT include tickers that are only briefly mentioned without any trading analysis
            4. Each ticker section MUST include: Timestamp, Sentiment, Resistance, Support, Target, Notes
            5. If a ticker is mentioned but has NO technical details (no prices, no support/resistance, no targets, no sentiment), DO NOT include it
            6. The goal is QUALITY over quantity - include all tickers with actionable trading information
            7. IMPORTANT: "En misli" is how "NBIS" sounds in Turkish transcription - if mentioned with support/resistance, MUST include NBIS in report (NOT NVIDIA/NVDA)
            8. IMPORTANT: "Celestica'ya da düşüş" at timestamp 4:54 is a transcription error - it's actually "Celsius'ya" (CELH) - check for different support/resistance levels (53-54, 66) vs CLS levels (297, 264, 370)
            
            **CRITICAL INDEX VS TICKER DISTINCTION:**
            - If transcript mentions "SMP 500", "S&P 500", or "S&P" - this is the S&P 500 INDEX, NOT "Standard Motor Products, Inc."
            - Use format: "S&P 500 Index (SPX)" for indices
            - If transcript mentions "NASDAQ" or "NDX" - this is NASDAQ 100 Index (NDX), not a stock ticker
            - If transcript mentions "RUSSELL" or "RUT" - this is Russell 2000 Index (RUT), not a stock ticker
            - If transcript mentions "VIX" - this is CBOE Volatility Index (VIX), not a stock ticker
            - NEVER confuse index names with stock ticker symbols
            
            ### [TICKER] - [Company/Asset Name] ([TICKER_CODE])
            OR
            ### [Index Name] ([INDEX_CODE]) - [Market Indicator]
            - **Timestamp**: [EXACT time when ticker is first mentioned in video - example: 2:45, 5:23, 12:45, 1:30:15 - ONLY actual time from video]
            - **Sentiment**: [Bullish/Bearish/Neutral] - [Reasoning]
            - **Resistance**: [Resistance level if mentioned - leave blank if not]
            - **Support**: [Support level if mentioned - leave blank if not]
            - **Target**: [Target price if mentioned - leave blank if not]
            - **Notes**: [Important notes, technical analysis, risk factors, trading strategy]
            
            [REPEAT THIS SECTION FOR EVERY TICKER/ASSET MENTIONED IN TRANSCRIPT - NO TICKER CAN BE SKIPPED]
            
            ## 🎯 HIGH POTENTIAL TRADES
            [All high profit potential tickers and positions requiring risk management - no limit on number]
            
            **MANDATORY TEMPLATE REQUIREMENT FOR HIGH POTENTIAL TRADES**:
            - Section header MUST be "HIGH POTENTIAL TRADES" (never "YÜKSEK POTANSİYELLİ İŞLEMLER")
            - Use English field labels: "Entry:", "Stop:", "Target:", "Risk:", "Risk/Reward:"
            - Content can be in Turkish (reasoning, descriptions, explanations)
            - Use "Reason:" as label but Turkish content for reasoning
            
            **CRITICAL ENTRY/STOP/TARGET INFERENCE RULES:**
            1. If speaker says "buy at support $550" → Entry = $550.00
            2. If speaker says "buy on pullback to $550" → Entry = $550.00
            3. If Support is $550 and speaker recommends buying → Entry = $550.00 (use Support as Entry)
            4. If Entry exists but Stop missing → Calculate Stop = Entry * 0.98 (2% buffer below Entry)
            5. If Entry/Stop exist → Calculate Risk = abs(Entry - Stop) / Entry * 100 (format: "X.X%")
            6. If Entry/Stop/Target exist → Calculate Risk/Reward = abs(Target - Entry) / abs(Entry - Stop) (format: "1:X.X")
            7. ALWAYS include Entry, Stop, Target, Risk, and Risk/Reward for every trade
            8. Use EXACT prices from transcript (no ~ approximations, no ranges - use first value if range given)
            
            **1.** **[COMPANY_NAME] ([TICKER_CODE])**: [BUY/SELL/HOLD] - [Entry: **$X.XX**] [Stop: **$X.XX**] [Target: **$X.XX**] [Risk: **X.X%**] [Risk/Reward: **1:X.X**]
               *[Reason: En yüksek kar potansiyeli - acil fırsat]*
            
            **2.** **[COMPANY_NAME] ([TICKER_CODE])**: [BUY/SELL/HOLD] - [Entry: **$X.XX**] [Stop: **$X.XX**] [Target: **$X.XX**] [Risk: **X.X%**] [Risk/Reward: **1:X.X**]
               *[Reason: Yüksek kar potansiyeli - teknik kırılım]*
            
            **3.** **[COMPANY_NAME] ([TICKER_CODE])**: [TAKE PROFIT/EXIT] - [Current: **$X.XX**] [Take Profit: **$X.XX**] [Stop: **$X.XX**] [Timing: Immediate]
               *[Reason: Risk yönetimi - zarar kaçınma önceliği]*
            
            [CONTINUE FOR ALL HIGH POTENTIAL TICKERS - NO LIMIT ON NUMBER]
            
            **CRITICAL**: You MUST include at least 3-10 high potential trades. If a ticker has Support/Target and BUY sentiment, it MUST be included in HIGH POTENTIAL TRADES with complete Entry/Stop/Target/Risk/Risk-Reward.
            
            **CRITICAL FORMAT REQUIREMENT**: In HIGH POTENTIAL TRADES section, ALWAYS use format: **Company Name (TICKER_CODE)** - NEVER use just ticker codes without company names
            
            **MANDATORY TICKER REQUIREMENT**: 
            - EVERY entry in HIGH POTENTIAL TRADES MUST include both company name AND ticker code
            - Format: **1.** **Apple (AAPL)**: BUY - [Entry: **$150.00**] [Stop: **$140.00**] [Target: **$180.00**]
            - Format: **2.** **Tesla (TSLA)**: SELL - [Entry: **$200.00**] [Stop: **$220.00**] [Target: **$180.00**]
            - NEVER write just "1. BUY" or "1. Apple" - ALWAYS include ticker code in parentheses
            - NEVER use "Belirtilmemiş" or "Not Specified" - ALWAYS find the actual ticker code
            - If ticker code is unknown, research and provide the most likely ticker symbol
            
            **CRITICAL TIMESTAMP REQUIREMENT**: 
            - The transcript includes timestamps in format [MM:SS] or [HH:MM:SS] at the start of each segment
            - Extract the EXACT timestamp from the transcript when a ticker is first mentioned
            - If Axon is mentioned at [02:45] in the transcript, the timestamp must be 2:45
            - If Tesla is mentioned at [15:30] in the transcript, the timestamp must be 15:30
            - If Apple is mentioned at [1:25:45] in the transcript, the timestamp must be 1:25:45
            - NEVER guess or estimate timestamps - use ONLY the timestamp from the transcript brackets
            - Format: Use MM:SS for times under 1 hour (e.g., 2:45, 15:30), HH:MM:SS for longer videos (e.g., 1:25:45)
            - If transcript has [timestamp] format, extract the timestamp from the brackets when ticker appears
            
            
            **CRITICAL ANTI-HALLUCINATION REQUIREMENTS:**
            
        🚫 **STRICT PROHIBITIONS:**
        - NEVER add tickers, prices, or information not explicitly mentioned in the transcript
        - NEVER use external knowledge or current market data
        - NEVER assume or infer information not directly stated
        - NEVER add technical analysis not explicitly described in the video
        - NEVER include market news or events not mentioned in the transcript
        - NEVER assume or guess years, dates, or timeframes not explicitly mentioned
        - NEVER fill in missing date information (year, month, day) if not stated in transcript
        - NEVER add current date or time unless explicitly mentioned in video
        - NEVER assume video date or report date - use only what is explicitly stated
        - NEVER be creative or make assumptions about any dates or times
            
            ✅ **MANDATORY REQUIREMENTS:**
            1. ONLY include tickers and assets explicitly mentioned in the transcript
            2. ONLY include prices that are explicitly stated in the video
            3. ONLY include technical analysis that is explicitly described
            4. ONLY include trading ideas that are explicitly mentioned
            5. If information is not in the transcript, state "Not mentioned in video"
            6. Use exact quotes from the transcript when possible
            7. Clearly mark any assumptions or interpretations as "Based on transcript interpretation"
            8. Validate all ticker symbols (use standard format like AAPL, MSFT, etc.)
            9. If prices are mentioned, include them; if not, state "Price not specified in video"
            10. Be specific about entry/exit points only if explicitly mentioned
            11. Focus on actionable information that can be executed on NASDAQ
            12. Maintain professional trading report format
        13. **CRITICAL DATE HANDLING**: If only day/month is mentioned without year, write exactly as stated (e.g., "16 Eylül" not "16 Eylül 2024")
        14. **DATE ACCURACY**: Never assume years - if year is not mentioned, leave it empty or state "Year not specified in video"
        15. **EXACT TRANSCRIPT DATES**: Use only dates explicitly mentioned in the transcript, no assumptions
        16. **NO DATE CREATIVITY**: Never add current date, report date, or any date not explicitly mentioned in video
        17. **VIDEO DATE ONLY**: Use only the date explicitly mentioned in the video content, nothing else
            
            🎯 **CRITICAL TICKER ORGANIZATION REQUIREMENTS:**
            16. Each ticker/asset must appear ONLY ONCE in the entire report
            17. Create ONE comprehensive section per ticker with ALL information about that ticker
            18. Include ONE timestamp per ticker (the first or most relevant mention)
            19. Consolidate all information about each ticker into its dedicated section
            20. Do NOT repeat the same ticker in multiple sections
            21. Group all related information (prices, analysis, recommendations) under each ticker's section
            22. If a ticker is mentioned multiple times in the video, combine all information into ONE section
            23. Use the "Timestamp" field to show the most relevant timestamp for the ticker
            
            🔍 **SOURCE VERIFICATION:**
            - Every piece of information must be traceable to the transcript
            - Use phrases like "According to the video" or "The speaker mentioned"
            - If uncertain, state "Unclear from transcript" rather than guessing
            - Never fill in gaps with external knowledge
            
        📝 **REPORTING STANDARDS:**
        - NEVER use predicted values, estimates, or future dates (e.g., "06 Haziran 2024, 15:30 (Tahmini)")
        - NEVER write "Videoda belirtilmemiş" or any placeholder text
        - NEVER generate fake dates or add current date/time
        - NEVER add report date or any date not explicitly mentioned in video
        - If no trading ideas are mentioned, leave the section completely blank
        - If no tickers are mentioned, leave the section completely blank
        - If no prices are mentioned, leave the price fields completely blank
        - If information is not mentioned, leave the field completely empty
        - Always prioritize accuracy over completeness
        - Only include information that is explicitly mentioned in the video
            - Include exact timestamps when tickers/assets are mentioned (e.g., "5:23", "12:45")
            - **CRITICAL TIMESTAMP ACCURACY**: Extract the EXACT moment when each ticker is first mentioned in the video (e.g., if Axon is mentioned at 2:45, use 2:45)
            - **TIMESTAMP FORMAT**: Use MM:SS format for times under 1 hour (e.g., 2:45, 15:30), HH:MM:SS for longer videos (e.g., 1:15:30)
            - **TIMESTAMP EXTRACTION**: If transcript has [MM:SS] or [HH:MM:SS] timestamps, use the EXACT timestamp from brackets when ticker is mentioned
            - **TICKER CODE REQUIREMENT**: Always include the ticker symbol in parentheses (e.g., "Apple (AAPL)", "Tesla (TSLA)")
            - **TICKER DETECTION**: Pay special attention to tickers mentioned with Turkish suffixes (e.g., "IREN'e", "IREN'i" = IREN ticker)
            - **COMPREHENSIVE TICKER COVERAGE**: Ensure ALL tickers mentioned in transcript are included, even if mentioned with Turkish grammar (possessive, dative cases)
            - Use only current/past information from the video, no future predictions
            - CRITICAL: Use ONLY dates explicitly mentioned in the video transcript
            
            🚫 **ELIMINATE REPETITIONS:**
            - Each piece of information appears ONLY ONCE in the entire report
            - Do NOT repeat the same ticker in multiple sections
            - Do NOT repeat the same price information
            - Do NOT repeat the same technical analysis
            - Do NOT repeat the same risk assessment
            - Consolidate all information about each ticker into ONE section only
            
            🎯 **CRYSTAL CLEAR TRADING ACTIONS:**
            - Make trading decisions immediately obvious (BUY/SELL/HOLD)
            - Provide specific entry prices, stop losses, and targets
            - Use direct, actionable language
            - Focus on immediate execution (0-24 hours priority)
            - Eliminate ambiguity - be definitive in recommendations
            
            📊 **CONCISE REPORT GENERATION:**
            - Generate MAXIMUM 2-3 page reports
            - Start with SHORT SUMMARY (2-3 sentences)
            - Include ONLY tickers with technical analysis details (support, resistance, targets, sentiment, price levels)
            - End with HIGH POTENTIAL TRADES (only tickers with explicit BUY/SELL/HOLD recommendations)
            - Use bullet points and clear formatting
            - Eliminate verbose explanations
            - Focus on specific price levels and trading signals
            - Prioritize immediate executable actions
            - Use direct, actionable language
            - Keep each section focused and concise
            - **QUALITY FIRST**: Only include tickers with actionable trading information
            
            🔍 **REPORT STRUCTURE REQUIREMENTS:**
            - **SHORT SUMMARY**: 2-3 sentences maximum
            - **TRADING OPPORTUNITIES**: ONLY tickers with technical analysis details (support, resistance, targets, sentiment, price levels)
            - **HIGH POTENTIAL TRADES**: ALL high-potential tickers (no limit) - MUST include company name and ticker code for each entry
            - **Eliminate**: Redundant sections, verbose explanations, generic analysis
            - **Focus on**: Specific price levels, trading signals, immediate actions
            - **Format**: Bullet points, clear headers, concise language
            - **Length**: Maximum 2-3 pages total
            - **Priority**: Immediate actions first, analysis second
            - **CRITICAL**: Include EVERY ticker mentioned in the transcript
            
            📋 **SPECIFIC INFORMATION TO CAPTURE:**
            - **ALL TICKERS**: Every ticker mentioned in transcript must be covered
            - **NO EXCEPTIONS**: No ticker can be skipped or omitted
            - **COMPREHENSIVE COVERAGE**: Each ticker gets full analysis section
            - **TIMESTAMP EXTRACTION**: Find the EXACT video timestamp when each ticker is first mentioned (e.g., if Axon is mentioned at 2:45 in video, use 2:45)
            - **TIMESTAMP ACCURACY**: Each timestamp must reflect the actual moment the ticker appears in the video transcript
            - **TIMESTAMP SOURCE**: Use timestamps from [MM:SS] or [HH:MM:SS] brackets in transcript - these are exact video timestamps
            - **TICKER CODE FORMAT**: Always include ticker symbol in format "Company Name (TICKER)" 
            - **TICKER DETECTION**: Watch for tickers with Turkish grammar (e.g., "TICKER'e", "TICKER'i", "TICKER'ı", "TICKER'ın") - these all refer to the ticker symbol
            - **BOLD NUMBERS**: All prices, percentages, and numbers in HIGH POTENTIAL TRADES must be bold
            - **TICKER NAMES**: Every entry in HIGH POTENTIAL TRADES must show "Company Name (TICKER_CODE)" format
            - All exact price levels (e.g., "6500 support", "6800 resistance")
            - All moving average levels (8-day, 21-day, 50-day, 100-day, 200-day)
            - All gap levels (e.g., "Tesla 398 gap", "AMD 202-170 gap")
            - All breakout levels (e.g., "192 resistance", "kırılım olmadan pozisyon alma")
            - All volume signals (e.g., "sert hacim çubuğu", "hacimli düşüş")
            - All trend signals (e.g., "yükseliş trendi bozuldu", "kırmızı kanal")
            - All risk management rules (e.g., "mutlaka stop loss", "nakit oranı %30")
            - All timing signals (e.g., "Trump tweeti", "Fed konuşması")
            - All position management (e.g., "stopları yukarı çek", "pozisyon kapat")
            - All market events (e.g., "CPI verileri", "bilanço sezonu")
            
            🚫 **CRITICAL REQUIREMENT**: 
            - Include tickers ONLY if they have technical analysis details (support, resistance, targets, sentiment, price levels, trading recommendations)
            - Do NOT include tickers that are only briefly mentioned without trading context
            - Each included ticker must have its own dedicated section with meaningful analysis
            - If a ticker is mentioned multiple times, consolidate all information into one section
            - **HIGH POTENTIAL TRADES SECTION MUST INCLUDE TICKER NAMES**: Every numbered entry must show "Company Name (TICKER_CODE)" format
            - **QUALITY OVER QUANTITY**: Better to have fewer tickers with detailed analysis than many tickers with no useful information
            
            **FINAL TEMPLATE ENFORCEMENT**:
            - Use English template structure (headers, labels, format)
            - Use "HIGH POTENTIAL TRADES" as section header
            - Use English field labels: "Timestamp:", "Sentiment:", "Resistance:", "Support:", "Target:", "Notes:"
            - Content can be in Turkish (analysis, reasoning, descriptions)
            - Keep Turkish content for analysis but English template structure
            
            **CRITICAL TICKER NAME REQUIREMENT - NO HALLUCINATION**:
            - NEVER invent or guess company names - ONLY use validated names from the VALIDATED TICKER REFERENCE above
            - If a ticker appears in the VALIDATED TICKER REFERENCE, you MUST use the exact company name from that list
            - NEVER create fictional company names like "I-ON Digital Corp" or any other invented names
            - If a ticker is mentioned in transcript but NOT in the VALIDATED TICKER REFERENCE, use the ticker symbol only: "Unknown Company (TICKER)"
            - If company name is mentioned in transcript but ticker is unclear, use the ticker symbol only: "Unknown Company (TICKER)"
            - Format: **Company Name (TICKER)** - use validated company name from reference list
            - NEVER use "Belirtilmemiş" or "Not Specified" - if ticker exists in reference, use that exact name
            - STRICT RULE: When you see ticker "IREN" in transcript, you MUST use the company name from VALIDATED TICKER REFERENCE (e.g., "IREN Limited" or "Iris Energy Ltd.")
            - STRICT RULE: When you see any ticker symbol, FIRST check the VALIDATED TICKER REFERENCE - if it exists there, use that exact company name
            - STRICT RULE: If ticker is not in VALIDATED TICKER REFERENCE, use format: "Unknown Company (TICKER)" - NEVER invent a company name
            """
            
            # Generate content with rate limiting and retry logic
            max_retries = 3
            retry_delay = 60  # Start with 60 seconds delay
            
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    analysis_text = response.text
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "quota" in error_msg or "rate" in error_msg or "limit" in error_msg:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Gemini API rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise Exception(f"Gemini API rate limit exceeded after {max_retries} attempts: {e}")
                    else:
                        raise e
            
            # CRITICAL POST-PROCESSING: Fix index name false positives
            # Replace "Standard Motor Products, Inc. (SMP)" with "S&P 500 Index (SPX)" when context suggests it's the index
            index_fixes = [
                (r'Standard Motor Products, Inc\.\s*\(SMP\)', 'S&P 500 Index (SPX)'),
                (r'Standard Motor Products\s*\(SMP\)', 'S&P 500 Index (SPX)'),
                (r'Unknown Company\s*\(SMP\s*500\)', 'S&P 500 Index (SPX)'),
                (r'Unknown Company\s*\(S&P\s*500\)', 'S&P 500 Index (SPX)'),
            ]
            for pattern, replacement in index_fixes:
                if re.search(pattern, analysis_text, re.IGNORECASE):
                    self.logger.info(f"Fixing index false positive: replacing pattern '{pattern}' with '{replacement}'")
                    analysis_text = re.sub(pattern, replacement, analysis_text, flags=re.IGNORECASE)
            
            # CRITICAL POST-PROCESSING: Video-specific ticker corrections
            # Fix ASDR -> MSTR for specific video (pNAIYv2UO-U)
            # Also check URL patterns for more robust matching
            video_id = video_metadata.get('video_id', '') if video_metadata else ''
            video_url = video_metadata.get('url', '') if video_metadata else ''
            
            # Check both video_id and URL pattern
            is_target_video = (
                video_id == 'pNAIYv2UO-U' or 
                'pNAIYv2UO-U' in video_url or
                'watch?v=pNAIYv2UO-U' in video_url
            )
            
            if is_target_video:
                # Replace ASDR with MSTR in this specific video
                if re.search(r'\bASDR\b', analysis_text, re.IGNORECASE):
                    self.logger.info(f"Applying video-specific correction: ASDR -> MSTR for video {video_id or video_url}")
                    analysis_text = re.sub(r'\bASDR\b', 'MSTR', analysis_text, flags=re.IGNORECASE)
                    # Also fix in section headers and bold text
                    analysis_text = re.sub(r'###\s+([^*\n(]+?)\s*\(ASDR\)', r'### \1 (MSTR)', analysis_text, flags=re.IGNORECASE)
                    analysis_text = re.sub(r'\*\*([^*\n(]+?)\s*\(ASDR\)', r'**\1 (MSTR)', analysis_text, flags=re.IGNORECASE)
            
            # CRITICAL POST-PROCESSING: Fix incorrect ticker symbols
            # Replace incorrect tickers with correct ones based on correction mapping
            for incorrect_ticker, correct_ticker in self.ticker_corrections.items():
                # Get validated company name for correct ticker if available
                validated_company_name = validated_ticker_map.get(correct_ticker, None)
                
                # If we don't have a validated company name, try to validate the correct ticker now
                if not validated_company_name or validated_company_name == correct_ticker:
                    is_valid, company_name, _, _ = self.ticker_validator.validate_ticker(correct_ticker)
                    if is_valid and company_name and company_name != correct_ticker:
                        validated_company_name = company_name
                        # Add to validated_ticker_map for future use
                        validated_ticker_map[correct_ticker] = validated_company_name
                        self.logger.info(f"Validated corrected ticker {correct_ticker} -> {validated_company_name}")
                
                # Use validated company name if available, otherwise use the correct ticker
                final_company_name = validated_company_name if validated_company_name and validated_company_name != correct_ticker else correct_ticker
                
                # Pattern 0: Replace when incorrect ticker appears as company name "### INCORRECT_TICKER (CORRECT_TICKER)"
                # This handles cases like "### ASTR (ALAB)" -> "### Astera Labs Inc. (ALAB)"
                # Or "### ASDR (MSTR)" -> "### MicroStrategy Inc. (MSTR)"
                pattern0 = rf'###\s+{re.escape(incorrect_ticker)}\s*\(([^)]+)\)'
                replacement0 = rf'### {final_company_name} ({correct_ticker})'
                if re.search(pattern0, analysis_text, re.IGNORECASE):
                    self.logger.info(f"Correcting ticker in section header (company name): {incorrect_ticker} -> {correct_ticker} ({final_company_name})")
                    analysis_text = re.sub(pattern0, replacement0, analysis_text, flags=re.IGNORECASE | re.MULTILINE)
                
                # Pattern 0b: Replace when incorrect ticker appears as company name "### INCORRECT_TICKER (INCORRECT_TICKER)"
                # This handles cases like "### ASDR (ASDR)" -> "### MicroStrategy Inc. (MSTR)"
                pattern0b = rf'###\s+{re.escape(incorrect_ticker)}\s*\({re.escape(incorrect_ticker)}\)'
                replacement0b = rf'### {validated_company_name} ({correct_ticker})'
                if re.search(pattern0b, analysis_text, re.IGNORECASE):
                    self.logger.info(f"Correcting ticker in section header (both positions): {incorrect_ticker} -> {correct_ticker} ({validated_company_name})")
                    analysis_text = re.sub(pattern0b, replacement0b, analysis_text, flags=re.IGNORECASE | re.MULTILINE)
                
                # Pattern 1: Replace in section headers "### Company Name (INCORRECT_TICKER)"
                pattern1 = rf'###\s+([^*\n(]+?)\s*\({re.escape(incorrect_ticker)}\)'
                replacement1 = rf'### \1 ({correct_ticker})'
                if re.search(pattern1, analysis_text, re.IGNORECASE):
                    self.logger.info(f"Correcting ticker in section header: {incorrect_ticker} -> {correct_ticker}")
                    analysis_text = re.sub(pattern1, replacement1, analysis_text, flags=re.IGNORECASE | re.MULTILINE)
                
                # Pattern 2: Replace in bold "**Company Name (INCORRECT_TICKER)**"
                pattern2 = rf'\*\*([^*\n(]+?)\s*\({re.escape(incorrect_ticker)}\)\*\*'
                replacement2 = rf'**\1 ({correct_ticker})**'
                if re.search(pattern2, analysis_text, re.IGNORECASE):
                    self.logger.info(f"Correcting ticker in bold: {incorrect_ticker} -> {correct_ticker}")
                    analysis_text = re.sub(pattern2, replacement2, analysis_text, flags=re.IGNORECASE | re.MULTILINE)
                
                # Pattern 3: Replace standalone "(INCORRECT_TICKER)" references
                pattern3 = rf'\({re.escape(incorrect_ticker)}\)'
                replacement3 = f'({correct_ticker})'
                if re.search(pattern3, analysis_text, re.IGNORECASE):
                    self.logger.info(f"Correcting standalone ticker reference: {incorrect_ticker} -> {correct_ticker}")
                    analysis_text = re.sub(pattern3, replacement3, analysis_text, flags=re.IGNORECASE)
            
            # NOTE: We do NOT automatically add missing tickers anymore
            # Tickers are only included if they have technical analysis details (support, resistance, targets, sentiment, etc.)
            # A ticker being mentioned in transcript is necessary but NOT sufficient - it needs trading context
            
            # CRITICAL POST-PROCESSING: Replace any hallucinated company names with validated ones
            # This provides a safety net in case Gemini still makes mistakes
            if validated_ticker_map:
                self.logger.info("Post-processing: Replacing any incorrect company names with validated ones...")
                for ticker, validated_company_name in validated_ticker_map.items():
                    # Pattern 1: Match "Unknown Company (TICKER)" - MUST be replaced first
                    pattern1 = rf'Unknown Company\s*\({re.escape(ticker)}\)'
                    replacement1 = f'{validated_company_name} ({ticker})'
                    if re.search(pattern1, analysis_text, re.IGNORECASE):
                        self.logger.info(f"Replacing 'Unknown Company ({ticker})' with '{validated_company_name} ({ticker})'")
                        analysis_text = re.sub(pattern1, replacement1, analysis_text, flags=re.IGNORECASE)
                    
                    # Pattern 1a: Fix common company name issues for specific tickers
                    # Fix "Coreweave (CRWV)" -> "CoreWave Inc. (CRWV)"
                    if ticker == 'CRWV':
                        pattern_coreweave = r'Coreweave\s*\(CRWV\)'
                        if re.search(pattern_coreweave, analysis_text, re.IGNORECASE):
                            self.logger.info(f"Replacing 'Coreweave (CRWV)' with '{validated_company_name} (CRWV)'")
                            analysis_text = re.sub(pattern_coreweave, f'{validated_company_name} (CRWV)', analysis_text, flags=re.IGNORECASE)
                        # Also fix in section headers
                        pattern_coreweave_header = r'###\s+Coreweave\s*\(CRWV\)'
                        if re.search(pattern_coreweave_header, analysis_text, re.IGNORECASE):
                            self.logger.info(f"Replacing section header '### Coreweave (CRWV)' with '### {validated_company_name} (CRWV)'")
                            analysis_text = re.sub(pattern_coreweave_header, f'### {validated_company_name} (CRWV)', analysis_text, flags=re.IGNORECASE)
                    
                    # Pattern 1a2: Fix ACPLE -> AAPL (Apple Inc.)
                    if ticker == 'AAPL':
                        pattern_acple = r'ACPLE\s*\(ACPLE\)'
                        if re.search(pattern_acple, analysis_text, re.IGNORECASE):
                            self.logger.info(f"Replacing 'ACPLE (ACPLE)' with '{validated_company_name} (AAPL)'")
                            analysis_text = re.sub(pattern_acple, f'{validated_company_name} (AAPL)', analysis_text, flags=re.IGNORECASE)
                        # Also fix in section headers
                        pattern_acple_header = r'###\s+ACPLE\s*\(ACPLE\)'
                        if re.search(pattern_acple_header, analysis_text, re.IGNORECASE):
                            self.logger.info(f"Replacing section header '### ACPLE (ACPLE)' with '### {validated_company_name} (AAPL)'")
                            analysis_text = re.sub(pattern_acple_header, f'### {validated_company_name} (AAPL)', analysis_text, flags=re.IGNORECASE)
                    
                    # Pattern 1b: Match "Unknown Company (TICKER) (some text)" - handle cases with additional text in parentheses
                    pattern1b = rf'Unknown Company\s*\({re.escape(ticker)}\)\s*\([^)]+\)'
                    replacement1b = f'{validated_company_name} ({ticker})'
                    if re.search(pattern1b, analysis_text, re.IGNORECASE):
                        self.logger.info(f"Replacing 'Unknown Company ({ticker}) (...)' with '{validated_company_name} ({ticker})'")
                        analysis_text = re.sub(pattern1b, replacement1b, analysis_text, flags=re.IGNORECASE)
                    
                    # Pattern 2: Match "Company Name (TICKER)" in section headers (###)
                    pattern2 = rf'###\s+([^*\n(]+?)\s*\({re.escape(ticker)}\)'
                    def replace_section_header(match):
                        matched_name = match.group(1).strip()
                        if matched_name.lower() != validated_company_name.lower():
                            self.logger.info(f"Replacing section header '{matched_name} ({ticker})' with '{validated_company_name} ({ticker})'")
                            return f'### {validated_company_name} ({ticker})'
                        return match.group(0)
                    analysis_text = re.sub(pattern2, replace_section_header, analysis_text, flags=re.IGNORECASE | re.MULTILINE)
                    
                    # Pattern 3: Match "Company Name (TICKER)" in bold (**)
                    pattern3 = rf'\*\*([^*\n(]+?)\s*\({re.escape(ticker)}\)'
                    def replace_bold(match):
                        matched_name = match.group(1).strip()
                        if matched_name.lower() != validated_company_name.lower() and len(matched_name) < 100:
                            # Check if it's a reasonable company name (not a full sentence)
                            if not any(word in matched_name.lower() for word in ['timestamp', 'sentiment', 'resistance', 'support', 'target', 'notes', 'reason:', 'entry:', 'stop:', 'timing:', 'current:', 'take profit:', 'risk:', 'risk/reward:']):
                                self.logger.info(f"Replacing bold '{matched_name} ({ticker})' with '{validated_company_name} ({ticker})'")
                                return f'**{validated_company_name} ({ticker})'
                        return match.group(0)
                    analysis_text = re.sub(pattern3, replace_bold, analysis_text, flags=re.IGNORECASE | re.MULTILINE)
            
            # CRITICAL POST-PROCESSING: Consolidate duplicate tickers
            # Even if LLM follows instructions, enforce consolidation as safety net
            analysis_text = self._consolidate_duplicate_tickers(analysis_text)
            
            # CRITICAL POST-PROCESSING: Enhance HIGH POTENTIAL TRADES section
            # Calculate missing Stop, Risk, Risk/Reward from Entry/Stop/Target
            # Infer Entry from Support when missing
            # Ensure at least some trades are shown (never 0 trades)
            analysis_text = self._enhance_high_potential_trades(analysis_text)
            
            # Validate tickers in the generated analysis (for internal use only)
            validation_results = self.validate_tickers_in_analysis(analysis_text)
            
            # Return analysis without validation sections in the report
            return analysis_text
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return None
    
    def run_accelerated_pipeline(self):
        """Run the accelerated pipeline"""
        self.logger.info("Starting Accelerated Nasdaq Trader Pipeline")
        
        # Optimize system
        self.optimize_system()
        
        # Load video URLs
        video_urls = self.load_video_urls()
        if not video_urls:
            self.logger.error("No video URLs found")
            return
        
        self.logger.info(f"Found {len(video_urls)} videos to process")
        
        # Process videos in parallel
        results = self.process_videos_parallel(video_urls)
        
        # Save results
        self.save_results(results)
        
        self.logger.info("Accelerated pipeline complete!")
        return results
    
    def save_results(self, results):
        """Save processing results"""
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            self.logger.info(f"Saving {len(successful_results)} successful results...")
            
            for result in successful_results:
                try:
                    metadata = result.get('metadata', {})
                    self.save_report(result['result'], result['url'], metadata)
                    self.logger.info(f"Saved report for {result['url']}")
                except Exception as e:
                    self.logger.error(f"Failed to save report for {result['url']}: {e}")
        else:
            self.logger.warning("No successful results to save")

    def save_report(self, analysis, url, metadata=None):
        """Save analysis report to file with metadata"""
        try:
            # Create summary directory
            os.makedirs('summary', exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = metadata.get('video_id', 'unknown') if metadata else (url.split('v=')[-1].split('&')[0] if 'v=' in url else 'unknown')
            
            # Build metadata header for text report
            metadata_header = f"Video URL: {url}\n"
            metadata_header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            # Always show Video Title, Channel, and Upload Date
            video_title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
            channel = metadata.get('channel', 'Unknown') if metadata else 'Unknown'
            upload_date = metadata.get('upload_date', 'Unknown') if metadata else 'Unknown'
            metadata_header += f"Video Title: {video_title}\n"
            metadata_header += f"Channel: {channel}\n"
            metadata_header += f"Upload Date: {upload_date}\n"
            metadata_header += f"{'='*50}\n\n"
            
            # Save text report
            txt_filename = f'summary/report_{video_id}_{timestamp}.txt'
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(metadata_header)
                f.write(analysis)
            
            # Save JSON report with full metadata
            json_filename = f'summary/report_{video_id}_{timestamp}.json'
            report_data = {
                'url': url,
                'timestamp': timestamp,
                'analysis': analysis,
                'generated_at': datetime.now().isoformat(),
                'video_metadata': metadata if metadata else {},
                'ticker_validation': self.validate_tickers_in_analysis(analysis)
            }
            
            import json
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Save HTML report for mobile viewing
            html_filename = f'summary/report_{video_id}_{timestamp}.html'
            self.save_html_report(analysis, url, html_filename, metadata)
            
            print(f"Report saved: {txt_filename}")
            print(f"Mobile-friendly: {html_filename}")
            
        except Exception as e:
            print(f"Failed to save report: {e}")

    def save_html_report(self, analysis, url, filename, metadata=None):
        """Save HTML report for mobile viewing"""
        try:
            # Convert markdown-style analysis to HTML
            html_content = self.convert_analysis_to_html(analysis, url, metadata)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"Failed to save HTML report: {e}")
    
    def convert_analysis_to_html(self, analysis, url, metadata=None):
        """Convert analysis text to mobile-friendly HTML"""
        # Simple and robust HTML conversion
        html_content = self.format_analysis_html(analysis)
        
        # Build metadata section for header
        metadata_html = f"<p>Video: {url}</p>"
        if metadata:
            if metadata.get('title'):
                metadata_html += f"<p><strong>Title:</strong> {metadata.get('title')}</p>"
            if metadata.get('channel'):
                metadata_html += f"<p><strong>Channel:</strong> {metadata.get('channel')}</p>"
            if metadata.get('upload_date'):
                metadata_html += f"<p><strong>Upload Date:</strong> {metadata.get('upload_date')}</p>"
            if metadata.get('duration'):
                metadata_html += f"<p><strong>Duration:</strong> {metadata.get('duration')}</p>"
        metadata_html += f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        
        html_template = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NASDAQ DAY & SWING TRADE REPORT</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 15px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .header p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .content {{
            padding: 20px;
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
            margin-top: 25px;
            font-size: 20px;
        }}
        h3 {{
            color: #333;
            margin-top: 20px;
            font-size: 18px;
            background: #f8f9fa;
            padding: 10px;
            border-left: 4px solid #667eea;
        }}
        .ticker-section {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin: 8px 0;
        }}
        p {{
            margin: 10px 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            text-align: center;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }}
        @media (max-width: 600px) {{
            .container {{
                margin: 0;
                border-radius: 0;
            }}
            body {{
                padding: 5px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 NASDAQ DAY & SWING TRADE REPORT</h1>
            {metadata_html}
        </div>
        <div class="content">
            {html_content}
        </div>
        <div class="timestamp">
            Report generated by Nasdaq Trader AI
        </div>
    </div>
</body>
</html>"""
        return html_template
    
    def format_analysis_html(self, analysis):
        """Format the analysis text into HTML structure - Simple and robust approach"""
        lines = analysis.split('\n')
        html_parts = []
        in_list = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                continue
                
            # Handle headers
            if line.startswith('# '):
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                html_parts.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                html_parts.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                html_parts.append(f'<h3>{line[4:]}</h3>')
            # Handle list items
            elif line.startswith('- '):
                if not in_list:
                    html_parts.append('<ul>')
                    in_list = True
                # Clean up the line content
                content = line[2:].strip()
                # Handle bold text properly - replace **text** with <strong>text</strong>
                import re
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
                html_parts.append(f'<li>{content}</li>')
            # Handle numbered entries (HIGH POTENTIAL TRADES)
            elif line.startswith('**') and '**:' in line:
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                # Handle bold text in numbered entries
                import re
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                html_parts.append(f'<p>{content}</p>')
            # Handle reasoning lines (start with *)
            elif line.startswith('   *') or line.startswith('*'):
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                # Handle italic text for reasoning
                import re
                content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line.strip())
                html_parts.append(f'<p>{content}</p>')
            else:
                # Regular paragraph
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                if line and not line.startswith('[') and not line.startswith('**'):
                    # Handle bold text in paragraphs
                    import re
                    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                    html_parts.append(f'<p>{content}</p>')
        
        # Close any open list
        if in_list:
            html_parts.append('</ul>')
        
        return '\n'.join(html_parts)
    

# This file contains the AcceleratedNasdaqTrader class
# Use run_pipeline.py to execute the trading analysis pipeline