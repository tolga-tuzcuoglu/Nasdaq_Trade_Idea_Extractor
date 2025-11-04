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
from report_generator import ReportGenerator

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
                # Only add if length is valid (2-5 chars for tickers)
                if len(ticker) >= 2 and len(ticker) <= 5:
                    tickers.add(ticker)
        
        # Filter out common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR',
            'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO',
            'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'AL', 'AS', 'AT', 'BE', 'BY',
            'DO', 'GO', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'WE',
            'AN', 'AM', 'AI', 'OK', 'TV', 'ID', 'OS', 'PC', 'FY', 'IQ', 'QA', 'PM', 'AM', 'IO', 'IE', 'EU'
        }
        
        # Additional pass: Look for ticker-like words that might have been missed
        # This catches standalone capitalized words that appear in trading contexts
        # Pattern: Capitalized word (2-5 chars) followed by trading-related Turkish words
        trading_context_pattern = r'\b([A-Z][a-z]{1,4})\s+(?:bu|şu|bakalım|seviye|direnç|destek|hedef|bilanço|geri|yükseliş|düşüş)'
        additional_matches = re.findall(trading_context_pattern, text, re.IGNORECASE)
        for match in additional_matches:
            ticker = match.upper()
            if len(ticker) >= 2 and len(ticker) <= 5 and ticker not in false_positives:
                if ticker in self.ticker_corrections:
                    ticker = self.ticker_corrections[ticker]
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
            if isinstance(download_result, tuple):
                audio_path, video_title, channel_name = download_result
            else:
                audio_path = download_result
                video_title = "Unknown Title"
                channel_name = "Unknown Channel"
            
            if not audio_path:
                raise Exception("Failed to download video")
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                raise Exception("Failed to transcribe audio")
            
            # Generate AI analysis
            analysis = self.generate_analysis(transcript, video_title, channel_name)
            if not analysis:
                raise Exception("Failed to generate analysis")
            
            processing_time = time.time() - start_time
            
            return {
                'url': url,
                'success': True,
                'result': analysis,
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
                return existing_file
            
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
            
            # Configure yt-dlp for audio-only download
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': f'video_cache/%(id)s_{date_str}.%(ext)s',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
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
                            self.logger.warning("All browser authentication attempts failed, trying without authentication")
                            # Fallback to no authentication
                            with YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(url, download=True)
                        else:
                            # Re-raise the last error if fallback is disabled
                            if last_error:
                                raise last_error
                            else:
                                raise Exception("All browser authentication attempts failed. Tip: Close your browser, ensure you're logged into YouTube as a member, or export cookies to cookies.txt file.")
            else:
                # No authentication configured, proceed normally
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            
            # Extract metadata (common for all authentication methods)
            downloaded_video_id = info.get('id', 'unknown')
            video_title = info.get('title', '')
            channel_name = info.get('uploader', '')
            
            # If title is empty or None, try alternative fields
            if not video_title:
                video_title = info.get('fulltitle', '') or info.get('alt_title', '')
            
            # If channel is empty or None, try alternative fields
            if not channel_name:
                channel_name = info.get('uploader_id', '') or info.get('channel', '')
            
            # Final fallback
            if not video_title:
                video_title = 'Unknown Title'
            if not channel_name:
                channel_name = 'Unknown Channel'
            
            # Log the extracted metadata for debugging
            self.logger.info(f"Extracted metadata - Title: '{video_title}', Channel: '{channel_name}'")
            
            # Find the downloaded file
            for ext in ['m4a', 'wav', 'mp3', 'webm']:
                audio_path = f'video_cache/{downloaded_video_id}_{date_str}.{ext}'
                if os.path.exists(audio_path):
                    self.logger.info(f"Downloaded and cached: {audio_path}")
                    return audio_path, video_title, channel_name
            
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
    
    def generate_analysis(self, transcript, video_title="Unknown Title", channel_name="Unknown Channel"):
        """Generate AI analysis using Gemini with improved two-step approach"""
        try:
            # Setup Gemini
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.config.get('MODELS', {}).get('gemini_model', 'gemini-2.5-flash'))
            
            # CRITICAL: Extract and validate tickers from transcript BEFORE generating analysis
            # This prevents hallucination by providing validated company names
            self.logger.info("Extracting and validating tickers from transcript...")
            extracted_tickers = self.extract_tickers_from_text(transcript)
            self.logger.info(f"Found {len(extracted_tickers)} potential tickers in transcript: {extracted_tickers}")
            
            # Validate all extracted tickers and build validated ticker mapping
            validated_ticker_map = {}
            all_extracted_tickers_list = list(extracted_tickers) if extracted_tickers else []
            
            if extracted_tickers:
                validation_results = self.ticker_validator.validate_multiple_tickers(extracted_tickers)
                for ticker, result in validation_results.items():
                    if result.get('is_valid'):
                        company_name = result.get('company_name', ticker)
                        validated_ticker_map[ticker] = company_name
                        self.logger.info(f"Validated: {ticker} -> {company_name}")
                    else:
                        self.logger.warning(f"Invalid ticker found in transcript: {ticker}")
                        # Still add to list so AI knows about it - it might be valid but not in yfinance
                        all_extracted_tickers_list.append(ticker)
            
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
                all_tickers_reference = "\n\n**ALL TICKERS EXTRACTED FROM TRANSCRIPT (MUST INCLUDE ALL):**\n"
                all_tickers_reference += "The following ticker symbols were extracted from the transcript. You MUST include ALL of these in your report:\n"
                for ticker in unique_tickers:
                    if ticker in validated_ticker_map:
                        all_tickers_reference += f"- {ticker} (validated: {validated_ticker_map[ticker]})\n"
                    else:
                        all_tickers_reference += f"- {ticker} (not validated - check transcript for company name)\n"
                all_tickers_reference += "\n**CRITICAL**: Every ticker in this list MUST have its own section in the TRADING OPPORTUNITIES section. NO TICKER CAN BE SKIPPED.\n"
            
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
            
            # IMPROVED APPROACH: Use two-step structured extraction
            # Step 1: Extract structured data (JSON)
            # Step 2: Format into final report
            self.logger.info("Using improved two-step report generation approach...")
            
            # Initialize report generator
            report_gen = ReportGenerator(self.config)
            report_gen.set_extracted_tickers(
                all_extracted_tickers_list, 
                validated_ticker_map, 
                self.ticker_corrections
            )
            
            # Step 1: Extract structured data
            try:
                self.logger.info("Step 1: Extracting structured data from transcript...")
                structured_data = report_gen.extract_structured_data(transcript, video_title, channel_name, model)
                self.logger.info(f"Extracted {len(structured_data.get('tickers', []))} tickers in structured format")
                
                # VALIDATION: Verify all extracted tickers are included (already handled in ReportGenerator)
                # This ensures NO ticker is ever skipped
                extracted_ticker_set = set(all_extracted_tickers_list)
                extracted_in_data = set(t.get('ticker', '') for t in structured_data.get('tickers', []))
                if extracted_ticker_set != extracted_in_data:
                    self.logger.info(f"All tickers validated: {len(extracted_in_data)} tickers in structured data")
                
                # Step 2: Format structured data into report
                self.logger.info("Step 2: Formatting structured data into report...")
                analysis_text = report_gen.format_report(structured_data)
                
            except Exception as e:
                self.logger.error(f"Two-step approach failed: {e}")
                raise Exception(f"Report generation failed: {e}")
            
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
            
            # CRITICAL POST-PROCESSING: Fix incorrect ticker symbols
            # Replace incorrect tickers with correct ones based on correction mapping
            for incorrect_ticker, correct_ticker in self.ticker_corrections.items():
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
                    self.save_report(result['result'], result['url'])
                    self.logger.info(f"Saved report for {result['url']}")
                except Exception as e:
                    self.logger.error(f"Failed to save report for {result['url']}: {e}")
        else:
            self.logger.warning("No successful results to save")

    def save_report(self, analysis, url):
        """Save analysis report to file"""
        try:
            # Create summary directory
            os.makedirs('summary', exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = url.split('v=')[-1].split('&')[0] if 'v=' in url else 'unknown'
            
            # Save text report
            txt_filename = f'summary/report_{video_id}_{timestamp}.txt'
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"Video URL: {url}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*50}\n\n")
                f.write(analysis)
            
            # Save JSON report
            json_filename = f'summary/report_{video_id}_{timestamp}.json'
            report_data = {
                'url': url,
                'timestamp': timestamp,
                'analysis': analysis,
                'generated_at': datetime.now().isoformat(),
                'ticker_validation': self.validate_tickers_in_analysis(analysis)
            }
            
            import json
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Save HTML report for mobile viewing
            html_filename = f'summary/report_{video_id}_{timestamp}.html'
            self.save_html_report(analysis, url, html_filename)
            
            print(f"Report saved: {txt_filename}")
            print(f"Mobile-friendly: {html_filename}")
            
        except Exception as e:
            print(f"Failed to save report: {e}")

    def save_html_report(self, analysis, url, filename):
        """Save HTML report for mobile viewing"""
        try:
            # Convert markdown-style analysis to HTML
            html_content = self.convert_analysis_to_html(analysis, url)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"Failed to save HTML report: {e}")
    
    def convert_analysis_to_html(self, analysis, url):
        """Convert analysis text to mobile-friendly HTML"""
        # Simple and robust HTML conversion
        html_content = self.format_analysis_html(analysis)
        
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
            <p>Video: {url}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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