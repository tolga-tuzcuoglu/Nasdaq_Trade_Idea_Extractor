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
        
        print(f"Accelerated Nasdaq Trader Initialized")
        print(f"   System: {self.system_info['cpu_cores']} cores, {self.system_info['ram_gb']:.1f}GB RAM")
        print(f"   Optimal: {self.optimal_settings['parallel_videos']} parallel videos")
        print(f"   Ticker Validation: Enabled with caching")
    
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
        
        # Common ticker patterns - improved to catch more tickers like IREN
        ticker_patterns = [
            r'\b[A-Z]{2,5}\b',  # 2-5 uppercase letters (increased min to reduce false positives)
            r'\$[A-Z]{2,5}\b',  # $ followed by 2-5 uppercase letters
            r'\b[A-Z]{2,5}\.\w{1,2}\b',  # Ticker with exchange suffix (e.g., AAPL.NASDAQ)
            # Pattern for tickers mentioned with Turkish possessive suffix
            # Handles: "IREN'e", "IREN'i", "Iren'in", "İren'in" (case-insensitive for first letter)
            r'\b([A-Z][A-Za-z]{1,4})[\'’][a-zığüşöç]',  # Ticker with Turkish possessive suffix (handles "Iren'in" -> IREN)
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
                # Convert to uppercase for consistency (handles "Iren" -> "IREN")
                ticker = ticker.upper()
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
        
        # Filter out false positives and return list
        filtered_tickers = [ticker for ticker in tickers if ticker not in false_positives]
        
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
        """Generate AI analysis using Gemini"""
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
            if extracted_tickers:
                validation_results = self.ticker_validator.validate_multiple_tickers(extracted_tickers)
                for ticker, result in validation_results.items():
                    if result.get('is_valid'):
                        company_name = result.get('company_name', ticker)
                        validated_ticker_map[ticker] = company_name
                        self.logger.info(f"Validated: {ticker} -> {company_name}")
                    else:
                        self.logger.warning(f"Invalid ticker found in transcript: {ticker}")
            
            # Build validated ticker reference string for prompt
            validated_ticker_reference = ""
            if validated_ticker_map:
                validated_ticker_reference = "\n\n**VALIDATED TICKER REFERENCE (MUST USE EXACTLY):**\n"
                validated_ticker_reference += "The following tickers have been validated with Yahoo Finance. You MUST use these exact company names:\n"
                for ticker, company_name in sorted(validated_ticker_map.items()):
                    validated_ticker_reference += f"- {ticker} = {company_name}\n"
                validated_ticker_reference += "\n**CRITICAL**: When mentioning any ticker in the report, you MUST use the exact company name from this list above. NEVER invent or guess company names.\n"
            
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
            
            # Create professional trading analysis prompt
            prompt = f"""
            As an experienced Nasdaq portfolio manager, analyze this trading video transcript and create a professional trading report in English.
            
            **CRITICAL TEMPLATE REQUIREMENT**: 
            - Use standard English section headers: "REPORT INFORMATION", "SHORT SUMMARY", "TRADING OPPORTUNITIES", "HIGH POTENTIAL TRADES"
            - Keep template structure in English (headers, labels, format)
            - Content can be in Turkish (analysis, reasoning, descriptions, notes)
            - Use English field labels: "Timestamp:", "Sentiment:", "Resistance:", "Support:", "Target:", "Notes:"
            - Use English section headers but Turkish content for analysis
            
            VIDEO INFORMATION:
            - Title: {video_title}
            - Channel: {channel_name}
            
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
            
            ## 📊 REPORT INFORMATION
            - **Source**: {video_title} - {channel_name}
            - **Video Date**: [Date mentioned in video - ONLY use dates mentioned in video, add year if not specified]
            
            **ÖNEMLİ TARİH KURALI**: Eğer video sadece "16 Eylül" diyorsa, "16 Eylül" yazın. "16 Eylül 2024" YAZMAYIN çünkü yıl belirtilmemiş.
            
            ## 📝 SHORT SUMMARY
            [Brief summary of video content - 2-3 sentences covering main message and trading opportunities]
            
            ## 📈 TRADING OPPORTUNITIES
            [CREATE SECTIONS FOR ALL TICKERS MENTIONED IN TRANSCRIPT - NO TICKER CAN BE SKIPPED]
            
            ### [TICKER] - [Company/Asset Name] ([TICKER_CODE])
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
            
            **1.** **[COMPANY_NAME] ([TICKER_CODE])**: [BUY/SELL/HOLD] - [Entry: **$X.XX**] [Stop: **$X.XX**] [Target: **$X.XX**] [Risk: **X%**] [Risk/Reward: **1:X**]
               *[Reason: En yüksek kar potansiyeli - acil fırsat]*
            
            **2.** **[COMPANY_NAME] ([TICKER_CODE])**: [BUY/SELL/HOLD] - [Entry: **$X.XX**] [Stop: **$X.XX**] [Target: **$X.XX**] [Risk: **X%**] [Risk/Reward: **1:X**]
               *[Reason: Yüksek kar potansiyeli - teknik kırılım]*
            
            **3.** **[COMPANY_NAME] ([TICKER_CODE])**: [TAKE PROFIT/EXIT] - [Current: **$X.XX**] [Take Profit: **$X.XX**] [Stop: **$X.XX**] [Timing: Immediate]
               *[Reason: Risk yönetimi - zarar kaçınma önceliği]*
            
            [CONTINUE FOR ALL HIGH POTENTIAL TICKERS - NO LIMIT ON NUMBER]
            
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
            - Include ALL tickers mentioned in transcript (no exceptions)
            - End with HIGH POTENTIAL TRADES (ALL high-potential tickers, no limit)
            - Use bullet points and clear formatting
            - Eliminate verbose explanations
            - Focus on specific price levels and trading signals
            - Prioritize immediate executable actions
            - Use direct, actionable language
            - Keep each section focused and concise
            - **MANDATORY**: Every ticker in transcript must be covered
            
            🔍 **REPORT STRUCTURE REQUIREMENTS:**
            - **SHORT SUMMARY**: 2-3 sentences maximum
            - **TRADING OPPORTUNITIES**: ALL tickers mentioned in transcript (no limit)
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
            - **TICKER DETECTION**: Watch for tickers with Turkish grammar (IREN'e, IREN'i, IREN'ı, IREN'ın) - these all refer to ticker IREN
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
            - EVERY ticker mentioned in the transcript MUST be included in the report
            - NO ticker can be skipped, omitted, or excluded
            - Each ticker must have its own dedicated section
            - If a ticker is mentioned multiple times, consolidate all information into one section
            - **HIGH POTENTIAL TRADES SECTION MUST INCLUDE TICKER NAMES**: Every numbered entry must show "Company Name (TICKER_CODE)" format
            
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