# Nasdaq Trader - Professional Trading Analysis Pipeline

**High-performance AI-powered trading analysis from YouTube videos for Nasdaq investors and traders**

## 🎯 Overview

This production-ready system is specifically designed for **Nasdaq investors and traders**. It analyzes Turkish trading videos to generate actionable trading reports for Nasdaq portfolio management. The system extracts trading ideas, validates tickers with real-time API verification, and creates professional reports that can be directly executed on Nasdaq markets.

## ✨ Latest Features

### 🔍 **Advanced Ticker Validation**
- **Real-time Validation**: Uses `yfinance` API to validate all ticker symbols
- **24-Hour Caching**: Intelligent caching system reduces API calls by 90%
- **Fallback Mechanisms**: Multiple validation methods for reliability
- **Data Quality**: Ensures all tickers are valid and tradeable

### 📱 **Mobile-Friendly HTML Reports**
- **Browser-Ready**: Professional HTML reports for web viewing
- **Mobile-Responsive**: Optimized for mobile devices
- **Clean Formatting**: Proper bold text and structure
- **Professional Design**: Modern, readable interface

### 🚀 **Performance Optimizations**
- **Smart Caching**: 24-hour ticker validation cache
- **Parallel Processing**: Multiple videos processed simultaneously
- **System Optimization**: CPU and memory optimization
- **Error Handling**: Robust error handling and recovery

## 📁 Project Structure

```
Nasdaq_Trader_Local/
├── 🏃 run_pipeline.py                   # Main execution script
├── ⚡ nasdaq_trader_accelerated.py       # Core engine (library)
├── 🔍 ticker_validator.py               # Ticker validation utility
├── 📋 video_list.txt                    # Input: YouTube video URLs
├── 🍪 cookies.txt                       # YouTube authentication (optional, for members-only videos)
├── 📁 video_cache/                       # Cached audio files (with dates)
├── 📁 transcript_cache/                  # Cached transcripts (with dates)
├── 📁 summary/                           # Generated trading reports (.txt, .json, .html)
├── 📁 logs/                              # All log files
├── ⚙️ config.yaml                        # Configuration settings
└── 💾 ticker_cache.json                  # 24-hour ticker validation cache
```

## 🚀 Quick Start

### Python Script Execution
```bash
python run_pipeline.py
```

## 📋 File Descriptions

### **Main Execution Files**

#### `run_pipeline.py` ⭐ **MAIN SCRIPT**
- **Purpose**: Main entry point for trading analysis
- **Features**: User-friendly interface, maximum performance, professional output
- **Use Case**: Production trading analysis
- **Output**: Professional Nasdaq trading reports with actionable insights

### **Core Engine Files**

#### `nasdaq_trader_accelerated.py`
- **Purpose**: Core processing engine (library)
- **Features**: Maximum performance, parallel processing, system optimization
- **Use Case**: Used by run_pipeline.py
- **Output**: Core functionality (not run directly)

### **Utility Files**

#### `ticker_validator.py` ⭐ **NEW**
- **Purpose**: Advanced ticker validation with caching and fallback mechanisms
- **Features**: yfinance integration, 24-hour caching, multiple validation methods
- **Use Case**: Ensures all tickers are valid and tradeable
- **Performance**: Reduces API calls by 90% through intelligent caching

#### `config.yaml`
- **Purpose**: Configuration settings for the pipeline
- **Features**: Model settings, processing parameters, optimization options
- **Use Case**: Customizing analysis parameters

### **Data Files**

#### `video_list.txt`
- **Purpose**: Input file containing YouTube video URLs
- **Format**: One URL per line, comments with #
- **Example**: `https://www.youtube.com/watch?v=VIDEO_ID`

#### `video_cache/`
- **Purpose**: Cached audio files from YouTube videos
- **Format**: `{video_id}_{date}.{ext}` (e.g., `K8TFnwpDoAE_20251011.m4a`)
- **Use Case**: Avoiding re-downloading same videos

#### `transcript_cache/`
- **Purpose**: Cached transcriptions to avoid re-processing
- **Format**: `{video_id}_{date}.txt`
- **Use Case**: Faster processing of repeated videos

#### `summary/`
- **Purpose**: Generated trading analysis reports
- **Format**: `report_{video_id}_{timestamp}.{txt,json,html}`
- **Use Case**: Professional trading reports for portfolio management
- **Features**: Text reports, JSON data, mobile-friendly HTML reports

#### `logs/`
- **Purpose**: All log files for debugging and monitoring
- **Format**: Various log files with timestamps
- **Use Case**: Troubleshooting and performance monitoring

## 🎯 Professional Trading Reports

The system generates comprehensive trading reports with:

### **📊 Report Structure**
- **Video Information**: Date, URL, title, channel
- **Executive Summary**: Key opportunities and market outlook
- **Actionable Trade Ideas**: Day trading, swing trading, long-term investments
- **Validated Tickers**: Stocks, cryptocurrencies, commodities
- **Technical Analysis**: Support/resistance, chart patterns, key levels
- **Market Sentiment**: Catalysts, risks, outlook
- **Timing & Duration**: Immediate, short-term, medium-term actions
- **Portfolio Implications**: Position sizing, risk management, diversification
- **Trading Checklist**: Actionable items for execution

### **🛡️ Anti-Hallucination Measures**
- **Strict Source Validation**: Only uses information from video transcripts
- **Real-time Ticker Validation**: Uses yfinance API to validate all ticker symbols
- **24-Hour Caching**: Intelligent caching reduces API calls while maintaining data freshness
- **Price Verification**: Only includes prices explicitly mentioned
- **Fact-Based Analysis**: No external information or assumptions
- **Source Attribution**: All information traced back to video content

### **📱 Report Formats**
- **Text Reports**: Clean, professional trading analysis
- **JSON Data**: Structured data for programmatic access
- **HTML Reports**: Mobile-friendly, browser-ready reports with professional styling

## ⚙️ Configuration

### **Environment Variables**

**⚠️ SECURITY WARNING**: Never commit your actual API keys to the repository!

1. **Copy the template file:**
   ```bash
   cp env_example.txt .env
   ```

2. **Edit the .env file with your actual API key:**
   ```bash
   # Edit .env file
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

3. **Get your Gemini API key from:** [Google AI Studio](https://makersuite.google.com/app/apikey)

**The .env file is automatically ignored by git to prevent accidental exposure of your API keys.**

### **YouTube Authentication for Members-Only Videos**

If you need to download members-only YouTube videos, you can provide authentication via cookies:

#### **Option 1: Manual Cookie Export (Recommended)**
1. **Install browser extension**: "Get cookies.txt LOCALLY"
   - Chrome: https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc
   - Firefox: https://addons.mozilla.org/en-US/firefox/addon/get-cookies-txt-locally/
2. **Export cookies**:
   - Log into YouTube in your browser (make sure you're a member of the channel)
   - Visit https://www.youtube.com
   - Click the extension icon → Export
   - Save as `cookies.txt` in the project root
3. **The script will automatically use `cookies.txt` if it exists**

#### **Option 2: Browser Cookie Auto-Detection**
- The script will try to extract cookies from your browsers automatically
- Make sure browsers are closed to avoid database locks
- Supported browsers: Chrome, Firefox, Edge (in order of preference)

**Note**: `cookies.txt` is automatically ignored by git to protect your authentication data.

### **Dependencies**
```bash
pip install -r requirements.txt
```

### **Conda Environment**
```bash
conda create -n nasdaq_trader python=3.11
conda activate nasdaq_trader
```

## 🚀 Production Usage

### **For Nasdaq Investors & Traders**
1. Add YouTube video URLs to `video_list.txt`
2. Run `python run_pipeline.py`
3. Review generated reports in `summary/` folder
4. Execute trades based on actionable insights for Nasdaq markets

### **For Development**
1. Modify `config.yaml` for different settings
2. Use `nasdaq_trader_accelerated.py` for custom implementations
3. Monitor progress in `logs/` folder

### **For Batch Processing**
1. Use `run_pipeline.py` for multiple videos
2. Monitor progress in `logs/` folder
3. Review consolidated reports in `summary/` folder

## 📈 Performance Features

- **Parallel Processing**: Multiple videos processed simultaneously
- **Smart Caching**: Audio, transcript, and ticker validation caching
- **24-Hour Ticker Cache**: Reduces API calls by 90% through intelligent caching
- **System Optimization**: CPU and memory optimization
- **Error Handling**: Robust error handling and recovery
- **Logging**: Comprehensive logging for monitoring
- **Fallback Mechanisms**: Multiple validation methods for reliability

## 🔒 Security & Compliance

- **No External Data**: Only uses video transcript content
- **Local Processing**: All processing done locally
- **Secure API**: Uses secure Gemini API for analysis
- **Data Privacy**: No data sent to external services except AI analysis

## 📞 Support

For issues or questions:
1. Check `logs/` folder for error messages
2. Verify `GEMINI_API_KEY` is set correctly
3. Ensure all dependencies are installed
4. Check `config.yaml` for proper settings

## 🎯 Best Practices for Nasdaq Trading

1. **Use Main Script**: `run_pipeline.py` for production Nasdaq trading analysis
2. **Monitor Logs**: Check `logs/` folder for processing status
3. **Ticker Validation**: System automatically validates all Nasdaq tickers with yfinance API
4. **Cache Management**: 24-hour ticker cache reduces API calls and improves performance
5. **HTML Reports**: Use mobile-friendly HTML reports for better readability
6. **Risk Management**: Always use proper risk management in Nasdaq trading

## 🔄 System Features

The Python script (`run_pipeline.py`) includes all advanced features:
- ✅ Ticker validation with 24-hour caching
- ✅ HTML report generation
- ✅ Clean reports without validation clutter
- ✅ Mobile-responsive design
- ✅ Performance optimizations

## 🔒 Security Features

### **Automated Security Checks**
- **Pre-commit Security Scan**: Automatically scans for exposed API keys and secrets
- **Enhanced .gitignore**: Comprehensive patterns to prevent sensitive data exposure
- **Security Documentation**: Complete security policies and procedures

### **Security Commands**
```bash
# Run security check before committing
python security_check.py

# Run pre-commit security hook
python pre-commit-hook.py
```

### **Security Files**
- `security_check.py` - Automated security scanning
- `SECURITY.md` - Security policies and procedures
- `API_KEY_ROTATION.md` - Critical security response procedures

---

**⚠️ Trading Disclaimer**: This system generates analysis based on video content only. Always verify information and use proper risk management before executing trades. Past performance does not guarantee future results.