#!/usr/bin/env python3
"""
Restaurant Review Sentiment Analyzer - Main Runner Script
This script orchestrates the entire pipeline: data collection → analysis → dashboard

Usage:
    python run_analysis.py --help
    python run_analysis.py --quick-demo
    python run_analysis.py --collect-data --dashboard
    python run_analysis.py --yelp-key YOUR_KEY --location "Chicago, IL"
"""

import argparse
import sys
import os
import subprocess
import pandas as pd
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'wordcloud', 
        'nltk', 'scikit-learn', 'textblob', 'requests', 
        'beautifulsoup4', 'streamlit', 'plotly'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}. Please install manually.")
    
    print("✅ Package installation complete!")

def setup_nltk():
    """Download required NLTK data"""
    print("📚 Setting up NLTK data...")
    
    try:
        import nltk
        import ssl
        
        # Handle SSL issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        downloads = ['punkt', 'stopwords', 'vader_lexicon']
        for download in downloads:
            nltk.download(download, quiet=True)
        
        print("✅ NLTK setup complete!")
        return True
        
    except Exception as e:
        print(f"⚠️  NLTK setup failed: {e}")
        return False

def create_project_files():
    """Create the necessary Python files if they don't exist"""
    print("📁 Checking project files...")
    
    files_needed = [
        'restaurant_analyzer.py',
        'data_collector.py', 
        'dashboard.py'
    ]
    
    missing_files = []
    for file in files_needed:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("💡 Please save the provided code artifacts as these filenames:")
        for file in missing_files:
            print(f"   • {file}")
        return False
    
    print("✅ All project files found!")
    return True

def run_quick_demo():
    """Run a quick demonstration with sample data"""
    print("🚀 Running Quick Demo...")
    print("=" * 50)
    
    # Setup
    if not setup_nltk():
        print("⚠️  Continuing without NLTK setup...")
    
    try:
        # Import and run the main analyzer
        print("🔍 Running sentiment analysis with sample data...")
        
        # Check if we can import our modules
        try:
            from restaurant_analyzer import RestaurantReviewAnalyzer
            
            # Run the analysis
            analyzer = RestaurantReviewAnalyzer()
            results_df = analyzer.run_complete_analysis()
            
            if results_df is not None and not results_df.empty:
                print(f"✅ Analysis complete! Processed {len(results_df)} reviews")
                
                # Save results
                results_df.to_csv('demo_results.csv', index=False)
                print("💾 Results saved to 'demo_results.csv'")
                
                return True
            else:
                print("❌ Analysis failed to produce results")
                return False
                
        except ImportError as e:
            print(f"❌ Cannot import restaurant_analyzer: {e}")
            print("💡 Make sure 'restaurant_analyzer.py' exists in the current directory")
            return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def collect_data(yelp_key=None, google_key=None, location="New York, NY"):
    """Collect real restaurant data"""
    print("📊 Collecting restaurant data...")
    print(f"📍 Location: {location}")
    
    try:
        from data_collector import RestaurantDataCollector
        
        collector = RestaurantDataCollector()
        
        # Configuration
        config = {
            'yelp_api_key': yelp_key,
            'google_api_key': google_key,
            'location': location,
            'use_datasets': True,  # Always try to use datasets as backup
            'use_scraping': False,  # Disabled by default for safety
            'local_files': []
        }
        
        # Collect data
        df = collector.collect_all_sources(config)
        
        if df is not None and not df.empty:
            # Save collected data
            df.to_csv('collected_restaurant_reviews.csv', index=False)
            print(f"✅ Collected {len(df)} reviews!")
            print("💾 Data saved to 'collected_restaurant_reviews.csv'")
            return True
        else:
            print("❌ Data collection failed")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import data_collector: {e}")
        print("💡 Make sure 'data_collector.py' exists in the current directory")
        return False
    except Exception as e:
        print(f"❌ Data collection error: {e}")
        return False

def analyze_collected_data():
    """Run sentiment analysis on collected data"""
    print("🧠 Analyzing collected data...")
    
    try:
        from restaurant_analyzer import RestaurantReviewAnalyzer
        
        analyzer = RestaurantReviewAnalyzer()
        
        # Try to load collected data first
        if Path('collected_restaurant_reviews.csv').exists():
            results_df = analyzer.run_complete_analysis('collected_restaurant_reviews.csv')
        else:
            print("⚠️  No collected data found, using sample data...")
            results_df = analyzer.run_complete_analysis()
        
        if results_df is not None and not results_df.empty:
            # Save analysis results
            results_df.to_csv('analysis_results.csv', index=False)
            print(f"✅ Analysis complete! Results saved to 'analysis_results.csv'")
            return True
        else:
            print("❌ Analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🌐 Launching dashboard...")
    
    try:
        # Check if dashboard.py exists
        if not Path('dashboard.py').exists():
            print("❌ dashboard.py not found!")
            print("💡 Please save the dashboard code as 'dashboard.py'")
            return False
        
        # Launch Streamlit
        print("🚀 Starting Streamlit dashboard...")
        print("📱 Dashboard will open in your browser at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the dashboard")
        
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'])
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True
    except FileNotFoundError:
        print("❌ Streamlit not installed!")
        print("💡 Install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")
        return False

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="🍽️ Restaurant Review Sentiment Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --quick-demo                    # Run demo with sample data
  python run_analysis.py --dashboard                     # Launch dashboard only  
  python run_analysis.py --collect-data --analyze --dashboard  # Full pipeline
  python run_analysis.py --yelp-key YOUR_KEY --location "Chicago, IL" --dashboard
        """
    )
    
    parser.add_argument('--install', action='store_true',
                       help='Install required packages')
    
    parser.add_argument('--quick-demo', action='store_true',
                       help='Run quick demo with sample data')
    
    parser.add_argument('--collect-data', action='store_true',
                       help='Collect restaurant review data')
    
    parser.add_argument('--analyze', action='store_true',
                       help='Run sentiment analysis')
    
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch web dashboard')
    
    parser.add_argument('--yelp-key', type=str,
                       help='Yelp API key for data collection')
    
    parser.add_argument('--google-key', type=str,
                       help='Google Places API key')
    
    parser.add_argument('--location', type=str, default='New York, NY',
                       help='Location for restaurant search (default: New York, NY)')
    
    args = parser.parse_args()
    
    # Print header
    print("🍽️" + "=" * 58)
    print("🍽️  Restaurant Review Sentiment Analyzer")
    print("🍽️  Advanced NLP Pipeline for Business Intelligence") 
    print("🍽️" + "=" * 58)
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n💡 Quick start: python run_analysis.py --quick-demo")
        return
    
    success = True
    
    # Install packages if requested
    if args.install:
        install_requirements()
        setup_nltk()
    
    # Check project files
    if not create_project_files():
        print("\n❌ Setup incomplete. Please create the missing files first.")
        return
    
    # Quick demo mode
    if args.quick_demo:
        success = run_quick_demo()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("💡 Next steps:")
            print("   • python run_analysis.py --dashboard  (launch web interface)")
            print("   • python run_analysis.py --collect-data --analyze --dashboard  (full pipeline)")
        else:
            print("\n❌ Demo failed. Check the error messages above.")
        return
    
    # Data collection
    if args.collect_data:
        success = collect_data(args.yelp_key, args.google_key, args.location)
        if not success:
            print("⚠️  Continuing with available data...")
    
    # Analysis
    if args.analyze:
        success = analyze_collected_data()
        if not success:
            print("❌ Analysis failed, cannot proceed to dashboard")
            return
    
    # Dashboard
    if args.dashboard:
        if success:
            launch_dashboard()
        else:
            print("❌ Cannot launch dashboard due to previous errors")
    
    # Final message
    if success and not args.dashboard:
        print("\n🎉 All operations completed successfully!")
        print("💡 Launch the dashboard with: python run_analysis.py --dashboard")

if __name__ == "__main__":
    main()