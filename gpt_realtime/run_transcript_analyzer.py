import logging
import argparse
from gpt_realtime.transcript_conversation_analyzer import ConversationAnalyzer

def setup_logging():
    """Configure logging for the analyzer."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup logging
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run transcript analysis')
    parser.add_argument('--username', help='Specific username to analyze')
    args = parser.parse_args()
    
    # Create analyzer instance
    print("Initializing Conversation Analyzer...")
    analyzer = ConversationAnalyzer(username=args.username)
    
    # Generate report
    print("\nGenerating analysis report...")
    report = analyzer.generate_report()
    
    # Create output filename based on username
    output_filename = f"conversation_analysis_report"
    if args.username:
        output_filename += f"_{args.username}"
    output_filename += ".json"
    
    # Save JSON report
    print(f"\nSaving JSON report to {output_filename}...")
    analyzer.save_report(report, output_filename)
    
    # Generate and print text report
    print("\nGenerating text report...")
    text_report = analyzer.generate_text_report(report)
    
    print("\n=== Analysis Complete ===")
    print("\nText Report:")
    print(text_report)
    print(f"\nJSON report saved to: {output_filename}")

if __name__ == "__main__":
    main() 
