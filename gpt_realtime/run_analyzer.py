import logging
from conversation_analyzer import ConversationAnalyzer

def setup_logging():
    """Configure logging for the analyzer."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup logging
    setup_logging()
    
    # Create analyzer instance
    print("Initializing Conversation Analyzer...")
    analyzer = ConversationAnalyzer()
    
    # Generate report
    print("\nGenerating analysis report...")
    report = analyzer.generate_report()
    
    # Save JSON report
    print("\nSaving JSON report...")
    analyzer.save_report(report, "conversation_analysis_report.json")
    
    # Generate and print text report
    print("\nGenerating text report...")
    text_report = analyzer.generate_text_report(report)
    
    print("\n=== Analysis Complete ===")
    print("\nText Report:")
    print(text_report)
    print("\nJSON report saved to: conversation_analysis_report.json")

if __name__ == "__main__":
    main() 