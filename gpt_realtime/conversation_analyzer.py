import json
import os
from pathlib import Path
import datetime
import logging
from collections import defaultdict
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ConversationAnalyzer:
    """Analyzes conversation transcripts and sentiment data to generate reports."""
    
    def __init__(self, transcripts_dir: str = "gpt_realtime/gpt_transcripts"):
        self.transcripts_dir = Path(transcripts_dir)
        self.sentiment_file = "sentiment_results.json"
        
        # Ensure directories exist
        self.transcripts_dir.mkdir(exist_ok=True)
        
        # Initialize analysis results
        self.conversation_metrics = {
            "total_conversations": 0,
            "total_messages": 0,
            "average_message_length": 0,
            "sentiment_trends": defaultdict(list),
            "key_topics": defaultdict(int),
            "emotion_distribution": defaultdict(float)
        }
    
    def load_transcripts(self) -> List[Dict]:
        """Load all transcript files from the transcripts directory."""
        transcripts = []
        for file in self.transcripts_dir.glob("conversation_*.txt"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    transcripts.append({
                        "filename": file.name,
                        "content": content,
                        "timestamp": self._extract_timestamp(file.name)
                    })
            except Exception as e:
                logger.error(f"Error loading transcript {file}: {e}")
        return transcripts
    
    def load_sentiment_data(self) -> List[Dict]:
        """Load sentiment analysis results from JSON file."""
        try:
            if os.path.exists(self.sentiment_file):
                with open(self.sentiment_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
        return []
    
    def _extract_timestamp(self, filename: str) -> Optional[datetime.datetime]:
        """Extract timestamp from transcript filename."""
        try:
            # Expected format: conversation_YYYYMMDD_HHMMSS.txt
            match = re.search(r'conversation_(\d{8}_\d{6})', filename)
            if match:
                return datetime.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except Exception as e:
            logger.error(f"Error parsing timestamp from {filename}: {e}")
        return None
    
    def analyze_conversation(self, transcript: Dict) -> Dict:
        """Analyze a single conversation transcript."""
        analysis = {
            "message_count": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "average_user_message_length": 0,
            "average_assistant_message_length": 0,
            "key_topics": defaultdict(int),
            "sentiment_scores": []
        }
        
        # Split into messages
        messages = re.split(r'\n\n', transcript["content"])
        total_user_length = 0
        total_assistant_length = 0
        
        for message in messages:
            if message.startswith("User:"):
                analysis["user_messages"] += 1
                content = message[5:].strip()
                total_user_length += len(content)
                # Extract key topics (simple word frequency for now)
                words = re.findall(r'\w+', content.lower())
                for word in words:
                    if len(word) > 3:  # Ignore short words
                        analysis["key_topics"][word] += 1
            elif message.startswith("Assistant:"):
                analysis["assistant_messages"] += 1
                content = message[10:].strip()
                total_assistant_length += len(content)
        
        analysis["message_count"] = analysis["user_messages"] + analysis["assistant_messages"]
        if analysis["user_messages"] > 0:
            analysis["average_user_message_length"] = total_user_length / analysis["user_messages"]
        if analysis["assistant_messages"] > 0:
            analysis["average_assistant_message_length"] = total_assistant_length / analysis["assistant_messages"]
        
        return analysis
    
    def analyze_sentiment(self, sentiment_data: List[Dict]) -> Dict:
        """Analyze sentiment data over time."""
        analysis = {
            "emotion_trends": defaultdict(list),
            "average_scores": defaultdict(float),
            "peak_emotions": defaultdict(float)
        }
        
        for entry in sentiment_data:
            timestamp = entry.get("timestamp")
            emotion_scores = entry.get("emotion_scores", {})
            
            for emotion, score in emotion_scores.items():
                analysis["emotion_trends"][emotion].append({
                    "timestamp": timestamp,
                    "score": score
                })
                
                # Update average scores
                analysis["average_scores"][emotion] = (
                    (analysis["average_scores"][emotion] * (len(analysis["emotion_trends"][emotion]) - 1) + score) /
                    len(analysis["emotion_trends"][emotion])
                )
                
                # Track peak emotions
                if score > analysis["peak_emotions"][emotion]:
                    analysis["peak_emotions"][emotion] = score
        
        return analysis
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive report of all conversations and sentiment analysis."""
        transcripts = self.load_transcripts()
        sentiment_data = self.load_sentiment_data()
        
        report = {
            "summary": {
                "total_conversations": len(transcripts),
                "analysis_period": self._get_analysis_period(transcripts),
                "total_messages": 0,
                "average_conversation_length": 0
            },
            "conversation_analysis": [],
            "sentiment_analysis": {},
            "key_insights": [],
            "recommendations": []
        }
        
        # Analyze each conversation
        total_messages = 0
        for transcript in transcripts:
            analysis = self.analyze_conversation(transcript)
            report["conversation_analysis"].append({
                "filename": transcript["filename"],
                "timestamp": transcript["timestamp"].isoformat() if transcript["timestamp"] else None,
                "analysis": analysis
            })
            total_messages += analysis["message_count"]
        
        # Update summary metrics
        report["summary"]["total_messages"] = total_messages
        if len(transcripts) > 0:
            report["summary"]["average_conversation_length"] = total_messages / len(transcripts)
        
        # Analyze sentiment data
        if sentiment_data:
            sentiment_analysis = self.analyze_sentiment(sentiment_data)
            report["sentiment_analysis"] = sentiment_analysis
            
            # Generate insights
            report["key_insights"] = self._generate_insights(
                report["conversation_analysis"],
                sentiment_analysis
            )
            
            # Generate recommendations
            report["recommendations"] = self._generate_recommendations(
                report["conversation_analysis"],
                sentiment_analysis
            )
        
        return report
    
    def _get_analysis_period(self, transcripts: List[Dict]) -> Dict:
        """Calculate the time period covered by the transcripts."""
        timestamps = [t["timestamp"] for t in transcripts if t["timestamp"]]
        if not timestamps:
            return {"start": None, "end": None}
        
        return {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        }
    
    def _generate_insights(self, conversation_analysis: List[Dict], sentiment_analysis: Dict) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # Analyze conversation patterns
        avg_user_length = sum(a["analysis"]["average_user_message_length"] 
                            for a in conversation_analysis) / len(conversation_analysis)
        avg_assistant_length = sum(a["analysis"]["average_assistant_message_length"] 
                                 for a in conversation_analysis) / len(conversation_analysis)
        
        insights.append(f"Average user message length: {avg_user_length:.1f} characters")
        insights.append(f"Average assistant message length: {avg_assistant_length:.1f} characters")
        
        # Analyze sentiment trends
        for emotion, trend in sentiment_analysis["emotion_trends"].items():
            if trend:
                avg_score = sum(t["score"] for t in trend) / len(trend)
                insights.append(f"Average {emotion} score: {avg_score:.2f}")
        
        return insights
    
    def _generate_recommendations(self, conversation_analysis: List[Dict], 
                                sentiment_analysis: Dict) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        
        # Analyze emotion distribution
        emotion_scores = sentiment_analysis["average_scores"]
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        lowest_emotion = min(emotion_scores.items(), key=lambda x: x[1])[0]
        
        recommendations.append(f"Focus on enhancing {lowest_emotion.lower()} in responses")
        recommendations.append(f"Maintain current levels of {dominant_emotion.lower()} expression")
        
        # Analyze conversation patterns
        if any(a["analysis"]["average_user_message_length"] < 10 
              for a in conversation_analysis):
            recommendations.append("Consider encouraging more detailed user responses")
        
        return recommendations
    
    def save_report(self, report: Dict, output_file: str = "conversation_report.json"):
        """Save the generated report to a JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def generate_text_report(self, report: Dict) -> str:
        """Generate a human-readable text version of the report."""
        text_report = []
        
        # Summary section
        text_report.append("=== Conversation Analysis Report ===")
        text_report.append(f"\nPeriod: {report['summary']['analysis_period']['start']} to "
                         f"{report['summary']['analysis_period']['end']}")
        text_report.append(f"Total Conversations: {report['summary']['total_conversations']}")
        text_report.append(f"Total Messages: {report['summary']['total_messages']}")
        text_report.append(f"Average Messages per Conversation: "
                         f"{report['summary']['average_conversation_length']:.1f}")
        
        # Key Insights
        text_report.append("\n=== Key Insights ===")
        for insight in report["key_insights"]:
            text_report.append(f"- {insight}")
        
        # Recommendations
        text_report.append("\n=== Recommendations ===")
        for recommendation in report["recommendations"]:
            text_report.append(f"- {recommendation}")
        
        # Sentiment Analysis
        if report["sentiment_analysis"]:
            text_report.append("\n=== Sentiment Analysis ===")
            for emotion, avg_score in report["sentiment_analysis"]["average_scores"].items():
                text_report.append(f"{emotion}: {avg_score:.2f}")
        
        return "\n".join(text_report) 