"""
Academic Paper Analyzer - LLM Application Demo
Uses Claude API to analyze academic papers

To get an API key:
1. Visit https://console.anthropic.com
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new key
"""

import anthropic
import json

class PaperAnalyzer:
    def __init__(self, api_key):
        """Initialize the analyzer with Claude API"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
    
    def analyze_paper(self, paper_text):
        """
        Main analysis function that coordinates multiple LLM tasks
        Returns structured analysis of the paper
        """
        # Task 1: Extract key metadata and structure
        metadata = self._extract_metadata(paper_text)
        
        # Task 2: Summarize methodology (uses previous context)
        methodology = self._summarize_methodology(paper_text, metadata)
        
        # Task 3: Identify contributions (builds on previous analysis)
        contributions = self._identify_contributions(paper_text, metadata, methodology)
        
        # Task 4: Suggest research directions
        research_directions = self._suggest_research_directions(
            paper_text, metadata, methodology, contributions
        )
        
        return {
            "metadata": metadata,
            "methodology": methodology,
            "contributions": contributions,
            "research_directions": research_directions
        }
    
    def _extract_metadata(self, paper_text):
        """Extract paper metadata using LLM"""
        prompt = f"""Analyze this academic paper excerpt and extract key metadata.
        
Paper text:
{paper_text}

Please provide:
1. Main research question or problem
2. Key findings (bullet points)
3. Primary research domain/field

Format your response as JSON with keys: research_question, key_findings (list), domain"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        self.conversation_history.append({
            "task": "metadata_extraction",
            "result": result
        })
        
        return result
    
    def _summarize_methodology(self, paper_text, metadata):
        """Summarize research methodology"""
        # Include previous context so LLM understands what was already extracted
        prompt = f"""Based on this paper excerpt and the previously extracted metadata, 
summarize the research methodology in 2-3 sentences.

Paper text:
{paper_text}

Previously extracted metadata:
{metadata}

Focus on: research design, data collection methods, and analysis approach."""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        self.conversation_history.append({
            "task": "methodology_summary",
            "result": result
        })
        
        return result
    
    def _identify_contributions(self, paper_text, metadata, methodology):
        """Identify key research contributions"""
        prompt = f"""Based on this paper and previous analysis, identify the top 3 
contributions to the field.

Paper text:
{paper_text}

Metadata:
{metadata}

Methodology:
{methodology}

List the contributions clearly and explain why each is significant."""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=768,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        self.conversation_history.append({
            "task": "contribution_identification",
            "result": result
        })
        
        return result
    
    def _suggest_research_directions(self, paper_text, metadata, methodology, contributions):
        """Suggest future research directions"""
        prompt = f"""Based on this complete analysis, suggest 3 future research directions 
that could build on this work.

Paper text:
{paper_text}

Complete analysis so far:
- Metadata: {metadata}
- Methodology: {methodology}
- Contributions: {contributions}

For each direction, explain the potential value and feasibility."""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        self.conversation_history.append({
            "task": "research_directions",
            "result": result
        })
        
        return result
    
    def get_conversation_log(self):
        """Return the full conversation history for debugging"""
        return self.conversation_history


# Example usage
def main():
    # INSERT YOUR API KEY HERE
    API_KEY = "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"  # Replace with your actual key
    
    analyzer = PaperAnalyzer(API_KEY)
    
    # Example 1: Machine Learning Paper
    paper_example_1 = """
    Title: Attention Is All You Need
    
    Abstract: The dominant sequence transduction models are based on complex recurrent 
    or convolutional neural networks that include an encoder and a decoder. The best 
    performing models also connect the encoder and decoder through an attention mechanism. 
    We propose a new simple network architecture, the Transformer, based solely on 
    attention mechanisms, dispensing with recurrence and convolutions entirely.
    
    The Transformer allows for significantly more parallelization and can reach a new 
    state of the art in translation quality. We show that the Transformer generalizes 
    well to other tasks by applying it successfully to English constituency parsing.
    
    Methodology: We trained models on the WMT 2014 English-to-German and English-to-French 
    translation tasks using 8 NVIDIA P100 GPUs. Models were trained for 300,000 steps 
    using the Adam optimizer.
    """
    
    print("=" * 80)
    print("EXAMPLE 1: Analyzing Transformer Paper")
    print("=" * 80)
    
    result_1 = analyzer.analyze_paper(paper_example_1)
    
    print("\n--- METADATA ---")
    print(result_1["metadata"])
    print("\n--- METHODOLOGY ---")
    print(result_1["methodology"])
    print("\n--- CONTRIBUTIONS ---")
    print(result_1["contributions"])
    print("\n--- RESEARCH DIRECTIONS ---")
    print(result_1["research_directions"])
    
    # Example 2: Social Science Paper
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Analyzing Social Network Paper")
    print("=" * 80)
    
    analyzer2 = PaperAnalyzer(API_KEY)  # New instance for fresh conversation
    
    paper_example_2 = """
    Title: The Strength of Weak Ties
    
    Abstract: Analysis of social networks is suggested as a tool for linking micro and 
    macro levels of sociological theory. The procedure is illustrated by elaboration 
    of the macro implications of one aspect of small-scale interaction: the strength 
    of dyadic ties.
    
    It is argued that the degree of overlap of two individuals' friendship networks 
    varies directly with the strength of their tie to one another. The impact of this 
    principle on diffusion of influence and information, mobility opportunity, and 
    community organization is explored.
    
    Methodology: Data were collected through interviews with professional, technical, 
    and managerial workers who had changed jobs within the past year. Survey questions 
    focused on how they learned about their new position and the nature of their 
    relationship with the contact person.
    """
    
    result_2 = analyzer2.analyze_paper(paper_example_2)
    
    print("\n--- METADATA ---")
    print(result_2["metadata"])
    print("\n--- METHODOLOGY ---")
    print(result_2["methodology"])
    print("\n--- CONTRIBUTIONS ---")
    print(result_2["contributions"])
    print("\n--- RESEARCH DIRECTIONS ---")
    print(result_2["research_directions"])


if __name__ == "__main__":
    main()