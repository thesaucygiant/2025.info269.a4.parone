"""
Academic Paper Analyzer - LLM Application Demo (PDF Version)
Uses Claude API to analyze academic papers from PDF files

To get an API key:
1. Visit https://console.anthropic.com
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new key

Required packages:
pip install anthropic PyPDF2
"""

'''
1. Provide a citation and a link to the article (Use any citation standard but please make it complete).
2. What is the value that the application provides to users or an organization?
3. What is the AI task the main AI agent executes? Note the question is about the main task as it is possible
that an application implements more than one.
4. Does the application have a name? Is it emerging or deployed? Is there a description of the target users?
5. What is the main AI method it uses?
6. What would have been the original query in the minds of the system designers?
7. Where is the data or knowledge obtained to engineer the AI method?
8. Why was the AI method selected?
9. Was the application tested or evaluated?
'''

import anthropic
import json
import os
from PyPDF2 import PdfReader
from pathlib import Path

class PaperAnalyzer:
    def __init__(self, api_key):
        """Initialize the analyzer with Claude API"""
        self.client = anthropic.Anthropic(api_key=api_key)
        #used to create context for all prompts
        self.conversation_history = []
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from all pages
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            
            # Take first 100000 characters to stay within token limits
            # (Full papers are often too long for single API call)
            if len(text) > 100000:
                print(f"  Note: Paper is long ({len(text)} chars). Using first 100000 characters.")
                text = text[:100000]
            
            return text
        except Exception as e:
            print(f"  Error reading PDF: {e}")
            return None
    
    def analyze_paper(self, paper_text):
        """
        Main analysis function that coordinates multiple LLM tasks
        Returns structured analysis of the paper
        """
        # Task 1: Extract key metadata and structure
        #metadata = self._extract_metadata(paper_text)
        # Task 0
        answers = self._extract_answers(paper_text)
        
        # Task 2: Summarize methodology (uses previous context)
        #methodology = self._summarize_methodology(paper_text, metadata)
        
        # Task 3: Identify contributions (builds on previous analysis)
        #contributions = self._identify_contributions(paper_text, metadata, methodology)
        
        # Task 4: Suggest research directions
        #research_directions = self._suggest_research_directions(paper_text, metadata, methodology, contributions)
        
        return {
            #"metadata": metadata,
            "Answers": answers
            #"methodology": methodology,
            #"contributions": contributions,
            #"research_directions": research_directions
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
    
    def _extract_answers(self, paper_text):
        """Extract paper answers using LLM"""
        prompt = f"""
Request:
Analyze this academic paper excerpt and extract the answers to the questions noted below.  Each answer should be succint and include the question number before the answer in an ordered list.
1. Provide a citation and a link to the article (Use any citation standard but please make it complete).
2. What is the value that the application provides to users or an organization?
3. What is the AI task the main AI agent executes? Note the question is about the main task as it is possible
that an application implements more than one.
4. Does the application have a name? Is it emerging or deployed? Is there a description of the target users?
5. What is the main AI method it uses?
6. What would have been the original query in the minds of the system designers?
7. Where is the data or knowledge obtained to engineer the AI method?
8. Why was the AI method selected?
9. Was the application tested or evaluated?


Academic Paper Text:
{paper_text}
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        print(result)
        #self.conversation_history.append({
           # "task": "authors_extraction",
            #"result": result
        #})
        
        return result   
    
    
    
    
    def _summarize_methodology(self, paper_text, metadata):
        """Summarize research methodology"""
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
    
    def get_conversation_log(self):
        """Return the full conversation history for debugging"""
        return self.conversation_history
    
    def analyze_pdf_file(self, pdf_path):
        """Analyze a single PDF file"""
        print(f"\nAnalyzing: {pdf_path}")
        
        # Extract text from PDF
        paper_text = self.extract_text_from_pdf(pdf_path)
        
        if paper_text is None or len(paper_text.strip()) < 100:
            print(f"  Skipping - could not extract sufficient text")
            return None
        
        # Analyze the paper
        try:
            result = self.analyze_paper(paper_text)
            result['filename'] = os.path.basename(pdf_path)
            return result
        except Exception as e:
            print(f"  Error during analysis: {e}")
            return None


def analyze_pdf_directory(api_key, directory_path, output_file="analysis_results.json"):
    """
    Analyze all PDF files in a directory
    
    Args:
        api_key: Your Claude API key
        directory_path: Path to directory containing PDF files
        output_file: Where to save the results (JSON format)
    """
    # Find all PDF files
    pdf_files = list(Path(directory_path).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to analyze")
    
    results = []
    
    for pdf_path in pdf_files:
        # Create new analyzer for each paper (fresh conversation)
        analyzer = PaperAnalyzer(api_key)
        
        result = analyzer.analyze_pdf_file(str(pdf_path))
        
        if result:
            results.append(result)
            print(f"  ✓ Analysis complete")
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! {len(results)} papers analyzed.")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return results


def print_summary_report(results):
    """Print a summary of all analyzed papers"""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['filename']}")
        print("-" * 80)
        #print("METADATA:")
        #print(result['metadata'][:200] + "..." if len(result['metadata']) > 200 else result['metadata'])
        #print("\nTOP CONTRIBUTION:")
        # Extract just the first contribution
        #contrib_lines = result['contributions'].split('\n')
        #print(contrib_lines[0] if contrib_lines else result['contributions'][:200])
        print()


# Example usage
def main():
    # INSERT YOUR API KEY HERE
    API_KEY = "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"  # Replace with your actual key
    
    # Option 1: Analyze all PDFs in a directory
    print("="*80)
    print("PDF PAPER ANALYZER")
    print("="*80)
    
    # Change this to your directory containing PDF papers
    pdf_directory = "./papers"  # Put your PDFs in a folder called "papers"
    
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        print(f"\nDirectory '{pdf_directory}' not found.")
        print("Creating example directory structure...")
        os.makedirs(pdf_directory, exist_ok=True)
        print(f"\nPlease place your PDF papers in the '{pdf_directory}' folder and run again.")
        print("\nAlternatively, you can analyze a single PDF:")
        print("  analyzer = PaperAnalyzer(API_KEY)")
        print("  result = analyzer.analyze_pdf_file('path/to/your/paper.pdf')")
        return
    
    # Analyze all PDFs
    results = analyze_pdf_directory(
        api_key=API_KEY,
        directory_path=pdf_directory,
        output_file="paper_analysis_results.json"
    )
    
    # Print summary report
    if results:
        print_summary_report(results)
    
    # Option 2: Analyze a single PDF file
    # Uncomment below to analyze just one file
    """
    analyzer = PaperAnalyzer(API_KEY)
    single_result = analyzer.analyze_pdf_file("path/to/your/paper.pdf")
    
    if single_result:
        print("\n--- METADATA ---")
        print(single_result["metadata"])
        print("\n--- METHODOLOGY ---")
        print(single_result["methodology"])
        print("\n--- CONTRIBUTIONS ---")
        print(single_result["contributions"])
        print("\n--- RESEARCH DIRECTIONS ---")
        print(single_result["research_directions"])
    """


if __name__ == "__main__":
    main()