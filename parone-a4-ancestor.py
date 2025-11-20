"""
RAG system for querying ancestor PDFs using Claude AI.

Usage:
    python ancestor_rag_complete.py
Todo: Understand RAG implmentation - which system , what it does and what the numbers/readout means... chunks, etc
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import pickle

# External imports (install with: pip install anthropic pypdf numpy scikit-learn)
import anthropic
from pypdf import PdfReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# ANCESTOR RAG CLASS
# ============================================================================

class AncestorRAG:
    """
    A RAG system for ancestor research using only Anthropic's Claude API.
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            anthropic_api_key: Your Anthropic API key (or set ANTHROPIC_API_KEY env variable)
        """
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Please provide an Anthropic API key or set the ANTHROPIC_API_KEY environment variable.\n"
                "Get your key at: https://console.anthropic.com/"
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.documents = []  # Store document chunks with metadata
        self.embeddings = []  # Store embeddings for each chunk
        
        print("✓ Ancestor RAG system initialized (using Anthropic API only)")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text content from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
    
    def chunk_text(
        #what do these values mean?
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks (preserves context)
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            start += chunk_size - overlap
        
        return chunks
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create a simple embedding for text using TF-IDF style approach.
        This is a basic method that works without external APIs.
        """
        # Normalize text
        text = text.lower()
        words = text.split()
        
        # Create a simple embedding based on word frequencies
        embedding_dim = 512
        embedding = np.zeros(embedding_dim)
        
        # Use word hashing for position
        for word in words:
            # Hash each word to a position in the embedding
            hash_val = hash(word) % embedding_dim
            embedding[hash_val] += 1
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def add_pdf(
        self, 
        pdf_path: str, 
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ):
        """
        Add a PDF document to the RAG system.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Optional metadata (e.g., {"ancestor": "John Doe", "year": 1900})
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        if not os.path.exists(pdf_path):
            print(f"❌ Error: File not found: {pdf_path}")
            return
        
        print(f"Processing {pdf_path}...")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"❌ Warning: No text extracted from {pdf_path}")
            return
        
        # Split into chunks
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        if not chunks:
            print(f"❌ Warning: No chunks created from {pdf_path}")
            return
        
        # Create embeddings for each chunk
        print(f"   Creating embeddings for {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            embedding = self.create_embedding(chunk)
            
            # Prepare metadata
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "source": pdf_path,
                "filename": Path(pdf_path).name,
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
            
            # Store document and embedding
            self.documents.append({
                "text": chunk,
                "metadata": doc_metadata
            })
            self.embeddings.append(embedding)
        
        print(f"✓ Added {len(chunks)} chunks from {Path(pdf_path).name}\n")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for document chunks relevant to the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks with metadata and scores
        """
        if not self.documents:
            return []
        
        # Create embedding for the query
        query_embedding = self.create_embedding(query)
        
        # Calculate similarity between query and all documents
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]
        
        # Get indices of top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(similarities[idx])
            })
        
        return results
    
    def query(
        self, 
        question: str, 
        top_k: int = 5,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question about your ancestors using RAG with Claude.
        
        Args:
            question: Your question
            top_k: Number of document chunks to retrieve
            show_sources: Whether to include source information in response
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        if not self.documents:
            return {
                "answer": "No documents loaded. Please add PDF files using add_pdf() first.",
                "sources": []
            }
        
        # Retrieve relevant documents
        results = self.search(question, top_k=top_k)
        
        # Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(results):
            filename = result['metadata']['filename']
            chunk_text = result['text']
            context_parts.append(f"[Source {i+1}: {filename}]\n{chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for Claude
        prompt = f"""You are a helpful genealogy research assistant. Answer the following question based on the provided information about ancestors.

Context from ancestor documents:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the documents
- Include specific details like dates, places, and names when available
- If the documents don't contain enough information to answer fully, say so
- Be conversational and helpful
- Don't make up information that isn't in the documents

Answer:"""
        
        # Get answer from Claude
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = message.content[0].text
            
        except Exception as e:
            return {
                "answer": f"Error querying Claude: {e}",
                "sources": []
            }
        
        # Prepare response
        response = {"answer": answer}
        
        if show_sources:
            response["sources"] = [
                {
                    "file": r['metadata']['filename'],
                    "chunk": r['metadata']['chunk_id'],
                    "score": r['score']
                }
                for r in results
            ]
        
        return response
    
    def interactive_mode(self):
        """Start an interactive Q&A session."""
        print("\n" + "="*70)
        print("🔍 Ancestor Research Assistant - Interactive Mode")
        print("="*70)
        print(f"📚 Loaded {len(self.documents)} document chunks from your PDFs")
        print("💡 Ask questions about your ancestors (type 'quit' to exit)")
        print("="*70 + "\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q', 'bye']:
                    print("\n👋 Goodbye! Happy researching!\n")
                    break
                
                print("\n🔎 Searching and generating answer...\n")
                
                result = self.query(question)
                
                print(f"💬 Answer:\n{result['answer']}\n")
                
                if result.get('sources'):
                    print("📖 Sources used:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source['file']} (chunk {source['chunk']}, relevance: {source['score']:.1%})")
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def save_index(self, filepath: str = "ancestor_index.pkl"):
        """
        Save the document index to a file for later use.
        This avoids re-processing PDFs next time.
        """
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"✓ Index saved to {filepath}")
        print(f"  ({len(self.documents)} chunks saved)")
    
    def load_index(self, filepath: str = "ancestor_index.pkl"):
        """
        Load a previously saved index.
        Much faster than re-processing PDFs!
        """
        if not os.path.exists(filepath):
            print(f"❌ Error: Index file '{filepath}' not found")
            return False
        
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            
            print(f"✓ Index loaded from {filepath}")
            print(f"  ({len(self.documents)} chunks loaded)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def get_stats(self):
        """Print statistics about loaded documents."""
        if not self.documents:
            print("No documents loaded yet.")
            return
        
        # Count unique files
        files = set(doc['metadata']['filename'] for doc in self.documents)
        
        print("\n📊 Document Statistics:")
        print(f"   Total chunks: {len(self.documents)}")
        print(f"   Unique files: {len(files)}")
        print(f"   Files loaded:")
        for filename in sorted(files):
            count = sum(1 for doc in self.documents if doc['metadata']['filename'] == filename)
            print(f"      • {filename}: {count} chunks")
        print()


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def ask_claude_directly(question, api_key):
    """Ask Claude a question without any document context (no RAG)."""
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": question}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {e}"


def main():
    """
    Interactive demo showing RAG benefits.
    """
    print("\n" + "="*70)
    print("🌳 Ancestor RAG Demo - See the Difference!")
    print("="*70)
    print("\nThis demo shows how RAG improves answers using your documents.\n")
    
    # Step 1: Get API key
    print("="*70)
    print("Step 1: API Key")
    print("="*70)
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if api_key:
        print("✓ Found API key in environment")
    else:
        print("\nEnter your Anthropic API key:")
        print("(Get one at: https://console.anthropic.com/)")
        api_key = input("API Key: ").strip()
        
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return
        
        os.environ["ANTHROPIC_API_KEY"] = api_key
    
    print()
    
    # Step 2: Ask question BEFORE loading documents
    print("="*70)
    print("Step 2: Ask a Question (WITHOUT your documents)")
    print("="*70)
    print("\nAsk a question about one of your ancestors.")
    print("Example: 'Where was Giovanni Parone born?'\n")
    
    question = input("Your question: ").strip()
    
    if not question:
        question = "Where was Giovanni Parone born?"
        print(f"Using default: {question}")
    
    print(f"\n🤔 Asking Claude WITHOUT documents...\n")
    
    answer_without_rag = ask_claude_directly(question, api_key)
    
    print("💬 Answer WITHOUT documents:")
    print("-" * 70)
    print(answer_without_rag)
    print("-" * 70)
    
    input("\nPress Enter to continue...")
    
    # Step 3: Find PDFs
    print("\n" + "="*70)
    print("Step 3: Load Your PDF Documents")
    print("="*70)
    
    # Find all PDFs in current directory
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("\n❌ No PDF files found in current directory.")
        print("Please add PDF files and run again.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = os.path.getsize(pdf) / (1024 * 1024)
        print(f"  {i}. {pdf} ({size_mb:.2f} MB)")
    
    # Ask user which PDFs to load
    print("\nWhich PDFs to load?")
    print("  • Enter numbers (e.g., 1,2,3)")
    print("  • Enter 'all' for all files")
    
    choice = input("\nChoice: ").strip().lower()
    
    if choice == 'all':
        selected_pdfs = pdf_files
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            selected_pdfs = [pdf_files[i-1] for i in indices if 1 <= i <= len(pdf_files)]
        except:
            print("⚠️  Invalid choice. Loading all PDFs.")
            selected_pdfs = pdf_files
    
    if not selected_pdfs:
        print("❌ No PDFs selected.")
        return
    
    # Step 4: Process PDFs with RAG (with timing)
    print("\n" + "="*70)
    print("Step 4: Processing Documents (RAG)")
    print("="*70)
    
    print(f"\n⚙️  Loading {len(selected_pdfs)} PDF(s) into RAG system...")
    print("This will:")
    print("  1. Extract text from PDFs")
    print("  2. Split into chunks")
    print("  3. Create embeddings")
    print()
    
    rag = AncestorRAG(anthropic_api_key=api_key)
    
    start_time = time.time()
    
    for pdf in selected_pdfs:
        print(f"📄 Processing: {pdf}")
        rag.add_pdf(pdf)
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Processing complete!")
    print(f"⏱️  Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"📊 Total chunks: {len(rag.documents)}")
    
    input("\nPress Enter to continue...")
    
    # Step 5: Ask same question WITH RAG
    print("\n" + "="*70)
    print("Step 5: Ask the Same Question WITH embedded document)")
    print("="*70)
    
    print(f"\nOriginal question: \"{question}\"")
    change = input("Ask a different question? (yes/no): ").strip().lower()
    
    if change in ['yes', 'y']:
        new_q = input("New question: ").strip()
        if new_q:
            question = new_q
    
    print(f"\n🔎 Asking Claude WITH embedded document (RAG)...\n")
    
    result = rag.query(question, top_k=5)
    
    print("💬 Answer WITH documents:")
    print("-" * 70)
    print(result['answer'])
    print("-" * 70)
    
    if result.get('sources'):
        print("\n📚 Sources:")
        for i, s in enumerate(result['sources'], 1):
            print(f"  {i}. {s['file']} (relevance: {s['score']:.0%})")
    
    # Step 6: Side-by-side comparison
    print("\n\n" + "="*70)
    print("🔍 COMPARISON - Before vs After RAG")
    print("="*70)
    
    print(f"\n❓ Question: {question}\n")
    
    print("🔴 BEFORE (without documents):")
    print("-" * 70)
    print(answer_without_rag)
    print()
    
    print("🟢 AFTER (with RAG):")
    print("-" * 70)
    print(result['answer'])
    print()
    
    print("="*70)
    print("💡 Key Differences:")
    print("  • BEFORE: Based only on Claude's general knowledge")
    print("  • AFTER: Based on YOUR specific documents")
    print("  • RAG provides accurate, sourced, personalized answers!")
    print("="*70)
    
    # Continue asking questions?
    print("\n" + "="*70)
    print("Continue Exploring?")
    print("="*70)
    
    cont = input("\nAsk more questions? (yes/no): ").strip().lower()
    
    if cont in ['yes', 'y']:
        print()
        rag.interactive_mode()
    
    # Save index?
    print("\n" + "="*70)
    print("Save Your Work?")
    print("="*70)
    
    save = input("\nSave the index to skip processing next time? (yes/no): ").strip().lower()
    
    if save in ['yes', 'y']:
        filename = input("Filename (press Enter for 'ancestor_index.pkl'): ").strip()
        if not filename:
            filename = "ancestor_index.pkl"
        
        rag.save_index(filename)
        print(f"\n✓ Saved! Load next time with: rag.load_index('{filename}')")
    
    print("\n✅ Demo complete! Happy researching! 🌳\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()