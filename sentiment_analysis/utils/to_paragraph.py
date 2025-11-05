import nltk
import re
from typing import List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def is_table_line(line: str) -> bool:
    """
    Detect if a line is likely part of a table.
    Tables often have:
    - Multiple whitespace separators
    - Pipe characters (|)
    - Lots of numbers
    - Consistent spacing patterns
    """
    # Check for pipe-delimited tables
    if '|' in line and line.count('|') >= 2:
        return True
    
    # Check for tab-delimited content
    if '\t' in line:
        return True
    
    # Check for multiple consecutive spaces (column alignment)
    if re.search(r'\s{3,}', line):
        return True
    
    # Check if line has high number density (common in tables)
    words = line.split()
    if len(words) >= 2:
        num_count = sum(1 for w in words if re.search(r'\d', w))
        if num_count / len(words) > 0.4:
            return True
    
    return False

def split_into_paragraphs(text: str, keep_tables_together: bool = True) -> List[str]:
    """
    Split text into paragraphs while keeping tables together.
    
    Args:
        text: Input text (financial report, etc.)
        keep_tables_together: If True, keeps table rows together as one chunk
    
    Returns:
        List of paragraph/table chunks
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    in_table = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines unless we're in a table
        if not stripped:
            if current_chunk and not in_table:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            elif in_table:
                # Empty line might be end of table
                # Check next non-empty line
                next_is_table = False
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip():
                        next_is_table = is_table_line(lines[j])
                        break
                
                if not next_is_table and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    in_table = False
            continue
        
        # Check if current line is part of a table
        is_table = is_table_line(line)
        
        if keep_tables_together:
            if is_table:
                if not in_table and current_chunk:
                    # Save previous paragraph and start table
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                in_table = True
                current_chunk.append(line)
            else:
                if in_table and current_chunk:
                    # End of table, save it
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    in_table = False
                current_chunk.append(line)
        else:
            current_chunk.append(line)
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return [chunk for chunk in chunks if chunk.strip()]

def split_paragraphs_with_nltk(text: str, keep_tables_together: bool = True) -> List[str]:
    """
    Enhanced splitting using NLTK's sentence tokenizer for non-table content.
    """
    # First, separate tables from regular text
    initial_chunks = split_into_paragraphs(text, keep_tables_together)
    
    final_chunks = []
    
    for chunk in initial_chunks:
        # Check if this chunk is a table
        lines = chunk.split('\n')
        table_line_count = sum(1 for line in lines if is_table_line(line))
        
        if table_line_count / len(lines) > 0.5:
            # This is mostly a table, keep it together
            final_chunks.append(chunk)
        else:
            # Regular text - can split by sentences if needed
            # For now, keep as paragraph but you can use nltk.sent_tokenize here
            sentences = nltk.sent_tokenize(chunk)
            
            # Group sentences into reasonable paragraphs (optional)
            if len(sentences) > 5:
                # Split long text into smaller paragraphs
                temp_para = []
                for sent in sentences:
                    temp_para.append(sent)
                    if len(' '.join(temp_para)) > 500:  # Adjust threshold as needed
                        final_chunks.append(' '.join(temp_para))
                        temp_para = []
                if temp_para:
                    final_chunks.append(' '.join(temp_para))
            else:
                final_chunks.append(chunk)
    
    return final_chunks

# Example usage
if __name__ == "__main__":
    sample_text = """
Financial Performance Report Q4 2024

Revenue Analysis

Our total revenue for Q4 2024 increased by 15% compared to the previous quarter.

Revenue Breakdown by Region:

Region          Q3 2024    Q4 2024    Growth
North America   $2.5M      $2.8M      12%
Europe          $1.8M      $2.1M      17%
Asia Pacific    $1.2M      $1.5M      25%

The strong performance in Asia Pacific was driven by increased market penetration.

Expense Summary

Operating expenses remained relatively stable:

Category        Amount     % of Revenue
Salaries        $1.2M      28%
Marketing       $0.8M      19%
R&D             $0.6M      14%

We expect continued growth in the next quarter based on current pipeline.
"""
    
    print("=" * 60)
    print("SPLITTING TEXT INTO PARAGRAPHS (TABLES PRESERVED)")
    print("=" * 60)
    
    chunks = split_paragraphs_with_nltk(sample_text)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
        print()







