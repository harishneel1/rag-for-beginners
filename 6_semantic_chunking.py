from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings  
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Same Tesla text but structured to show semantic grouping
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Semantic Chunker - groups by meaning, not structure
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=70
)

chunks = semantic_splitter.split_text(tesla_text)

print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()