from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tesla text to chunk
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

# Create the prompt
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

# Get AI response
print("🤖 Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

# Split the text at the markers
chunks = marked_text.split("<<<SPLIT>>>")

# Clean up the chunks (remove extra whitespace)
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:  # Only keep non-empty chunks
        clean_chunks.append(cleaned)

# Show results
print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()