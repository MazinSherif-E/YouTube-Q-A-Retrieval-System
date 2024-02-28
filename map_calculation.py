from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm.auto import tqdm
import torch

# Load dataset
ytt = load_dataset(
    "pinecone/yt-transcriptions",
    split="train",
    revision="926a45"
)

# Initialize SentenceTransformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')
retriever.to(device)
embed_dim = retriever.get_sentence_embedding_dimension()

# Initialize Pinecone index
pinecone.init(
    api_key="<<YOUR_API_KEY>>",
    environment="us-west1-gcp"
)
pinecone.create_index("youtube-search", dimension=embed_dim, metric="cosine")
index = pinecone.Index("youtube-search")

# Compute MAP
total_queries = 0
total_map = 0

for query_data in tqdm(ytt):
    query = query_data['text']
    ground_truth = query_data['relevant_documents']  # Assuming you have relevant documents for each query
    xq = retriever.encode([query]).tolist()
    xc = index.query(vector=xq, top_k=len(ground_truth), include_metadata=True)
    retrieved_documents = [context['metadata']['text'] for context in xc['results'][0]['matches']]
    
    # Compute Average Precision
    num_relevant = sum(1 for doc in retrieved_documents if doc in ground_truth)
    precision_at_k = [sum(1 for doc in retrieved_documents[:i+1] if doc in ground_truth) / (i + 1) for i in range(len(retrieved_documents))]
    average_precision = sum(precision_at_k[i] for i in range(len(retrieved_documents)) if retrieved_documents[i] in ground_truth) / num_relevant if num_relevant > 0 else 0
    
    total_map += average_precision
    total_queries += 1

# Compute MAP
map_score = total_map / total_queries
print("Mean Average Precision (MAP):", map_score)
