# main_pipeline.py

import os
import re
import pinecone
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool

from speaker import extract_intro_text, extract_roles_with_gpt

# --- Load full transcript and chunk it ---
def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# --- Normalize speaker names for lookup ---
def normalize_roles_dict(raw_dict):
    return {
        re.sub(r"^MR\\.\\s*", "", name.strip(), flags=re.IGNORECASE).lower(): role.strip()
        for name, role in raw_dict.items()
    }

# --- Assign speaker/role metadata ---
def assign_speaker_metadata(chunks, roles_dict):
    roles_dict = normalize_roles_dict(roles_dict)
    for chunk in chunks:
        lines = chunk.page_content.splitlines()
        for line in lines:
            match = re.match(r"^([A-Z][a-z]+(?:\\s[A-Z][a-z]+)*)\\s*:", line)
            if match:
                name = match.group(1)
                name_key = name.lower()
                chunk.metadata["speaker"] = name
                chunk.metadata["role"] = roles_dict.get(name_key, "Unknown")
                break
    return chunks

# --- Embed and store in Pinecone ---
def embed_and_store(chunks, index_name="earnings-call-index"):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in [i.name for i in pc.list_indexes().indexes]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    vectors = []
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata or {}
        vectors.append({
            "id": f"doc-{i}",
            "values": embeddings[i],
            "metadata": {
                "text": chunk.page_content,
                "page": str(metadata.get("page", "")),
                "date": metadata.get("date", ""),
                "speaker": metadata.get("speaker", ""),
                "role": metadata.get("role", "")
            }
        })

    index.upsert(vectors=vectors, namespace="__default__")
    print(f"✅ Stored {len(vectors)} chunks in Pinecone index '{index_name}'.")

# --- RAG Query Tool ---
def rag_query(question: str) -> str:
    query_embedding = OpenAIEmbeddings(model="text-embedding-3-small").embed_query(question)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("earnings-call-index")

    results = index.query(
        namespace="__default__",
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    hits = results.get("matches", [])
    if not hits or max(hit["score"] for hit in hits) < 0.2:
        return "I don't have information related to your question in the transcript."

    context_chunks = []
    citations = []

    for match in hits:
        text = match["metadata"].get("text", "")[:1500]
        page = match["metadata"].get("page", "Unknown")
        role = match["metadata"].get("role", "Unknown")
        speaker = match["metadata"].get("speaker", "Unknown")
        context_chunks.append(f"[Page {page} | {speaker}, {role}]\n{text}")
        citations.append(f"Page {page}")

    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial AI assistant that answers questions strictly using the given context. "
                "If the answer is not present in the context, say 'I don't have information.' "
                "Avoid making up facts or speculating. Always cite the page number(s)."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        }
    ]

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)
    final_answer = response.content.strip()

    return f"{final_answer}\n\n📄 Source(s): {', '.join(set(citations))}"

# --- Agent Wrapper ---
rag_tool = Tool(
    name="EarningsCallRAG",
    func=rag_query,
    description="Use this tool to answer financial questions based on the earnings call transcript."
)

query_agent = initialize_agent(
    tools=[rag_tool],
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    agent="zero-shot-react-description",
    verbose=True
)

# --- Main Flow ---
if __name__ == "__main__":
    transcript_path = "bel_transcript.pdf"

    print("\n🔹 Extracting intro text for speaker roles...")
    intro_text = extract_intro_text(transcript_path)
    roles_dict = extract_roles_with_gpt(intro_text)
    print(roles_dict)

    print("\n🔹 Loading and chunking full transcript...")
    chunks = load_and_chunk_pdf(transcript_path)

    print("\n🔹 Assigning speaker metadata to chunks...")
    enriched_chunks = assign_speaker_metadata(chunks, roles_dict)

    print("\n🔹 Embedding and storing in Pinecone...")
    embed_and_store(enriched_chunks)

    print("\n✅ Done: Document chunks embedded with speaker roles.")

    # Sample agent query
    #query = "What was BEL’s revenue in Q2 FY23 and H1 FY23?"
    query="How much orders have been booked in H1?"
    answer = query_agent.run(f"EarningsCallRAG: {query}")
    print("\nAnswer is:\n", answer)
