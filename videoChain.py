import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    regex = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url_or_id)
    if match:
        return match.group(1)
    return url_or_id if len(url_or_id) == 11 else None

def main():
    user_input = input("Enter YouTube URL or Video ID: ").strip()
    video_id = extract_video_id(user_input) if user_input else "Gfr50f6ZBvo"
    if not video_id:
        print("Invalid YouTube URL or Video ID.")
        return

    # Fetch and prepare transcript document
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except Exception as exc:
        print(f"Failed to fetch transcript: {exc}")
        return

    # Split text and build FAISS retriever
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Define LCEL RAG components
    qa_prompt = PromptTemplate.from_template(
        "Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    llm = ChatOllama(model="gpt-oss:120b-cloud")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Clean, short LangChain Expression Language (LCEL) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt | llm | StrOutputParser()
    )

    # Interactive Q&A loop
    while True:
        try:
            question = input("\nAsk a question (or type 'exit' to quit): ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                break
            
            answer = rag_chain.invoke(question)
            print(f"\n{answer}\n")
        except (KeyboardInterrupt, SystemExit):
            break

if __name__ == "__main__":
    main()