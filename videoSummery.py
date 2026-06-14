import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Setup API Key
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set. Please set it in your environment or .env file.")

def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    regex = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url_or_id)
    if match:
        return match.group(1)
    return url_or_id if len(url_or_id) == 11 else None

def get_video_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        return " ".join(chunk.text for chunk in transcript_list)
    except Exception:
        return None

def generate_summary(transcript, llm):
    summary_prompt = PromptTemplate.from_template(
        """You are a professional video summarizer.
Provide a clear, detailed, and structured summary of the following video transcript.
Use bullet points and headings to organize the key takeaways, main topics discussed, and conclusions.

Transcript:
{transcript}

Summary:"""
    )
    chain = summary_prompt | llm
    return chain.invoke({"transcript": transcript}).content

def main():
    user_input = input("Enter YouTube URL or Video ID: ").strip()
    video_id = extract_video_id(user_input) if user_input else "Gfr50f6ZBvo"
    if not video_id:
        return

    transcript = get_video_transcript(video_id)
    if not transcript:
        return

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    summary = generate_summary(transcript, llm)
    print(summary + "\n")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    qa_prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer the user's question ONLY using the provided transcript context below.
If the context is insufficient or doesn't mention the topic, say you don't know based on the transcript.

Context:
{context}

Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )

    while True:
        try:
            question = input("Ask a question: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                break
                
            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            formatted_prompt = qa_prompt.invoke({"context": context_text, "question": question})
            answer = llm.invoke(formatted_prompt)
            print(f"\n{answer.content}\n")
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            pass

if __name__ == "__main__":
    main()
