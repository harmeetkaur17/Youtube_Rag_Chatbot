from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import yt_dlp
import requests
from dotenv import load_dotenv

load_dotenv()

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

# Initialize session state
if "pipeline_loaded" not in st.session_state:
    st.session_state["pipeline_loaded"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")

st.title("🎥 YouTube RAG Chatbot")
st.write("Ask questions about the content of any YouTube video with English captions!")


# --------------------------
# Step 1: Get Video ID
# --------------------------
video_url = st.text_input("Enter YouTube video URL or ID:")

if st.button("Load Video"):
    if not video_url:
        st.error("Please enter a valid YouTube video URL or ID.")
    else:
        # Extract video ID if a full URL
        if "youtube.com" in video_url or "youtu.be" in video_url:
            import re
            match = re.search(r"(?:v=|be/)([a-zA-Z0-9_-]{11})", video_url)
            if match:
                video_id = match.group(1)
            else:
                st.error("Could not extract video ID.")
                st.stop()
        else:
            video_id = video_url  # assume user entered ID directly


    
    # --------------------------
    # Step 2: Fetch Transcript
    # --------------------------

    ydl_opts = {
    "skip_download": True,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["en"],
    "quiet": True,
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            # Check manual or auto captions
            subs = info.get("subtitles") or info.get("automatic_captions")
            if not subs or "en" not in subs:
                st.error("❌ No English transcript available for this video.")
                st.stop()

            # Fetch the first English subtitle URL
            captions_url = subs["en"][0]["url"]
            r = requests.get(captions_url)
            r.raise_for_status()
            caption_json = r.json()
            transcript_text = ""
            for event in caption_json.get("events", []):
                if "segs" in event:
                    for seg in event["segs"]:
                        if "utf8" in seg:
                            transcript_text += seg["utf8"]
            transcript_text = transcript_text.replace("\n", " ").strip()

            st.success("✅ Transcript fetched successfully!")
            st.text_area("Transcript Preview", transcript_text[:2000], height=300)
            
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {e}")
        st.stop()

    # --------------------------
    # Step 3: Text Splitting

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    # --------------------------
    # Step 4: Embedding and Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)


    # --------------------------
    # Step 5: Retrival
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # --------------------------
    # Step 6: Augmented Generation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
    )

    # Save objects in session_state so they persist across reruns
    st.session_state["retriever"] = retriever
    st.session_state["llm"] = llm
    st.session_state["prompt"] = prompt
    st.session_state["pipeline_loaded"] = True
    st.success("✅ Pipeline ready! You can now ask questions.")

# --------------------------
# Step 2: Ask Questions
# --------------------------
if st.session_state.get("pipeline_loaded", False):

    st.subheader("Ask a question about the video")
    user_question = st.text_input("Type your question here:")

    
    if st.button("Get Answer"):
        retriever = st.session_state["retriever"]
        llm = st.session_state["llm"]
        prompt = st.session_state["prompt"]
        parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()

        main_chain = parallel_chain | prompt | llm | parser

        answer = main_chain.invoke(user_question)
        st.text_area("Chatbot Answer", answer, height=200) 
