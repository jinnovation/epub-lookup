# for some reason not importing torch first...segfaults??
import torch
import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import TextLoader, UnstructuredEPubLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import faiss
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# TODO: Start ollama in background if not already running


# TODO: persist the embeddings database. find a way to cache the embeddings for
# each book by some combination of: hash of the entire file; embedding model
# used


# TODO: complain if the file provided is not epub


# TODO: support loading and performing queries on an entire Calibre library


# TODO: Use llama2:7b-chat instead of vanilla llama; use ChatOllama "component"


# TODO: Support JSON schema? https://python.langchain.com/docs/integrations/chat/ollama#extraction
def main(file: str, query: str = "What is the title of the book?"):
    loader = UnstructuredEPubLoader(file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)

    chunked_docs = splitter.split_documents(docs)

    # embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    embeddings_model = HuggingFaceEmbeddings()

    db = faiss.FAISS.from_documents(chunked_docs, embeddings_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs=dict(k=4))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    llm = Ollama(model="llama2")

    msgs = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "Answer the question based on your knowledge. Use the following context to help:\n{context}",
            },
            {
                "role": "user",
                "content": "{question}",
            },
        ],
        tokenize=False,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=msgs,
    )

    llm_chain = prompt | llm | StrOutputParser()

    rag_chain = {
        "context": db.as_retriever(),
        "question": RunnablePassthrough(),
    } | llm_chain

    print(rag_chain.invoke(query))


if __name__ == "__main__":
    typer.run(main)
