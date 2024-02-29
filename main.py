# for some reason not importing torch first...segfaults??
import torch
import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


def main(file: str, query: str = "What is the title of the book?"):
    # loader = UnstructuredEPubLoader(file)
    loader = TextLoader(file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)

    chunked_docs = splitter.split_documents(docs)
    print("split")

    # embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    embeddings_model = HuggingFaceEmbeddings()
    print("embedding model loaded")

    db = faiss.FAISS.from_documents(chunked_docs, embeddings_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs=dict(k=4))

    print("db")

    model_name = "HuggingFaceH4/zephyr-7b-beta"

    # NB(@jinnovation): Does not support M1
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        # model_name, quantization_config=bnb_config
        model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("model and tokenizer")

    text_gen_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=400,
    )

    llm = HuggingFacePipeline(
        pipeline=text_gen_pipeline,
    )

    prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm_chain = prompt | llm | StrOutputParser()

    rag_chain = {
        "context": db.as_retriever(),
        "question": RunnablePassthrough(),
    } | llm_chain

    print("chains")
    print(rag_chain.invoke(query))


if __name__ == "__main__":
    typer.run(main)
