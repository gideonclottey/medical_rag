from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document



def load_pdf_files(data: str) -> List[Document]:
    """
    Load PDFs from a directory using LangChain DirectoryLoader + PyPDFLoader.

    Guard rails:
    - Validates path exists + is a directory
    - Accepts relative or absolute paths
    - Checks that PDFs exist before trying to load
    - Catches loader errors and re-raises with clearer context
    """

    # Resolve path safely (works on Windows + Git Bash + Jupyter)
    data_path = Path(data).expanduser()
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()

    # 1) Directory checks
    if not data_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {data_path}\n"
            f"Tip: check your current working directory or pass an absolute path."
        )
    if not data_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {data_path}")

    # 2) PDF presence check (matches glob="*.pdf")
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {data_path}\n"
            f"Tip: confirm your PDFs are in that folder and end with .pdf"
        )

    # 3) Load with defensive error handling
    try:
        loader = DirectoryLoader(
            str(data_path),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            # silent_errors=True,  # optional: skip bad PDFs instead of failing the whole load
        )
        documents = loader.load()

    except PermissionError as e:
        raise PermissionError(
            f"Permission error reading from: {data_path}\n"
            f"Tip: close PDFs that might be open, or move the folder somewhere accessible."
        ) from e

    except Exception as e:
        # Catch-all with context
        raise RuntimeError(
            f"Failed while loading PDFs from: {data_path}\n"
            f"Found {len(pdf_files)} pdf(s): {[p.name for p in pdf_files][:5]}{'...' if len(pdf_files) > 5 else ''}\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    # 4) Sanity check on output
    if not documents:
        raise RuntimeError(
            f"Loader ran but returned 0 documents from: {data_path}\n"
            f"Tip: one of the PDFs might be empty, scanned-only, or unreadable."
        )

    return documents


#The content meta data is too much so will take what we want and exclude the rest

from typing import List
from langchain.schema import Document


def filter_to_minimal_docs(docs:List[Document])-> List[Document]:
    ''' This function will return only the source, page lable and page_content'''
    minimal_docs:List[Document] =[]
    for doc in docs:
        src =doc.metadata.get("source")
        page_label =doc.metadata.get("page_label")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src,
                          "page_label": page_label
}
            )
        )
    return minimal_docs


# Now we have our new docs so we have to chunk it 
from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_split(new_filtered_doc):
    ''' This function helps you '''

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =500,
        chunk_overlap =20,
    )
    chunk_text = text_splitter.split_documents(new_filtered_doc)
    return chunk_text

#now we have the chunckes, we need to convert them to numerical representation called embedding 

from langchain.embeddings import HuggingFaceBgeEmbeddings

def download_embeddings():
    ''' Download and return the hugging face embeddings model'''
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name =model_name
    )
    return embeddings

embeddings = download_embeddings()
