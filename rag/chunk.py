import os

from langchain_community.vectorstores import FAISS

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 镜像源，解决 huggingface 网络问题

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import re
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 批量读取 data 文件夹下所有.txt 文件
data_dir = r"E:\Agent\face_protect-agent\data"
all_docs = []

# 遍历文件夹，加载所有 txt
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        logger.info(f"正在加载：{filename}")

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        all_docs.extend(docs)  # 把文档放进列表

logger.info(f"全部文本加载完成，共 {len(all_docs)} 个文件")


# 文本清洗
def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[§￥%……&*]+", " ", text)
    return text.strip()


logger.info(f"开始文本清洗与分块...")

# 清洗 + 分块所有文档
all_chunks = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=15,
    length_function=len,
    separators=["\n\n", "\n", "。", " ", ""]
)

for doc in all_docs:
    cleaned = clean_text(doc.page_content)
    chunks = splitter.split_text(cleaned)
    all_chunks.extend(chunks)

logger.info(f"文本分块完成，总块数：{len(all_chunks)}")
logger.info(f"下一步进行文本向量化并储存")

# 向量化并储存
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

#faiss向量化数据存在本地
vectorstore = FAISS.from_texts(all_chunks, embeddings)
vectorstore.save_local(r"E:\Agent\face_protect-agent\faiss_index")  # 存在项目目录下

logger.info(f"所有文档向量化完成！")