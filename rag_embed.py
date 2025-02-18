import json
from tqdm import tqdm
import logging
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import hashlib
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Any
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncJsonToChromaProcessor:
    def __init__(self, persist_directory="chroma_db", batch_size=100, max_workers=4):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="reddit_content", metadata={"hnsw:space": "cosine"})
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
    
    def generate_unique_id(self, content: str) -> str:
        """Generate a unique ID for a document."""
        return hashlib.md5(f"{content}|{uuid.uuid4()}".encode()).hexdigest()[:16]
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using thread executor."""
        async with self.semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.embeddings.embed_documents,
                texts
            )
    
    async def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of documents with embeddings."""
        texts = [doc['text'] for doc in documents]
        embeddings = await self.generate_embeddings(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc['embedding'] = emb
        
        return documents
    
    async def store_batch(self, batch: List[Dict[str, Any]]):
        """Store a batch of documents in ChromaDB."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                partial(
                    self.collection.add,
                    ids=[doc['id'] for doc in batch],
                    documents=[doc['text'] for doc in batch],
                    embeddings=[doc['embedding'] for doc in batch],
                    metadatas=[doc['metadata'] for doc in batch]
                )
            )
        except Exception as e:
            logger.error(f"Error storing batch: {e}")
            raise
    
    def process_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single post."""
        return {
            'id': self.generate_unique_id(post.get('title', '')),
            'text': f"{post.get('title', '')} {post.get('content', '')}",
            'metadata': {
                'source': 'post',
                'type': 'post',
                'author': post.get('author', ''),
                'timestamp': post.get('timestamp', ''),
                'subreddit': post.get('subreddit', ''),
                'score': post.get('score', 0),
                'num_comments': post.get('num_comments', 0),
                'url': post.get('url', '')
            }
        }
    
    def process_comment(self, comment: Dict[str, Any], parent_id: str = None, level: int = 0) -> Dict[str, Any]:
        """Process a single comment."""
        return {
            'id': self.generate_unique_id(comment.get('body', '')),
            'text': comment.get('body', ''),
            'metadata': {
                'source': 'comment',
                'type': 'comment',
                'parent_id': parent_id,
                'depth_level': level,
                'author': comment.get('author', ''),
                'timestamp': comment.get('timestamp', ''),
                'score': comment.get('score', 0),
                'parent_author': comment.get('parent_author', '')
            }
        }
    
    def process_comments_recursive(self, comments: List[Dict[str, Any]], parent_id: str = None, level: int = 0) -> List[Dict[str, Any]]:
        """Process comments recursively."""
        processed = []
        for comment in comments:
            comment_doc = self.process_comment(comment, parent_id, level)
            processed.append(comment_doc)
            
            if comment.get('replies'):
                processed.extend(self.process_comments_recursive(
                    comment['replies'], 
                    comment_doc['id'], 
                    level + 1
                ))
        return processed

    async def process_documents_parallel(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents in parallel batches."""
        batches = [documents[i:i + self.batch_size] 
                  for i in range(0, len(documents), self.batch_size)]
        
        processed_batches = await asyncio.gather(
            *[self.process_batch(batch) for batch in batches]
        )
        
        return [doc for batch in processed_batches for doc in batch]

    async def store_documents_parallel(self, documents: List[Dict[str, Any]]):
        """Store documents in parallel batches."""
        batches = [documents[i:i + self.batch_size] 
                  for i in range(0, len(documents), self.batch_size)]
        
        await asyncio.gather(
            *[self.store_batch(batch) for batch in batches]
        )

    async def load_json_to_chroma(self, json_path: str) -> int:
        """Load and process JSON file into ChromaDB asynchronously."""
        try:
            # Load JSON file
            logger.info(f"Loading JSON from {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Process posts and comments
            logger.info("Processing posts and comments")
            for post in tqdm(data.get('posts', []), desc="Processing posts"):
                post_doc = self.process_post(post)
                documents.append(post_doc)
                
                if post.get('comments'):
                    documents.extend(self.process_comments_recursive(
                        post['comments'],
                        post_doc['id']
                    ))
            
            # Process users
            logger.info("Processing users")
            for user in tqdm(data.get('users', []), desc="Processing users"):
                documents.append({
                    'id': self.generate_unique_id(user.get('username', '')),
                    'text': f"User {user.get('username', '')} profile information",
                    'metadata': {
                        'source': 'user',
                        'type': 'user',
                        'username': user.get('username', ''),
                        'account_age_days': user.get('account_age_days', 0),
                        'total_karma': user.get('total_karma', 0),
                        'posts_per_day': user.get('posts_per_day', 0),
                        'comments_per_day': user.get('comments_per_day', 0),
                        'is_mod': user.get('is_mod', False)
                    }
                })
            
            # Process documents in parallel
            logger.info("Generating embeddings")
            processed_docs = await self.process_documents_parallel(documents)
            
            # Store documents in parallel
            logger.info("Storing documents in ChromaDB")
            await self.store_documents_parallel(processed_docs)
            
            logger.info(f"Successfully processed {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error processing JSON: {e}")
            raise
        finally:
            self.executor.shutdown(wait=True)

async def main():
    # Example usage
    processor = AsyncJsonToChromaProcessor(
        persist_directory="reddit_chroma_db2",
        batch_size=100,
        max_workers=4
    )
    
    # Process JSON file
    num_docs = await processor.load_json_to_chroma("reddit_data_20250212_105139.json")
    print(f"Processed {num_docs} documents")

if __name__ == "__main__":
    asyncio.run(main())