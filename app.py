import streamlit as st
from functools import lru_cache
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import numpy as np
from datetime import datetime
import ollama

st.set_page_config(page_title="Reddit User Analysis", page_icon="👤", layout="wide")
logging.basicConfig(level=logging.INFO)

class Config:
    embedding_model = 'all-MiniLM-L6-v2'
    persist_directory = './reddit_chroma_db2'
    collection_name = 'reddit_content'
    similarity_threshold = 0.3
    max_results = 50

class OptimizedRedditRAGPipeline:
    def __init__(self, config):
        self.config = config
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
            self.chroma_client = chromadb.PersistentClient(
                path=config.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.error(f"Error initializing pipeline: {e}")
            raise

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> list:
        return self.embeddings.embed_query(text)

    def format_timestamp(self, timestamp):
        try:
            return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S UTC')
        except (ValueError, TypeError):
            return 'Unknown'

    def get_user_details(self, username: str):
        try:
            user_results = self.collection.query(
                query_embeddings=self.get_embedding(username),
                where={"$and": [{"source": {"$eq": "user"}}, {"username": {"$eq": username}}]},
                n_results=1
            )
            
            if not user_results['documents']:
                st.warning(f"User {username} not found")
                return None
            
            content_results = self.collection.query(
                query_embeddings=self.get_embedding(username),
                where={"$and": [
                    {"$or": [{"source": {"$eq": s}} for s in ["post", "comment"]]},
                    {"author": {"$eq": username}}
                ]},
                n_results=1000
            )

            return self._process_user_content(
                user_results['metadatas'][0][0],
                content_results['documents'][0],
                content_results['metadatas'][0]
            )
        except Exception as e:
            st.error(f"Error fetching user details: {str(e)}")
            return None

    def _process_user_content(self, user_metadata, documents, metadata):
        posts, comments = [], []
        for doc, meta in zip(documents, metadata):
            content = {
                'text': doc,
                'timestamp': meta.get('timestamp', ''),
                'created_time': self.format_timestamp(meta.get('created_utc')),
                'score': meta.get('score', 0),
                'subreddit': meta.get('subreddit', '')
            }
            
            if meta['source'] == 'post':
                content.update({
                    'title': meta.get('title', ''),
                    'num_comments': meta.get('num_comments', 0),
                    'url': meta.get('url', ''),
                    'permalink': meta.get('permalink', '')
                })
                posts.append(content)
            else:
                content.update({
                    'parent_author': meta.get('parent_author', ''),
                    'permalink': meta.get('permalink', ''),
                    'replies': [{
                        'text': r.get('body', ''),
                        'author': r.get('author', ''),
                        'score': r.get('score', 0),
                        'created_time': self.format_timestamp(r.get('created_utc')),
                        'permalink': r.get('permalink', '')
                    } for r in meta.get('replies', [])]
                })
                comments.append(content)

        return {"user_details": user_metadata, "posts": posts, "comments": comments}

    def find_users_by_topic(self, topic: str):
        try:
            topic_embedding = self.get_embedding(topic)
            results = self.collection.query(
                query_embeddings=topic_embedding,
                n_results=self.config.max_results
            )
            
            if not results['documents'][0]:
                return {}
            
            users_content = {}
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                similarity = np.dot(self.get_embedding(doc), topic_embedding) / (
                    np.linalg.norm(self.get_embedding(doc)) * np.linalg.norm(topic_embedding)
                )
                
                if similarity >= self.config.similarity_threshold:
                    author = meta.get('author', 'Unknown')
                    if author not in users_content:
                        users_content[author] = {'posts': [], 'comments': [], 'relevance': []}
                    
                    content_type = 'posts' if meta['source'] == 'post' else 'comments'
                    if content_type == 'posts' or (content_type == 'comments' and doc not in [c['text'] for c in users_content[author]['comments']]):
                        users_content[author][content_type].append({
                            'text': doc,
                            'created_time': self.format_timestamp(meta.get('created_utc')),
                            'subreddit': meta.get('subreddit', 'Unknown'),
                            'score': meta.get('score', 0)
                        })
                    users_content[author]['relevance'].append(similarity)

            return {
                user: {
                    'posts': content['posts'],
                    'comments': content['comments'],
                    'relevance': float(np.mean(content['relevance'])),
                    'total_relevant_content': len(content['posts']) + len(content['comments'])
                }
                for user, content in users_content.items()
            }
        except Exception as e:
            st.error(f"Error in topic search: {str(e)}")
            return {}

    def get_content_for_synthesis(self, query: str):
        try:
            results = self.collection.query(
                query_embeddings=self.get_embedding(query),
                n_results=10
            )
            
            if not results['documents'][0]:
                return []
            
            return [{
                'type': meta['source'],
                'text': doc,
                'author': meta.get('author', 'Unknown'),
                'subreddit': meta.get('subreddit', 'Unknown'),
                'created_time': self.format_timestamp(meta.get('created_utc')),
                'score': meta.get('score', 0),
                'title': meta.get('title', '') if meta['source'] == 'post' else '',
                'num_comments': meta.get('num_comments', 0) if meta['source'] == 'post' else 0,
                'replies': [{
                    'text': r.get('body', ''),
                    'author': r.get('author', ''),
                    'score': r.get('score', 0),
                    'created_time': self.format_timestamp(r.get('created_utc'))
                } for r in meta.get('replies', [])] if meta['source'] == 'comment' else []
            } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
        except Exception as e:
            st.error(f"Error retrieving content for synthesis: {str(e)}")
            return []

    def synthesize_content(self, query: str, content: list):
        try:
            context = "\n\n".join(
                f"{'Post' if item['type'] == 'post' else 'Comment'} by u/{item['author']} in r/{item['subreddit']} ({item['created_time']}):\n"
                f"{f'Title: {item['title']}\n' if item['type'] == 'post' else ''}"
                f"Content: {item['text']}\n"
                f"Score: {item['score']}{f', Comments: {item['num_comments']}' if item['type'] == 'post' else ''}\n"
                f"{chr(10).join(f'|- Reply by u/{r['author']} ({r['created_time']}):\n   {r['text']}\n   Score: {r['score']}' for r in item['replies']) if item['replies'] else ''}"
                for item in content
            )
            
            response = ollama.chat(model="llama3.2", messages=[{
                "role": "user",
                "content": f"""
                Here is relevant Reddit content about the following query:
                Query: {query}

                Content:
                {context}

                Please provide a comprehensive synthesis that:
                1. Summarizes the main points and insights
                2. Identifies common themes and patterns
                3. Notes any significant disagreements or alternative viewpoints
                4. Highlights particularly insightful or well-received contributions
                5. Maintains factual accuracy without speculation

                Synthesis:
                """
            }])
            
            return response.get('message', {}).get('content', "Error: Could not generate synthesis.")
        except Exception as e:
            return f"Error generating synthesis: {str(e)}"

def main():
    st.sidebar.title("Reddit Insights")
    mode = st.sidebar.radio("Select Mode", ["User Details", "Topic Exploration", "Content Synthesis"])
    
    try:
        pipeline = OptimizedRedditRAGPipeline(Config())
        
        if mode == "User Details":
            st.title("👤 User Details Analysis")
            if username := st.text_input("Enter username"):
                with st.spinner("Fetching user data..."):
                    if details := pipeline.get_user_details(username):
                        cols = st.columns(4)
                        metrics = [
                            ("Total Karma", f"{details['user_details'].get('total_karma', 0):,}"),
                            ("Account Age", f"{details['user_details'].get('account_age_days', 0)} days"),
                            ("Posts", len(details['posts'])),
                            ("Comments", len(details['comments']))
                        ]
                        for col, (label, value) in zip(cols, metrics):
                            col.metric(label, value)
                        
                        with st.expander("📊 Detailed Profile", expanded=True):
                            st.json(details['user_details'])
                        
                        tab1, tab2 = st.tabs(["📝 Posts", "💬 Comments"])
                        
                        with tab1:
                            for post in details['posts']:
                                with st.expander(f"Post: {post.get('title', 'No title')}"):
                                    st.markdown(post.get('text', ''))
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Score", post.get('score', 0))
                                    with col2:
                                        st.text(f"r/{post.get('subreddit', 'unknown')}")
                                    with col3:
                                        st.text(f"Created: {post.get('timestamp', 'unknown')}")
                                    if post.get('url'):
                                        st.markdown(f"[View Post]({post['url']})")
                        
                        with tab2:
                            for comment in details['comments']:
                                with st.expander(f"Comment in r/{comment.get('subreddit', 'unknown')}"):
                                    st.markdown(comment.get('text', ''))
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Score", comment.get('score', 0))
                                    with col2:
                                        st.text(f"Parent: u/{comment.get('parent_author', 'unknown')}")
                                    with col3:
                                        st.text(f"Created: {comment.get('timestamp', 'unknown')}")
                                    
                                    if comment.get('replies'):
                                        st.subheader("Replies:")
                                        for reply in comment['replies']:
                                            with st.container():
                                                st.markdown(f"u/{reply['author']} ({reply['created_time']}):")
                                                st.markdown(reply['text'])
                                                st.metric("Score", reply['score'])
                                                if reply.get('permalink'):
                                                    st.markdown(f"[View Reply](https://reddit.com{reply['permalink']})")
                                    
                                    if comment.get('permalink'):
                                        st.markdown(f"[View Thread](https://reddit.com{comment['permalink']})")
        
        elif mode == "Topic Exploration":
            st.title("🔍 Topic Analysis")
            if topic := st.text_input("Enter topic to search"):
                with st.spinner("Searching..."):
                    if users := pipeline.find_users_by_topic(topic):
                        st.success(f"Found {len(users)} relevant users")
                        for username, data in users.items():
                            with st.expander(f"u/{username} (Relevance: {data['relevance']:.2f})"):
                                st.metric("Total Content", data['total_relevant_content'])
                                for content_type in ['posts', 'comments']:
                                    if data[content_type]:
                                        st.subheader(content_type.title())
                                        for item in data[content_type]:
                                            #st.markdown(f"Posted in r/{item['subreddit']} - {item['created_time']}")
                                            st.markdown(f"- {item['text']}")
                                            st.metric("Score", item['score'])
        
        else:
            st.title("🤖 Content Synthesis")
            st.markdown("Enter a query about any topic, and I'll analyze relevant Reddit content to provide a comprehensive synthesis.")
            
            if query := st.text_input("Enter your query:"):
                with st.spinner("Retrieving and analyzing relevant content..."):
                    if content := pipeline.get_content_for_synthesis(query):
                        st.markdown("### 📝 Synthesis")
                        st.markdown(pipeline.synthesize_content(query, content))
                        
                        with st.expander("View Source Content", expanded=False):
                            for item in content:
                                st.markdown(f"---\n*{item['type'].title()} by u/{item['author']} in r/{item['subreddit']}*")
                                st.markdown(f"{item['created_time']}")
                                if item['type'] == 'post':
                                    st.markdown(f"Title: {item['title']}")
                                st.markdown(item['text'])
                                st.markdown(f"Score: {item['score']}")
                                
                                if item['replies']:
                                    st.markdown("Replies:")
                                    for reply in item['replies']:
                                        st.markdown(f"> u/{reply['author']} ({reply['created_time']}):")
                                        st.markdown(f"> {reply['text']}")
                                        st.markdown(f"> Score: {reply['score']}")
                    else:
                        st.warning("No relevant content found for your query.")
                        
    except Exception as e:
        st.error(f"Failed to initialize the application: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()