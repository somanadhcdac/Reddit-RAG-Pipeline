import praw
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import unicodedata
import re
from prawcore.exceptions import TooManyRequests, ResponseException
import random
import pytz

def setup_reddit_client():
    """Initialize Reddit API client with direct credentials"""
    return praw.Reddit(
        client_id='OcIAPOXK3LXJal99JgQYjw',
        client_secret='vtWx1bQPGWmOXTb5_fAJeepu8NO8Ug',
        user_agent='Used_Camera_9923'
    )

def format_timestamp(timestamp):
    """Convert Unix timestamp to IST format"""
    ist = pytz.timezone('Asia/Kolkata')
    dt = datetime.fromtimestamp(timestamp).astimezone(ist)
    return dt.strftime('%Y-%m-%d %H:%M:%S IST')

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    unicode_map = {
        '"': '"', '"': '"', '„': '"', '⹂': '"',
        ''': "'", ''': "'", '‚': "'", '‛': "'",
        '–': '-', '—': '-', '―': '-', '‐': '-', '‑': '-', '‒': '-',
        '\u200b': '', '\u2009': ' ', '\u200a': ' ',
        '…': '...', '⋯': '...', '‹': '<', '›': '>',
        '«': '<<', '»': '>>', '′': "'", '″': '"',
        '‴': "'''", '⁗': '""""',
        '\n': ' ', '\r': ' ', '\t': ' ', '\f': ' ', '\v': ' '
    }
    
    for unicode_char, ascii_char in unicode_map.items():
        text = text.replace(unicode_char, ascii_char)
    
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(char for char in text if not unicodedata.category(char).startswith('C') 
                  or unicodedata.category(char) in {'Co'})
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

class RateLimitHandler:
    def __init__(self, initial_delay=1, max_delay=64, max_retries=5):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        
    def execute_with_retry(self, func, *args, **kwargs):
        delay = self.initial_delay
        retries = 0
        
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except (TooManyRequests, ResponseException) as e:
                if '429' not in str(e) or retries == self.max_retries - 1:
                    raise
                
                sleep_time = min(delay * (1 + random.random()), self.max_delay)
                print(f"\nRate limited. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                delay *= 2
                retries += 1
        
        raise Exception("Max retries exceeded")

class RedditExtractor:
    def __init__(self, subreddits):
        self.reddit = setup_reddit_client()
        self.users_data = {}
        self.rate_limiter = RateLimitHandler()
        self.subreddits = subreddits

    def get_user_data(self, username):
        """Get comprehensive user details"""
        if username == '[deleted]' or username in self.users_data:
            return
            
        try:
            user = self.reddit.redditor(username)
            created_utc = user.created_utc
            account_age = (datetime.now() - datetime.fromtimestamp(created_utc)).days
            
            self.users_data[username] = {
                'username': username,
                'timestamp': format_timestamp(created_utc),
                'account_age_days': account_age,
                'link_karma': user.link_karma,
                'comment_karma': user.comment_karma,
                'total_karma': user.link_karma + user.comment_karma,
                'posts_per_day': round(user.link_karma / max(1, account_age), 2),
                'comments_per_day': round(user.comment_karma / max(1, account_age), 2),
                'has_verified_email': bool(getattr(user, 'has_verified_email', False)),
                'is_gold': bool(getattr(user, 'is_gold', False)),
                'is_mod': bool(getattr(user, 'is_mod', False))
            }
            
        except Exception as e:
            print(f"Error fetching user {username}: {str(e)}")
            self.users_data[username] = {'username': username}

    def process_comments(self, comments, depth=0):
        """Process all comments recursively without limits"""
        processed = []
        for comment in comments:
            try:
                if not hasattr(comment, 'author'):
                    continue

                author = str(comment.author) if comment.author else '[deleted]'
                parent_author = '[deleted]'
                
                if hasattr(comment, 'parent') and isinstance(comment.parent(), praw.models.Comment):
                    parent_author = str(comment.parent().author) if comment.parent().author else '[deleted]'
                
                if author != '[deleted]':
                    self.get_user_data(author)
                if parent_author != '[deleted]':
                    self.get_user_data(parent_author)
                
                comment_data = {
                    'author': author,
                    'parent_author': parent_author,
                    'timestamp': format_timestamp(comment.created_utc),
                    'body': clean_text(comment.body),
                    'score': comment.score,
                    'depth_level': depth,
                    'replies': []
                }
                
                if hasattr(comment, 'replies'):
                    comment_data['replies'] = self.process_comments(comment.replies, depth + 1)
                    
                processed.append(comment_data)
                
            except Exception as e:
                print(f"Error processing comment: {str(e)}")
                continue
                
        return processed

    def extract_post(self, submission):
        """Extract post data with rate limit handling"""
        try:
            if submission.author:
                self.get_user_data(str(submission.author))
            
            submission.comments.replace_more(limit=None)  # Get all comments
            
            return {
                'title': clean_text(submission.title),
                'author': str(submission.author) if submission.author else '[deleted]',
                'timestamp': format_timestamp(submission.created_utc),
                'content': clean_text(submission.selftext),
                'url': clean_text(submission.url),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'subreddit': clean_text(submission.subreddit.display_name),
                'comments': self.process_comments(submission.comments)
            }
        except Exception as e:
            print(f"Error extracting post {submission.id}: {str(e)}")
            return None

    def extract_subreddit(self, subreddit_name):
        """Extract all posts from a subreddit"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            for submission in subreddit.top(limit=100):  # No limit on posts
                post_data = self.extract_post(submission)
                if post_data:
                    posts.append(post_data)
            
            return posts
        except Exception as e:
            print(f"Error accessing subreddit {subreddit_name}: {str(e)}")
            return []

    def run_extraction(self):
        """Run extraction process"""
        all_data = {'posts': [], 'users': []}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(self.extract_subreddit, subreddit): subreddit 
                      for subreddit in self.subreddits}
            
            with tqdm(total=len(self.subreddits), desc="Extracting subreddits") as pbar:
                for future in as_completed(futures):
                    subreddit = futures[future]
                    try:
                        posts = future.result()
                        all_data['posts'].extend(posts)
                        pbar.update(1)
                        pbar.set_description(f"Processed r/{subreddit}")
                    except Exception as e:
                        print(f"Error processing r/{subreddit}: {str(e)}")

        all_data['users'] = list(self.users_data.values())
        return all_data

def main():
    # List of subreddits to scrape
    subreddits = ['LocalLLaMA','MachineLearning','OpenAI','artificial','datascience']  # Add your subreddits here
    
    start_time = time.time()
    extractor = RedditExtractor(subreddits)
    data = extractor.run_extraction()
    
    output_file = f"reddit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nData saved to {output_file}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Total posts collected: {len(data['posts'])}")
    print(f"Total unique users: {len(data['users'])}")

if __name__ == "__main__":
    main()