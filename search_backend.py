import pickle
import os
import re
import math
from collections import Counter, defaultdict
from google.cloud import storage
from inverted_index_gcp import InvertedIndex
import nltk
from nltk.corpus import stopwords

import gzip, csv #for pagerank

nltk.download('stopwords')

class SearchBackend: #Backend class to run when running the search_frontend script
    def __init__(self):
        print("Initializing Backend")
        self.bucket_name = 'eyalir1'   #the bucket we will get the Inverted Indexes/PageRank/Pageviews from
        self.client = storage.Client() #in order to reach the bucket we will do so using storage client
        self._download_all_data() #running private method to locally download all the data
        
        print("Loading indices...")
        self.body_index = InvertedIndex.read_index('postings_gcp_body', 'index_body') #Inverted Index for the body
        self.body_index.base_dir = 'postings_gcp_body' # for getting the correct path
        self.title_index = InvertedIndex.read_index('postings_gcp_title', 'index_title') #Inverted Index for the title
        self.title_index.base_dir = 'postings_gcp_title' #for getting the correct path
        self.anchor_index = InvertedIndex.read_index('postings_gcp_anchor', 'index_anchor') #Inverted Index for the anchor
        self.anchor_index.base_dir = 'postings_gcp_anchor' #for getting the correct path

        with open('doc_title_mapping.pkl', 'rb') as f: #setting the doc title mapping pickle to a variable 
            self.doc_title = pickle.load(f)
        if os.path.exists('pageviews.pkl'): #setting the pageviews pickle to a variable
            with open('pageviews.pkl', 'rb') as f:
                self.pageviews = pickle.load(f)
        else:
            self.pageviews = {}

        self.pagerank = self._load_pagerank('pr') #loading the pagerank using private method
        self.N = len(self.doc_title) #getting the number of documents
        
        self.all_stopwords = frozenset(stopwords.words('english')) #the stopwords we will use
        print("Backend ready")

    def _download_all_data(self):
        """
        A private method for downloading all the data locally
        """
        files = [ #the pickle files we need to download
            ('postings_gcp_body/index_body.pkl', 'postings_gcp_body/index_body.pkl'),
            ('postings_gcp_title/index_title.pkl', 'postings_gcp_title/index_title.pkl'),
            ('postings_gcp_anchor/index_anchor.pkl', 'postings_gcp_anchor/index_anchor.pkl'),
            ('doc_title_mapping.pkl', 'doc_title_mapping.pkl'),
            ('pageviews.pkl', 'pageviews.pkl')
        ]
        
        for remote, local in files: #going through each pickle file, and downloading it
            self._download_file(remote, local)

        if not os.path.exists('postings_gcp_title') or not os.listdir('postings_gcp_title'): #downloading the title .bin files. We check if the folder exists and if it has files inside
             print("Downloading Title Index")
             self._download_folder('postings_gcp_title/')
        else:
             print("Title Index found locally. Skipping download.")

        if not os.path.exists('postings_gcp_anchor') or not os.listdir('postings_gcp_anchor'):#downloading the anchor .bin files. We check if the folder exists and if it has files inside
            print("Downloading Anchor Index...")
            self._download_folder('postings_gcp_anchor/')
        else:
            print("Anchor Index found locally. Skipping download.")
        
        if not os.path.exists('postings_gcp_body') or not os.listdir('postings_gcp_body'): #downloading the body .bin files. We check if the folder exists and if it has files inside
            print("Downloading Body Index...")
            self._download_folder('postings_gcp_body/')
        else:
            print("Body Index found locally. Skipping download.")

        if not os.path.exists('pr') or not os.listdir('pr'): #downloading the pagerank .bin files. We check if the folder exists and if it has files inside
            print("Downloading PageRank...")
            self._download_folder('pr/')
        else:
            print("PageRank found locally. Skipping download.")

    def _download_folder(self, prefix):
        """Downloads all files in a bucket folder to local disk"""
        if not os.path.exists(prefix):
            os.makedirs(prefix, exist_ok=True) #incase the directory does not exist, we will make it
            
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix) #the blobs from the client connected to the GCP bucke 
        for blob in blobs:
            if blob.name.endswith('/'): continue # skipping directories
            local_path = blob.name
            if not os.path.exists(local_path): #checking if the file exists to save time for possible repititions
                print(f"Downloading {local_path}...")
                blob.download_to_filename(local_path)

    def _download_file(self, remote, local):
        """
        Private method to download files. The remote is to get the blobl and the local is the local directory for the file
        """
        if os.path.exists(local): return #if the path exists, dont do anything
        directory = os.path.dirname(local) #the local directory
        if directory: os.makedirs(directory, exist_ok=True) 
        blob = self.client.bucket(self.bucket_name).blob(remote) #getting the blob
        if blob.exists(): blob.download_to_filename(local) #if the blob exists, we will use the download to filename method 

    def _load_pagerank(self, pr_dir): 
        """
        Method for loading the pagerank given a pagerank directory
        """
        pr_dict = {} #the final pagerank
        
        for filename in os.listdir(pr_dir): #getting all the filenames from the pagerank directory
            if filename.endswith(".csv.gz"): #incase the filename is of type csv.gz, we will open it and add the content to the pagerank
                with gzip.open(os.path.join(pr_dir, filename), 'rt') as f:
                    for row in csv.reader(f):
                        pr_dict[int(row[0])] = float(row[1])
        return pr_dict #return the final pagerank as a dictionary

    def tokenize(self, text):
        """
        method for tokenizing a text, will be used to tokenize queries for the 'search' methods
        """
        return [t.group() for t in re.finditer(r'\w+', text.lower()) if t.group() not in self.all_stopwords]

    def _get_posting_list(self, index, token):
        """
        given the index and token we will use the read_a_posting_list method from the given Inverted Index class
        """
        return index.read_a_posting_list('', token, bucket_name=None)
    def get_pagerank(self, wiki_ids): 
        """#getting a pagerank for a specific set of wiki ids """
        return [self.pagerank.get(int(doc_id), 0) for doc_id in wiki_ids]

    def get_pageview(self, wiki_ids):
        """
        getting the pageviews for a set of specific wiki ids
        """
        return [self.pageviews.get(int(doc_id), 0) for doc_id in wiki_ids]

    def search_body(self, query, use_cos_sim=False):
        """
        method for searching the body inverted index given a query
        """
        tokens = self.tokenize(query) #tokenize the queries
        if not tokens: return [] #if smth went wrong return an empty list

        scores = defaultdict(float) #the scores for relevant documents
        threshold_df = 80000  #a threshold for the df to ease on runtime
        
        query_counts = Counter(tokens) #using a counter for getting metrics
        
        for token, q_tf in query_counts.items(): #for each token and the amount of times it shows up in the query, do the following
            if token in self.body_index.df: #incase the token is in the body index df
                df = self.body_index.df[token] #get the df of that token
                
                if df > threshold_df: #incase the token is too common, we skip it for runtime and precision reasons 
                    continue 
                
                pl = self._get_posting_list(self.body_index, token) #the postinglist of the token
                idf = math.log(self.N / df, 10) #get the idf of the token

                if use_cos_sim:
                    # === COSINE SIMILARITY (DOT PRODUCT VERSION) ===
                    # We calculate the numerator of the Cosine formula which satisfies "using tf-idf on the body"
                    
                    w_query = q_tf * idf # calculating the weight of a term in query
                    
                    
                    for doc_id, tf in pl: # multiplying by weight of term in doc (Dot Product)
                        w_doc = tf * idf
                        scores[doc_id] += w_query * w_doc
                else:
                    for doc_id, tf in pl: #for each doc and its tf in the posting list of the token
                        score = (tf / (tf + 1.2)) * idf * q_tf #calculate the following, which is the bm25 lite saturation formula
                        scores[doc_id] += (score)
                    
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100] #return top 100 most relevant documents

    def search_title(self, query):
        """
        method for searching the title inverted index given a query
        """
        tokens = self.tokenize(query) #tokenizing the query
        scores = defaultdict(int)
        for token in set(tokens): #going over unique tokens (we use set() so we dont count the same word twice)
            if token in self.title_index.df: # retrieving documents that contain this token in their title
                
                for doc_id, _ in self._get_posting_list(self.title_index, token): # we ignore the frequency because binary is what matters
                    scores[doc_id] += 1 #adding 1 to the score of that document
        return sorted(scores.items(), key=lambda x: x[1], reverse=True) #returning all documents

    def search_anchor(self, query):
        """
        method for searching the title inverted index given a query
        """
        tokens = self.tokenize(query) #tokenizing the query
        scores = defaultdict(int)
        for token in set(tokens):  #going over unique tokens (we use set() so we dont count the same word twice)
            if token in self.anchor_index.df: # retrieving documents that contain this token in their title
                for doc_id, _ in self._get_posting_list(self.anchor_index, token): # we ignore the frequency because binary is what matters
                    scores[doc_id] += 1 #adding 1 to the score of that document
        return sorted(scores.items(), key=lambda x: x[1], reverse=True) #returning all documents

    def search(self, query, w_title=1.0, w_anchor=0.5, w_body=1.0, w_pr=0.1,w_pv=0.1):
        """
        Main search function. Here we execute the search logic by combining scores from 
        multiple indices and the ranking boosters which are pagerank and pageviews

        The scoring strategy includes a Binary matching with weighted presence on the title, a logarithmic scaling of term frequency in anchor text,
        TF-IDF scoring with BM25 and stopword pruning (df > 80k) and log-scaled pageRank and pageview scores
        Returns:
            list: A sorted list of the top 100 tuples (doc_id, score)
        """
        tokens = self.tokenize(query) #tokenizing the query
        if not tokens: return []
        
        merged_scores = defaultdict(float) #the final scores
        
        for token in set(tokens): #the title score, same as the the title function without getting top 100 scores
            if token in self.title_index.df:
                for doc_id, _ in self._get_posting_list(self.title_index, token):
                    merged_scores[doc_id] += w_title

        anchor_counts = defaultdict(int) #the anchor score, same as the anchor functionwithout getting the top 100 scores
        for token in set(tokens):
            if token in self.anchor_index.df:
                for doc_id, _ in self._get_posting_list(self.anchor_index, token):
                    anchor_counts[doc_id] += 1
        
        for doc_id, count in anchor_counts.items():  #using log scaling for anchor to prevent spam (from experience, 'Mount Everest')
            merged_scores[doc_id] += w_anchor * math.log(1 + count, 10)

        threshold_df = 80000 #skipping very common words to save time
        
        query_counts = Counter(tokens) #same functionality as the search body but without returning top 100
        for token, q_tf in query_counts.items(): #for each token and the amount of times it shows up in the query, do the following
            if token in self.body_index.df: #incase the token is in the body index df
                df = self.body_index.df[token] #get the df of that token
                
                if df > threshold_df: #incase the token is too common, we skip it for runtime and precision reasons 
                    continue 
                
                pl = self._get_posting_list(self.body_index, token) #the postinglist of the token
                idf = math.log(self.N / df, 10) #get the idf of the token
                
                for doc_id, tf in pl: #for each doc and its tf in the posting list of the token
                    score = (tf / (tf + 1.2)) * idf * q_tf #calculate the following, which is the bm25 lite saturation formula
                    merged_scores[doc_id] += (score)

        for doc_id in merged_scores.keys(): #getting the pagerank (pr) and pageview (pv) of the document for all the found docs with logarithmic scaling
            pr = self.pagerank.get(doc_id, 0)
            pv = self.pageviews.get(doc_id, 0)
            
            if pr > 0: merged_scores[doc_id] += w_pr * math.log(pr, 10) #adding the pr and pv to the scores
            if pv > 0: merged_scores[doc_id] += w_pv * math.log(pv, 10)

        return sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:100] #returning top 100 scores
