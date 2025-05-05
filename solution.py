import json
import os
import re
import math
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set, Union
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class EnhancedMovieKnowledgeEngine:
    """
    An advanced learning engine that creates intelligent relationships between 
    movies and knowledge base documents using hybrid search techniques
    """
    
    def __init__(self, cache_file="movie_kb_cache.pkl"):
        self.cache_file = cache_file
        self.movie_vectors = {}
        self.kb_vectors = {}
        self.movie_keywords = {}
        self.kb_keywords = {}
        self.movie_kb_relations = defaultdict(list)
        self.user_interactions = defaultdict(list)
        self.keyword_index = defaultdict(set)
        self.genre_mapping = defaultdict(set)
        self.era_mapping = defaultdict(set)
        self.theme_mapping = defaultdict(set)
        self.similarity_threshold = 0.15  # Default threshold
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Knowledge graph for collaborative learning
        self.knowledge_graph = {
            'movie_to_movie': defaultdict(set),  # Similar movies
            'keyword_to_movie': defaultdict(set),  # Keywords to movies
            'kb_to_movie': defaultdict(set),      # KB docs to movies
            'theme_to_movie': defaultdict(set)    # Themes to movies
        }
        
        # Load cached data if available
        self._load_cache()
        
        # Initialize stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback if NLTK data not available
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                           'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that',
                           'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                           'have', 'has', 'had', 'do', 'does', 'did', 'to', 'from', 'in', 'out',
                           'on', 'off', 'over', 'under', 'at', 'by', 'for', 'with', 'about'}
            self.stop_words = common_words
    
    def _load_cache(self):
        """Load previously cached data if available"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.movie_vectors = cache_data.get('movie_vectors', {})
                    self.kb_vectors = cache_data.get('kb_vectors', {})
                    self.movie_kb_relations = cache_data.get('movie_kb_relations', defaultdict(list))
                    self.movie_keywords = cache_data.get('movie_keywords', {})
                    self.kb_keywords = cache_data.get('kb_keywords', {})
                    self.keyword_index = cache_data.get('keyword_index', defaultdict(set))
                    self.genre_mapping = cache_data.get('genre_mapping', defaultdict(set))
                    self.era_mapping = cache_data.get('era_mapping', defaultdict(set))
                    self.theme_mapping = cache_data.get('theme_mapping', defaultdict(set))
                    self.user_interactions = cache_data.get('user_interactions', defaultdict(list))
                    self.knowledge_graph = cache_data.get('knowledge_graph', {
                        'movie_to_movie': defaultdict(set),
                        'keyword_to_movie': defaultdict(set),
                        'kb_to_movie': defaultdict(set),
                        'theme_to_movie': defaultdict(set)
                    })
                    self.similarity_threshold = cache_data.get('similarity_threshold', 0.15)
                    # Convert from dict back to defaultdict if needed
                    for key, value in self.knowledge_graph.items():
                        if not isinstance(value, defaultdict):
                            self.knowledge_graph[key] = defaultdict(set, value)
                print("Cache loaded successfully")
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save current state to cache"""
        try:
            # Convert defaultdicts to regular dicts for serialization
            knowledge_graph_dict = {}
            for key, value in self.knowledge_graph.items():
                knowledge_graph_dict[key] = dict(value)
                
            cache_data = {
                'movie_vectors': self.movie_vectors,
                'kb_vectors': self.kb_vectors,
                'movie_kb_relations': dict(self.movie_kb_relations),
                'movie_keywords': self.movie_keywords,
                'kb_keywords': self.kb_keywords,
                'keyword_index': dict(self.keyword_index),
                'genre_mapping': dict(self.genre_mapping),
                'era_mapping': dict(self.era_mapping),
                'theme_mapping': dict(self.theme_mapping),
                'user_interactions': dict(self.user_interactions),
                'knowledge_graph': knowledge_graph_dict,
                'similarity_threshold': self.similarity_threshold
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better vectorization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stop words and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                           if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(processed_tokens)
    
    def _extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract most important keywords from text"""
        # Clean text
        processed_text = self._preprocess_text(text)
        
        # Count word frequencies
        word_counts = Counter(processed_text.split())
        
        # Remove very common words and get top keywords
        most_common = word_counts.most_common(top_n)
        
        # Return just the keywords
        return [word for word, _ in most_common]
    
    def _extract_themes(self, text: str) -> Set[str]:
        """Extract potential themes from text"""
        # List of common movie themes to look for
        common_themes = {
            'space travel', 'time travel', 'adventure', 'war', 'romance', 'mystery',
            'crime', 'fantasy', 'superhero', 'horror', 'coming of age', 'revenge',
            'artificial intelligence', 'dystopia', 'post-apocalyptic', 'cyberpunk',
            'survival', 'conspiracy', 'espionage', 'heist', 'zombie', 'vampire',
            'martial arts', 'historical', 'political', 'satire', 'psychological',
            'thriller', 'family drama', 'documentary', 'biography', 'musical',
            'western', 'noir', 'sports', 'road trip', 'buddy', 'comedy',
            'supernatural', 'science fiction', 'disaster', 'monster', 'alien invasion'
        }
        
        text_lower = text.lower()
        found_themes = set()
        
        # Look for themes in text
        for theme in common_themes:
            if theme in text_lower:
                found_themes.add(theme)
        
        # Also check for partial matches on significant themes
        if 'space' in text_lower and ('planet' in text_lower or 'galaxy' in text_lower or 'astronaut' in text_lower):
            found_themes.add('space travel')
            
        if 'future' in text_lower and ('technology' in text_lower or 'robot' in text_lower):
            found_themes.add('science fiction')
            
        if 'investigat' in text_lower and ('murder' in text_lower or 'crime' in text_lower):
            found_themes.add('mystery')
            
        return found_themes
    
    def _map_year_to_era(self, year: int) -> List[str]:
        """Map a year to film eras for better contextual matching"""
        eras = []
        
        if year < 1930:
            eras.append('silent era')
        if 1930 <= year < 1946:
            eras.append('golden age')
        if 1946 <= year < 1960:
            eras.append('post war')
        if 1960 <= year < 1980:
            eras.append('new hollywood')
        if 1980 <= year < 2000:
            eras.append('blockbuster era')
        if year >= 2000:
            eras.append('digital era')
        if year >= 2010:
            eras.append('streaming era')
        
        return eras
    
    def _extract_genres(self, text: str) -> Set[str]:
        """Extract potential genres from text"""
        common_genres = {
            'action', 'comedy', 'drama', 'horror', 'thriller', 'romance', 'western',
            'documentary', 'animation', 'sci-fi', 'fantasy', 'adventure', 'crime',
            'biography', 'musical', 'history', 'mystery', 'war', 'family', 'sport'
        }
        
        text_lower = text.lower()
        found_genres = set()
        
        # Look for explicit genre mentions
        for genre in common_genres:
            if genre in text_lower:
                found_genres.add(genre)
                
        # Add sci-fi detection
        if 'science fiction' in text_lower or 'sci-fi' in text_lower or 'sci fi' in text_lower:
            found_genres.add('sci-fi')
            
        return found_genres
    
    def process_data(self, movies: List[Dict[str, Any]], kb_docs: Dict[str, str]):
        """Process movie and KB data to build semantic representations and knowledge graph"""
        print(f"Processing {len(movies)} movies and {len(kb_docs)} knowledge base documents...")
        
        # Create corpus for vectorization
        movie_docs = {}
        for movie in movies:
            # Create rich text representation including title, summary, and year
            movie_text = f"{movie['title']} {movie['summary']} {movie['year']}"
            movie_docs[movie['title']] = self._preprocess_text(movie_text)
            
            # Extract and store keywords for this movie
            self.movie_keywords[movie['title']] = self._extract_keywords(movie_text)
            
            # Extract and map themes for this movie
            movie_themes = self._extract_themes(movie_text)
            for theme in movie_themes:
                self.theme_mapping[theme].add(movie['title'])
                # Add to knowledge graph
                self.knowledge_graph['theme_to_movie'][theme].add(movie['title'])
            
            # Map movie to era based on year
            eras = self._map_year_to_era(movie['year'])
            for era in eras:
                self.era_mapping[era].add(movie['title'])
            
            # Extract and map genres
            genres = self._extract_genres(movie_text)
            for genre in genres:
                self.genre_mapping[genre].add(movie['title'])
        
        # Preprocess KB documents
        processed_kb_docs = {}
        for kb_file, content in kb_docs.items():
            processed_kb_docs[kb_file] = self._preprocess_text(content)
            
            # Extract and store keywords for KB docs
            self.kb_keywords[kb_file] = self._extract_keywords(content)
            
            # Extract themes for KB docs and build theme-to-KB mappings
            kb_themes = self._extract_themes(content)
            for theme in kb_themes:
                # Add to knowledge graph
                for movie_title in self.theme_mapping.get(theme, set()):
                    self.knowledge_graph['kb_to_movie'][kb_file].add(movie_title)
        
        # Create a vocabulary index for keyword search
        self._build_keyword_index(movies, kb_docs)
        
        # Build vectorizer and transform documents
        all_docs = list(movie_docs.values()) + list(processed_kb_docs.values())
        
        # Create and fit TF-IDF vectorizer with n-grams for better semantic capture
        self.vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.85, ngram_range=(1, 2), 
            sublinear_tf=True, use_idf=True
        )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_docs)
            
            # Store vector representations
            movie_titles = list(movie_docs.keys())
            for i, title in enumerate(movie_titles):
                self.movie_vectors[title] = tfidf_matrix[i].toarray().flatten()
                
            kb_files = list(processed_kb_docs.keys())
            for i, kb_file in enumerate(kb_files):
                self.kb_vectors[kb_file] = tfidf_matrix[i + len(movie_titles)].toarray().flatten()
                
            # Build initial relationships between movies and KB docs
            self._build_initial_relationships()
            
            # Create movie-to-movie similarities
            self._build_movie_similarities(movie_titles)
            
            # Save updated data
            self._save_cache()
            
            print("Data processing completed successfully")
            
        except Exception as e:
            print(f"Error during vectorization: {e}")
            # Fallback to manual TF-IDF calculation if scikit-learn fails
            self._manual_vectorization(movie_docs, processed_kb_docs)
    
    def _manual_vectorization(self, movie_docs: Dict[str, str], kb_docs: Dict[str, str]):
        """Fallback method for manual TF-IDF calculation if scikit-learn fails"""
        print("Using manual TF-IDF calculation as fallback...")
        
        # Combine all documents
        all_docs = list(movie_docs.values()) + list(kb_docs.values())
        
        # Create vocabulary
        vocabulary = set()
        for doc in all_docs:
            vocabulary.update(doc.split())
        
        # Calculate document frequencies
        doc_freq = {}
        for term in vocabulary:
            doc_freq[term] = sum(1 for doc in all_docs if term in doc.split())
        
        # Calculate IDF
        num_docs = len(all_docs)
        idf = {term: math.log(num_docs / freq) for term, freq in doc_freq.items()}
        
        # Calculate TF-IDF vectors for each document
        movie_titles = list(movie_docs.keys())
        for title in movie_titles:
            doc = movie_docs[title]
            term_freq = Counter(doc.split())
            tfidf_vector = np.zeros(len(vocabulary))
            
            for i, term in enumerate(vocabulary):
                tf = term_freq.get(term, 0) / max(len(doc.split()), 1)
                tfidf_vector[i] = tf * idf.get(term, 0)
                
            self.movie_vectors[title] = tfidf_vector
        
        kb_files = list(kb_docs.keys())
        for kb_file in kb_files:
            doc = kb_docs[kb_file]
            term_freq = Counter(doc.split())
            tfidf_vector = np.zeros(len(vocabulary))
            
            for i, term in enumerate(vocabulary):
                tf = term_freq.get(term, 0) / max(len(doc.split()), 1)
                tfidf_vector[i] = tf * idf.get(term, 0)
                
            self.kb_vectors[kb_file] = tfidf_vector
            
        # Continue with relationship building
        self._build_initial_relationships()
        self._build_movie_similarities(movie_titles)
    
    def _build_keyword_index(self, movies: List[Dict[str, Any]], kb_docs: Dict[str, str]):
        """Build inverted index of keywords to movies for faster search"""
    # Index movie keywords
        for movie in movies:
            title = movie['title']
        
        # Process movie text and extract significant words
            movie_text = f"{title} {movie['summary']}"
            keywords = set(self._preprocess_text(movie_text).split())
        
        # Add movie title to index for each keyword
            for keyword in keywords:
                if len(keyword) > 2:  # Skip very short words
                    self.keyword_index[keyword].add(title)
                # Also add to knowledge graph
                    self.knowledge_graph['keyword_to_movie'][keyword].add(title)
    
    # Index KB document keywords
        for kb_file, content in kb_docs.items():
            keywords = set(self._preprocess_text(content).split())
        
        # For each keyword, find potential movie matches
            for keyword in keywords:
                if keyword in self.keyword_index:
                # For each movie that shares this keyword, strengthen KB-movie relationship
                    for movie_title in self.keyword_index[keyword]:
                    # Make sure movie_title exists in relations dictionary
                        if movie_title not in self.movie_kb_relations:
                            self.movie_kb_relations[movie_title] = []
                        
                    # Add to movie-KB relations if not already there
                        if not any(kb_file == existing_kb for existing_kb, _ in self.movie_kb_relations[movie_title]):
                            self.movie_kb_relations[movie_title].append((kb_file, 0.5))  # Initial relationship strength
    
    def _build_initial_relationships(self):
        """Build initial relationships between movies and KB docs based on cosine similarity"""
        if not self.movie_vectors or not self.kb_vectors:
            print("No vectors available to build relationships")
            return
            
        # For each movie, find the most similar KB docs
        for movie_title, movie_vec in self.movie_vectors.items():
            # Calculate similarity with each KB doc
            similarities = []
            for kb_file, kb_vec in self.kb_vectors.items():
                # Use cosine similarity between vectors
                similarity = self._calculate_cosine_similarity(movie_vec, kb_vec)
                similarities.append((kb_file, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Store top 3 most similar KB docs (if above threshold)
            self.movie_kb_relations[movie_title] = [
                (kb_file, sim) for kb_file, sim in similarities[:3]
                if sim > self.similarity_threshold
            ]
            
            # Update knowledge graph
            for kb_file, sim in similarities[:3]:
                if sim > self.similarity_threshold:
                    self.knowledge_graph['kb_to_movie'][kb_file].add(movie_title)
    
    def _build_movie_similarities(self, movie_titles: List[str]):
        """Build movie-to-movie similarity relationships for collaborative filtering"""
        for i, title1 in enumerate(movie_titles):
            if title1 not in self.movie_vectors:
                continue
                
            # Find similar movies based on vector similarity
            for title2 in movie_titles:
                if title1 != title2 and title2 in self.movie_vectors:
                    similarity = self._calculate_cosine_similarity(
                        self.movie_vectors[title1], 
                        self.movie_vectors[title2]
                    )
                    
                    # If similarity is above threshold, add to knowledge graph
                    if similarity > 0.3:  # Higher threshold for movie-movie similarity
                        self.knowledge_graph['movie_to_movie'][title1].add(title2)
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        # Handle empty vectors
        if np.sum(vec1) == 0 or np.sum(vec2) == 0:
            return 0
            
        # Calculate dot product and magnitudes
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def search_movies(self, search_query: str, minimum_rating: float, movies: List[Dict[str, Any]]) -> List[Tuple[str, float, List[str]]]:
        """
        Enhanced search for movies using hybrid approach:
        1. Keyword matching
        2. Vector similarity
        3. Collaborative filtering from knowledge graph
        """
        print(f"Searching for '{search_query}' with minimum rating {minimum_rating}...")
        
        # Preprocess search query
        processed_query = self._preprocess_text(search_query)
        query_keywords = processed_query.split()
        
        # Check if query matches any known themes
        query_themes = self._extract_themes(search_query)
        
        # Track all matched movies with their scores and matched KB docs
        matched_movies = {}  # {movie_title: {'score': float, 'kb_docs': list}}
        
        # 1. First pass: Direct keyword matching (fastest)
        self._keyword_based_search(query_keywords, query_themes, matched_movies, minimum_rating, movies)
        
        # 2. Second pass: Vector similarity search (more accurate but slower)
        if self.vectorizer:
            self._vector_based_search(processed_query, matched_movies, minimum_rating, movies)
        
        # 3. Third pass: Knowledge graph traversal for collaborative recommendations
        self._knowledge_graph_search(query_keywords, query_themes, matched_movies, minimum_rating, movies)
        
        # Sort results by overall score
        sorted_results = sorted(
            [(title, data['score'], data.get('kb_docs', [])) 
             for title, data in matched_movies.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"Found {len(sorted_results)} movies matching the query")
        return sorted_results
    
    def _keyword_based_search(self, query_keywords: List[str], query_themes: Set[str], 
                             matched_movies: Dict[str, Dict], minimum_rating: float, 
                             movies: List[Dict[str, Any]]):
        """Perform direct keyword matching search"""
        # Find movies by direct keyword matching
        keyword_matched_movies = set()
        
        # Check keyword index for each query keyword
        for keyword in query_keywords:
            if keyword in self.keyword_index:
                keyword_matched_movies.update(self.keyword_index[keyword])
        
        # Check theme mapping for each query theme
        for theme in query_themes:
            if theme in self.theme_mapping:
                keyword_matched_movies.update(self.theme_mapping[theme])
        
        # Calculate initial scores for keyword matched movies
        for title in keyword_matched_movies:
            # Find the movie object to check rating
            movie_obj = next((m for m in movies if m['title'] == title), None)
            if not movie_obj or movie_obj['rating'] < minimum_rating:
                continue
                
            # Count matching keywords
            movie_text = f"{title} {movie_obj['summary']}"
            movie_words = set(self._preprocess_text(movie_text).split())
            
            # Calculate keyword match score (percentage of query keywords found)
            matches = sum(1 for k in query_keywords if k in movie_words)
            keyword_score = matches / max(len(query_keywords), 1)
            
            # Check for theme matches
            movie_themes = self._extract_themes(movie_text)
            theme_matches = len(query_themes.intersection(movie_themes))
            theme_score = theme_matches * 0.2  # Bonus for theme matches
            
            # Combined score with theme bonus
            combined_score = keyword_score + theme_score
            
            # Add to matched movies with score
            matched_movies[title] = {
                'score': combined_score,
                'kb_docs': []  # Will be populated later
            }
    
    def _vector_based_search(self, processed_query: str, matched_movies: Dict[str, Dict], 
                            minimum_rating: float, movies: List[Dict[str, Any]]):
        """Perform vector similarity search"""
        try:
            # Transform query to vector space
            query_vector = self.vectorizer.transform([processed_query]).toarray().flatten()
            
            # Calculate similarity between query and all movie vectors
            for title, movie_vector in self.movie_vectors.items():
                # Skip if already matched with high score
                if title in matched_movies and matched_movies[title]['score'] > 0.8:
                    continue
                    
                # Find the movie object to check rating
                movie_obj = next((m for m in movies if m['title'] == title), None)
                if not movie_obj or movie_obj['rating'] < minimum_rating:
                    continue
                
                # Calculate vector similarity
                similarity = self._calculate_cosine_similarity(query_vector, movie_vector)
                
                # Only consider if similarity is above threshold
                if similarity > self.similarity_threshold:
                    # If movie already matched via keywords, update score
                    if title in matched_movies:
                        # Blend scores, giving more weight to vector similarity
                        matched_movies[title]['score'] = (
                            matched_movies[title]['score'] * 0.3 + similarity * 0.7
                        )
                    else:
                        # Add new match
                        matched_movies[title] = {
                            'score': similarity,
                            'kb_docs': []
                        }
        except Exception as e:
            print(f"Vector search error: {e}")
    
    def _knowledge_graph_search(self, query_keywords: List[str], query_themes: Set[str], 
                               matched_movies: Dict[str, Dict], minimum_rating: float, 
                               movies: List[Dict[str, Any]]):
        """Use knowledge graph for collaborative filtering"""
        # Step 1: Find movies directly related to query keywords
        keyword_movies = set()
        for keyword in query_keywords:
            keyword_movies.update(self.knowledge_graph['keyword_to_movie'].get(keyword, set()))
        
        # Step 2: Find movies related to query themes
        theme_movies = set()
        for theme in query_themes:
            theme_movies.update(self.knowledge_graph['theme_to_movie'].get(theme, set()))
        
        # Step 3: From the movies found, expand to similar movies using movie-to-movie graph
        candidate_movies = keyword_movies.union(theme_movies)
        similar_movies = set()
        
        for movie in candidate_movies:
            similar_movies.update(self.knowledge_graph['movie_to_movie'].get(movie, set()))
        
        # Add all candidate and similar movies (unless already scored higher)
        collaborative_candidates = candidate_movies.union(similar_movies)
        
        for title in collaborative_candidates:
            # Find the movie object to check rating
            movie_obj = next((m for m in movies if m['title'] == title), None)
            if not movie_obj or movie_obj['rating'] < minimum_rating:
                continue
                
            # If not already matched or low-scored, add with collaborative score
            if title not in matched_movies or matched_movies[title]['score'] < 0.3:
                # Direct keyword/theme matches get higher score than similar movies
                if title in candidate_movies:
                    score = 0.7
                else:
                    score = 0.4
                    
                matched_movies[title] = {
                    'score': score,
                    'kb_docs': []
                }
    
    def find_relevant_kb(self, movie_title: str) -> List[Tuple[str, float]]:
        """Find relevant knowledge base documents for a movie with scores"""
        # Check if we have learned relationships
        if movie_title in self.movie_kb_relations and self.movie_kb_relations[movie_title]:
            # Return all relevant KB docs with their scores
            return sorted(self.movie_kb_relations[movie_title], 
                         key=lambda x: x[1], reverse=True)
        
        # If no learned relationships, try semantic matching
        if movie_title not in self.movie_vectors:
            return []
        
        # If no KB vectors, we can't match
        if not self.kb_vectors:
            return []
            
        movie_vector = self.movie_vectors[movie_title]
        
        # Find the most similar KB documents
        similar_kbs = []
        for kb_file, kb_vector in self.kb_vectors.items():
            similarity = self._calculate_cosine_similarity(movie_vector, kb_vector)
            if similarity > self.similarity_threshold:
                similar_kbs.append((kb_file, similarity))
        
        # Sort by similarity score
        similar_kbs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 matches or fewer if not enough
        top_matches = similar_kbs[:3]
        
        # Record these relationships for future learning
        if top_matches:
            self.movie_kb_relations[movie_title] = top_matches
            self._save_cache()
        
        return top_matches
    
    def provide_feedback(self, movie_title: str, kb_file: str, is_relevant: bool, strength: float = 1.0):
        """Learn from user feedback about movie-KB relationships"""
        # Record the interaction
        self.user_interactions[movie_title].append({
            'kb_file': kb_file,
            'is_relevant': is_relevant,
            'strength': strength,
            'count': 1
        })
        
        # Make sure movie_title exists in relations dictionary
        if movie_title not in self.movie_kb_relations:
            self.movie_kb_relations[movie_title] = []
        
        # Update the relationship
        if is_relevant:
            # Find if this relationship already exists
            exists = False
            for i, (existing_kb, score) in enumerate(self.movie_kb_relations[movie_title]):
                if existing_kb == kb_file:
                    # Increase the score based on feedback strength
                    new_score = min(1.0, score + (0.2 * strength))
                    self.movie_kb_relations[movie_title][i] = (existing_kb, new_score)
                    exists = True
                    break
            
            if not exists:
                # Add new relationship with initial score
                initial_score = 0.5 * strength
                self.movie_kb_relations[movie_title].append((kb_file, initial_score))
                
            # Update knowledge graph
            self.knowledge_graph['kb_to_movie'][kb_file].add(movie_title)
        else:
            # If negative feedback, reduce relationship strength
            for i, (existing_kb, score) in enumerate(self.movie_kb_relations[movie_title]):
                if existing_kb == kb_file:
                    # Decrease score
                    new_score = max(0.0, score - (0.3 * strength))
                    
                    # If score drops too low, remove relationship
                    if new_score < 0.1:
                        self.movie_kb_relations[movie_title].pop(i)
                        # Also remove from knowledge graph if present
                        if movie_title in self.knowledge_graph['kb_to_movie'].get(kb_file, set()):
                            self.knowledge_graph['kb_to_movie'][kb_file].remove(movie_title)
                    else:
                        self.movie_kb_relations[movie_title][i] = (existing_kb, new_score)
                    break
        
        # Learn from feedback to optimize threshold
        self._adjust_threshold_from_feedback()
        
        # Save updated data
        self._save_cache()
    
    def _adjust_threshold_from_feedback(self):
        """Adapt similarity threshold based on feedback history"""
        # Collect positive and negative feedback samples
        positive_scores = []
        negative_scores = []
        
        # Analyze all user interactions
        for movie, interactions in self.user_interactions.items():
            if movie not in self.movie_vectors:
                continue
            
            movie_vector = self.movie_vectors[movie]
            
            for interaction in interactions:
                kb_file = interaction['kb_file']
                if kb_file not in self.kb_vectors:
                    continue
                
                kb_vector = self.kb_vectors[kb_file]
                similarity = self._calculate_cosine_similarity(movie_vector, kb_vector)
                
                if interaction['is_relevant']:
                    positive_scores.append(similarity)
                else:
                    negative_scores.append(similarity)
        
        # Only adjust if we have enough data
        if len(positive_scores) >= 3 and len(negative_scores) >= 2:
            # Set threshold between average positive and negative scores
            avg_positive = sum(positive_scores) / len(positive_scores)
            avg_negative = sum(negative_scores) / len(negative_scores)
            
            # New threshold is 80% of the way from negative to positive average
            new_threshold = avg_negative + 0.8 * (avg_positive - avg_negative)
            
            # Apply with dampening to avoid drastic changes
            self.similarity_threshold = 0.7 * self.similarity_threshold + 0.3 * new_threshold
    
    def analyze_keyword_relevance(self, query: str, movies: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze which movies are most relevant to specific keywords in the query"""
        # Preprocess and extract keywords from query
        processed_query = self._preprocess_text(query)
        query_keywords = processed_query.split()
        
        # Track relevance of each keyword
        keyword_to_movies = {}
        
        # For each keyword, find relevant movies
        for keyword in query_keywords:
            if len(keyword) < 3:  # Skip very short words
                continue
                
            relevant_movies = []
            
            # Check direct index matches
            if keyword in self.keyword_index:
                for title in self.keyword_index[keyword]:
                    # Find the movie object
                    movie_obj = next((m for m in movies if m['title'] == title), None)
                    if movie_obj:
                        relevant_movies.append(title)
            
            # Check knowledge graph
            if keyword in self.knowledge_graph['keyword_to_movie']:
                for title in self.knowledge_graph['keyword_to_movie'][keyword]:
                    movie_obj = next((m for m in movies if m['title'] == title), None)
                    if movie_obj and title not in relevant_movies:
                        relevant_movies.append(title)
            
            # Only add keywords with matches
            if relevant_movies:
                keyword_to_movies[keyword] = relevant_movies
        
        return keyword_to_movies
    
    def get_theme_recommendations(self, theme: str, minimum_rating: float, movies: List[Dict[str, Any]]) -> List[str]:
        """Get movie recommendations based on a specific theme"""
        recommended_movies = []
        
        # First check theme mapping
        if theme in self.theme_mapping:
            for title in self.theme_mapping[theme]:
                movie_obj = next((m for m in movies if m['title'] == title), None)
                if movie_obj and movie_obj['rating'] >= minimum_rating:
                    recommended_movies.append(title)
        
        # Then check knowledge graph
        if theme in self.knowledge_graph['theme_to_movie']:
            for title in self.knowledge_graph['theme_to_movie'][theme]:
                movie_obj = next((m for m in movies if m['title'] == title), None)
                if movie_obj and movie_obj['rating'] >= minimum_rating and title not in recommended_movies:
                    recommended_movies.append(title)
        
        return recommended_movies


# Main search function that uses the enhanced engine
def enhanced_search_movies(search_string: str, minimum_rating: float) -> List[Dict[str, Any]]:
    """Main function to find movies matching search criteria and their relevant KB docs"""
    # Paths - handle both script execution and notebook environments
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # For Jupyter notebooks and environments where __file__ is not defined
        current_dir = os.getcwd()
    
    movies_path = os.path.join(current_dir, "data", "movies.json")
    kb_dir = os.path.join(current_dir, "data", "gk")
    
    # Load data
    movies = load_movies_data(movies_path)
    kb_docs = load_kb_documents(kb_dir)
    
    # Initialize the enhanced engine
    engine = EnhancedMovieKnowledgeEngine()
    
    # Process data
    engine.process_data(movies, kb_docs)
    
    # Search for matching movies with enhanced algorithm
    matching_movies = engine.search_movies(search_string, minimum_rating, movies)
    
    # Print some debug info
    print(f"\nFound {len(matching_movies)} matching movies for '{search_string}' with rating >= {minimum_rating}\n")
    
    # Get keyword analysis
    keyword_analysis = engine.analyze_keyword_relevance(search_string, movies)
    if keyword_analysis:
        print("\nKeyword analysis:")
        for keyword, movie_titles in keyword_analysis.items():
            print(f"- '{keyword}' matches: {', '.join(movie_titles[:3])}" + 
                 (f" and {len(movie_titles)-3} more" if len(movie_titles) > 3 else ""))
    
    # Extract themes from search
    search_themes = engine._extract_themes(search_string)
    if search_themes:
        print("\nDetected themes:")
        for theme in search_themes:
            theme_movies = engine.get_theme_recommendations(theme, minimum_rating, movies)
            print(f"- '{theme}' matches: {len(theme_movies)} movies")
    
    # Find relevant KB documents for each movie and create result structure
    results = []
    for movie_title, score, _ in matching_movies:
        # Get relevant KB documents with scores
        kb_results = engine.find_relevant_kb(movie_title)
        
        # Find the movie object
        movie_obj = next((m for m in movies if m['title'] == movie_title), None)
        
        if movie_obj:
            # Create a result dictionary with all movie details
            result = {
                'title': movie_obj['title'],
                'rating': movie_obj['rating'],
                'year': movie_obj['year'],
                'summary': movie_obj['summary'],
                'search_score': score,
                'knowledge_base': [kb_file for kb_file, _ in kb_results] if kb_results else ["NA"]
            }
            results.append(result)
    
    # Save the engine state for future use
    engine._save_cache()
    
    return results

# Helper function to load movies data
def load_movies_data(filepath: str) -> List[Dict[str, Any]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading movies data: {e}")
        return []

# Helper function to save movies data
def save_movies_data(filepath: str, movies: List[Dict[str, Any]]) -> bool:
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(movies, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving movies data: {e}")
        return False

# Helper function to load knowledge base documents
def load_kb_documents(directory: str) -> Dict[str, str]:
    kb_docs = {}
    try:
        for file_path in Path(directory).glob("*.md"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                kb_docs[file_path.name] = content
    except Exception as e:
        print(f"Error loading knowledge base documents: {e}")
    return kb_docs

def print_movie_details(movies: List[Dict[str, Any]]):
    """Print detailed information about movies"""
    print("\nResults:")
    print("=" * 80)
    
    for i, movie in enumerate(movies, 1):
        print(f"Movie #{i}: {movie['title']}")
        print(f"Rating: {movie['rating']}")
        print(f"Year: {movie['year']}")
        print(f"Search Relevance: {movie.get('search_score', 'N/A'):.2f}")
        print(f"Summary: {movie['summary']}")
        print(f"Knowledge Base: {', '.join(movie['knowledge_base'])}")
        print("-" * 80)

def add_new_movie() -> Dict[str, Any]:
    """Function to gather information for a new movie from user input"""
    print("\nAdding a new movie:")
    print("=" * 80)
    
    title = input("Enter movie title: ").strip()
    
    # Validate rating input
    while True:
        try:
            rating = float(input("Enter rating (0-10): ").strip())
            if 0 <= rating <= 10:
                break
            else:
                print("Rating must be between 0 and 10")
        except ValueError:
            print("Please enter a valid number for rating")
    
    # Validate year input
    while True:
        try:
            year = int(input("Enter release year: ").strip())
            if 1888 <= year <= 2030:  # Reasonable range for movies
                break
            else:
                print("Year must be a valid movie release year (1888-2030)")
        except ValueError:
            print("Please enter a valid year")
    
    summary = input("Enter movie summary: ").strip()
    
    # Create new movie object
    new_movie = {
        "title": title,
        "rating": rating,
        "year": year,
        "summary": summary
    }
    
    return new_movie

def save_new_movie(movie: Dict[str, Any]) -> bool:
    """Save a new movie to the movies.json file"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    
    movies_path = os.path.join(current_dir, "data", "movies.json")
    
    # Load existing movies
    movies = load_movies_data(movies_path)
    
    # Check if movie with same title already exists
    if any(m['title'] == movie['title'] for m in movies):
        print(f"\nA movie with title '{movie['title']}' already exists!")
        return False
    
    # Add new movie
    movies.append(movie)
    
    # Save updated movie list
    if save_movies_data(movies_path, movies):
        print(f"\nMovie '{movie['title']}' added successfully!")
        return True
    else:
        print("\nFailed to save the new movie.")
        return False

def main():
    """Main function to run the enhanced movie search and management system"""
    print("\n==== Enhanced Movie Search and Knowledge System ====\n")
    
    while True:
        print("\nMenu:")
        print("1. Search Movies")
        print("2. Search by Theme")
        print("3. Add Movie")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            search_term = input("\nEnter search term: ").strip()
            
            # Validate minimum rating input
            while True:
                try:
                    min_rating = float(input("Enter minimum rating (0-10): ").strip())
                    if 0 <= min_rating <= 10:
                        break
                    else:
                        print("Rating must be between 0 and 10")
                except ValueError:
                    print("Please enter a valid number for rating")
            
            # Search movies with enhanced algorithm
            results = enhanced_search_movies(search_term, min_rating)
            
            # Display results
            if results:
                print_movie_details(results)
            else:
                print("\nNo movies found matching your criteria.")
        
        elif choice == '2':
            print("\nAvailable themes: space travel, time travel, adventure, war, romance, mystery, etc.")
            theme = input("Enter theme to search for: ").strip().lower()
            
            # Validate minimum rating input
            while True:
                try:
                    min_rating = float(input("Enter minimum rating (0-10): ").strip())
                    if 0 <= min_rating <= 10:
                        break
                    else:
                        print("Rating must be between 0 and 10")
                except ValueError:
                    print("Please enter a valid number for rating")
            
            # Initialize engine
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                current_dir = os.getcwd()
            
            movies_path = os.path.join(current_dir, "data", "movies.json")
            kb_dir = os.path.join(current_dir, "data", "gk")
            
            movies = load_movies_data(movies_path)
            kb_docs = load_kb_documents(kb_dir)
            
            engine = EnhancedMovieKnowledgeEngine()
            engine.process_data(movies, kb_docs)
            
            # Get movies for this theme
            theme_movies = engine.get_theme_recommendations(theme, min_rating, movies)
            
            if theme_movies:
                results = []
                for title in theme_movies:
                    movie_obj = next((m for m in movies if m['title'] == title), None)
                    if movie_obj:
                        # Get relevant KB documents
                        kb_results = engine.find_relevant_kb(title)
                        
                        result = {
                            'title': movie_obj['title'],
                            'rating': movie_obj['rating'],
                            'year': movie_obj['year'],
                            'summary': movie_obj['summary'],
                            'knowledge_base': [kb_file for kb_file, _ in kb_results] if kb_results else ["NA"]
                        }
                        results.append(result)
                
                print_movie_details(results)
            else:
                print(f"\nNo movies found with theme '{theme}' and minimum rating {min_rating}.")
                
        elif choice == '3':
            # Add a new movie
            new_movie = add_new_movie()
            save_new_movie(new_movie)
            
        elif choice == '4':
            print("\nThank you for using the Enhanced Movie Search System. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
