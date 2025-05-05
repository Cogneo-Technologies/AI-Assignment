import json
import os
import re
from pathlib import Path
import math
import pickle
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

class MovieKnowledgeEngine:
    """A learning engine that dynamically creates relationships between movies and knowledge base documents"""
    
    def __init__(self, cache_file="movie_kb_cache.pkl"):
        self.cache_file = cache_file
        self.movie_vectors = {}
        self.kb_vectors = {}
        self.movie_kb_relations = defaultdict(list)
        self.idf = {}
        self.feedback_data = {}
        self.similarity_threshold = 0.01  # Initial threshold, will be adjusted based on feedback
        
        # Load cached data if available
        self._load_cache()
    
    def _load_cache(self):
        """Load previously cached data if available"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.movie_vectors = cache_data.get('movie_vectors', {})
                    self.kb_vectors = cache_data.get('kb_vectors', {})
                    self.movie_kb_relations = cache_data.get('movie_kb_relations', defaultdict(list))
                    self.idf = cache_data.get('idf', {})
                    self.feedback_data = cache_data.get('feedback_data', {})
                    self.similarity_threshold = cache_data.get('similarity_threshold', 0.01)
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save current state to cache"""
        try:
            cache_data = {
                'movie_vectors': self.movie_vectors,
                'kb_vectors': self.kb_vectors,
                'movie_kb_relations': dict(self.movie_kb_relations),
                'idf': self.idf,
                'feedback_data': self.feedback_data,
                'similarity_threshold': self.similarity_threshold
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def process_data(self, movies: List[Dict[str, Any]], kb_docs: Dict[str, str]):
        """Process movie and KB data to build semantic representations"""
        # Create document corpus
        movie_docs = {movie['title']: f"{movie['title']} {movie['summary']}" for movie in movies}
        
        # Compute TF-IDF representations
        all_docs = list(movie_docs.values()) + list(kb_docs.values())
        doc_tfs, self.idf = self._calculate_tf_idf(all_docs)
        
        # Store vector representations
        movie_titles = list(movie_docs.keys())
        for i, title in enumerate(movie_titles):
            self.movie_vectors[title] = self._calculate_tfidf_vector(doc_tfs[i], self.idf)
        
        kb_files = list(kb_docs.keys())
        for i, kb_file in enumerate(kb_files):
            self.kb_vectors[kb_file] = self._calculate_tfidf_vector(doc_tfs[i + len(movie_titles)], self.idf)
        
        # If we have feedback data, use it to learn relationships
        self._learn_from_feedback()
        
        # Save updated data
        self._save_cache()
    
    def _calculate_tf_idf(self, documents: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """Calculate TF-IDF representations for a corpus of documents"""
        # Tokenize and create term frequency for each document
        doc_tfs = []
        doc_freq = {}  # Document frequency of terms
        
        for doc in documents:
            # Enhanced tokenization with n-grams for better semantic capture
            tokens = self._tokenize(doc)
            
            # Calculate term frequency for this document
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            
            # Update document frequency
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
            
            doc_tfs.append(tf)
        
        # Calculate IDF for all terms
        num_docs = len(documents)
        idf = {term: math.log(num_docs / freq) for term, freq in doc_freq.items()}
        
        return doc_tfs, idf
    
    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with unigrams and bigrams"""
        # Convert to lowercase and tokenize
        tokens = re.findall(r'\w+', text.lower())
        
        # Add unigrams
        result = tokens.copy()
        
        # Add bigrams (useful for capturing phrases)
        if len(tokens) > 1:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            result.extend(bigrams)
        
        return result
    
    def _calculate_tfidf_vector(self, tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
        """Calculate TF-IDF vector for a document"""
        return {term: freq * idf.get(term, 0) for term, freq in tf.items()}
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Calculate dot product
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)
    
    def search_movies(self, search_query: str, minimum_rating: float, movies: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Search for movies matching the query with a minimum rating"""
        # Tokenize search query
        query_tokens = self._tokenize(search_query)
        
        # Calculate TF for search query
        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
        
        # Calculate TF-IDF vector for search query
        query_tfidf = self._calculate_tfidf_vector(query_tf, self.idf)
        
        # Find matching movies
        results = []
        for movie in movies:
            # Check if movie meets minimum rating
            if movie['rating'] < minimum_rating:
                continue
            
            title = movie['title']
            # If we've seen this movie before, use its vector
            if title in self.movie_vectors:
                movie_tfidf = self.movie_vectors[title]
            else:
                # Otherwise, calculate it on the fly
                movie_doc = f"{title} {movie['summary']}"
                movie_tokens = self._tokenize(movie_doc)
                movie_tf = {}
                for token in movie_tokens:
                    movie_tf[token] = movie_tf.get(token, 0) + 1
                movie_tfidf = self._calculate_tfidf_vector(movie_tf, self.idf)
                # Store for future use
                self.movie_vectors[title] = movie_tfidf
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_tfidf, movie_tfidf)
            
            # If similarity is above threshold, consider it a match
            if similarity > self.similarity_threshold:
                results.append((title, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def find_relevant_kb(self, movie_title: str) -> Optional[str]:
        """Find relevant knowledge base document for a movie"""
        # Check if we have a learned relationship
        if movie_title in self.movie_kb_relations and self.movie_kb_relations[movie_title]:
            # Return the most frequently associated KB doc
            return max(self.movie_kb_relations[movie_title], key=lambda x: x[1])[0]
        
        # If no learned relationship, use semantic matching
        if movie_title not in self.movie_vectors:
            return None
        
        # If no KB vectors, we can't match
        if not self.kb_vectors:
            return None
            
        movie_tfidf = self.movie_vectors[movie_title]
        
        # Find the most similar KB document
        max_similarity = -1
        best_kb_file = None
        
        for kb_file, kb_tfidf in self.kb_vectors.items():
            similarity = self._cosine_similarity(movie_tfidf, kb_tfidf)
            if similarity > max_similarity:
                max_similarity = similarity
                best_kb_file = kb_file
        
        # If similarity is too low, return None
        # Using a lower threshold for broader matching
        if max_similarity < 0.01:  
            return None
        
        # Make sure movie_title exists in relations dictionary
        if movie_title not in self.movie_kb_relations:
            self.movie_kb_relations[movie_title] = []
            
        # Record this relationship for future learning
        self.movie_kb_relations[movie_title].append((best_kb_file, 1))
        self._save_cache()
        
        return best_kb_file
    
    def provide_feedback(self, movie_title: str, kb_file: str, is_relevant: bool):
        """Allow the system to learn from user feedback"""
        # Record the feedback
        if movie_title not in self.feedback_data:
            self.feedback_data[movie_title] = []
        
        self.feedback_data[movie_title].append((kb_file, is_relevant))
        
        # Make sure movie_title exists in relations dictionary
        if movie_title not in self.movie_kb_relations:
            self.movie_kb_relations[movie_title] = []
        
        # Update the relationship
        if is_relevant:
            # Find if this relationship already exists
            exists = False
            for i, (existing_kb, count) in enumerate(self.movie_kb_relations[movie_title]):
                if existing_kb == kb_file:
                    # Increase the count
                    self.movie_kb_relations[movie_title][i] = (existing_kb, count + 1)
                    exists = True
                    break
            
            if not exists:
                # Add new relationship
                self.movie_kb_relations[movie_title].append((kb_file, 1))
        elif self.movie_kb_relations[movie_title]:  # Only try to remove if there are items
            # Remove or decrease count for this relationship
            for i, (existing_kb, count) in enumerate(self.movie_kb_relations[movie_title]):
                if existing_kb == kb_file:
                    if count > 1:
                        self.movie_kb_relations[movie_title][i] = (existing_kb, count - 1)
                    else:
                        self.movie_kb_relations[movie_title].pop(i)
                    break
        
        # Learn from feedback to adjust similarity threshold
        self._learn_from_feedback()
        
        # Save updated data
        self._save_cache()
    
    def _learn_from_feedback(self):
        """Learn from feedback data to improve matching"""
        if not self.feedback_data:
            return
        
        # Calculate optimal threshold based on feedback
        positive_similarities = []
        negative_similarities = []
        
        for movie_title, feedbacks in self.feedback_data.items():
            if movie_title not in self.movie_vectors:
                continue
                
            movie_vec = self.movie_vectors[movie_title]
            
            for kb_file, is_relevant in feedbacks:
                if kb_file not in self.kb_vectors:
                    continue
                    
                kb_vec = self.kb_vectors[kb_file]
                similarity = self._cosine_similarity(movie_vec, kb_vec)
                
                if is_relevant:
                    positive_similarities.append(similarity)
                else:
                    negative_similarities.append(similarity)
        
        # If we have enough data, adjust threshold
        if positive_similarities and negative_similarities:
            # Simple approach: set threshold to midpoint between average positive and negative similarities
            avg_positive = sum(positive_similarities) / len(positive_similarities)
            avg_negative = sum(negative_similarities) / len(negative_similarities)
            
            # Adjust threshold
            self.similarity_threshold = (avg_positive + avg_negative) / 2
        elif positive_similarities:
            # If only positive feedback, lower threshold slightly from minimum positive
            min_positive = min(positive_similarities)
            self.similarity_threshold = max(0.01, min_positive * 0.9)

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

def search_movies(search_string: str, minimum_rating: float) -> List[Dict[str, Any]]:
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
    
    # Initialize the engine
    engine = MovieKnowledgeEngine()
    
    # Process data
    engine.process_data(movies, kb_docs)
    
    # Search for matching movies
    matching_movies = engine.search_movies(search_string, minimum_rating, movies)
    
    # Print some debug info
    print(f"\nFound {len(matching_movies)} matching movies for '{search_string}' with rating >= {minimum_rating}\n")
    
    # Find relevant KB documents for each movie and create result structure
    results = []
    for movie_title, score in matching_movies:
        kb_file = engine.find_relevant_kb(movie_title)
        
        # If no specific KB file found but we have KB docs, try to find one based on genre/theme matching
        if kb_file is None and kb_docs:
            # Find the movie object
            movie_obj = next((m for m in movies if m['title'] == movie_title), None)
            
            if movie_obj:
                # Try to find KB documents with keywords from the movie summary
                summary_words = set(re.findall(r'\b\w{5,}\b', movie_obj['summary'].lower()))
                
                # Find KB document with highest keyword match
                max_matches = 0
                best_match = None
                
                for file_name, content in kb_docs.items():
                    content_lower = content.lower()
                    matches = sum(1 for word in summary_words if word in content_lower)
                    
                    if matches > max_matches:
                        max_matches = matches
                        best_match = file_name
                
                # If we found a good match, use it and add to engine memory
                if max_matches >= 1:
                    kb_file = best_match
                    # Let the engine learn this relationship
                    engine.provide_feedback(movie_title, kb_file, True)
        
        # Find the full movie object
        movie_obj = next((m for m in movies if m['title'] == movie_title), None)
        
        if movie_obj:
            # Create a result dictionary with all movie details
            result = {
                'title': movie_obj['title'],
                'rating': movie_obj['rating'],
                'year': movie_obj['year'],
                'summary': movie_obj['summary'],
                'knowledge_base': kb_file if kb_file else "NA"
            }
            results.append(result)
    
    # Save the engine state for future use
    engine._save_cache()
    
    return results

def print_movie_details(movies: List[Dict[str, Any]]):
    """Print detailed information about movies"""
    print("\nResults:")
    print("=" * 80)
    
    for i, movie in enumerate(movies, 1):
        print(f"Movie #{i}: {movie['title']}")
        print(f"Rating: {movie['rating']}")
        print(f"Year: {movie['year']}")
        print(f"Summary: {movie['summary']}")
        print(f"Knowledge Base: {movie['knowledge_base']}")
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
    """Main function to run the movie search and management system"""
    print("\n==== Movie Search and Management System ====\n")
    
    while True:
        print("\nMenu:")
        print("1. Search Movies")
        print("2. Add Movie")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
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
            
            # Search movies
            results = search_movies(search_term, min_rating)
            
            # Display results
            if results:
                print_movie_details(results)
            else:
                print("\nNo movies found matching your criteria.")
                
        elif choice == '2':
            # Add a new movie
            new_movie = add_new_movie()
            save_new_movie(new_movie)
            
        elif choice == '3':
            print("\nThank you for using the Movie Search System. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
