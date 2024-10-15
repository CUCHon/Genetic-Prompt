import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter
from nltk import ngrams
from typing import List, Tuple
from itertools import chain, islice
import random
class SampleManager:
    def __init__(self, model_name='stsb-roberta-large'):
        self.model = SentenceTransformer(model_name)
        self.samples = []
        self.encoded_samples = None
        self.index = None
        self.checked_pairs = set()

    def add_samples(self, new_samples):
        new_encoded = self.model.encode(new_samples).astype('float32')
        if self.encoded_samples is None:
            self.encoded_samples = new_encoded
        else:
            self.encoded_samples = np.vstack((self.encoded_samples, new_encoded))
        self.samples.extend(new_samples)
        self._update_index()
    
    def _update_index(self):
        n, d = self.encoded_samples.shape
        faiss.normalize_L2(self.encoded_samples)
        self.index = faiss.IndexFlatIP(d)  # Inner Product index for cosine similarity
        self.index.add(self.encoded_samples)

        # Precompute distance matrix as cosine distances, adjusting sign for max distance search
        self.distance_matrix = 1 - np.dot(self.encoded_samples, self.encoded_samples.T)
        np.fill_diagonal(self.distance_matrix, -np.inf)  # Set diagonal to negative infinity



    def find_least_similar_pair(self):
        if self.distance_matrix is None or len(self.samples) < 2:
            return None, None

        # Mask already checked pairs by setting their distances to negative infinity
        for i, j in self.checked_pairs:
            self.distance_matrix[i, j] = -np.inf
            self.distance_matrix[j, i] = -np.inf  # Since the matrix is symmetric

        # Find the maximum distance in the matrix
        max_distance = np.max(self.distance_matrix)
        max_index = np.argmax(self.distance_matrix)
        
        # Convert the flat index back to 2D index
        n = self.distance_matrix.shape[0]
        i, j = divmod(max_index, n)

        # Track this pair as checked
        self.checked_pairs.add((i, j))

        return (i, j), max_distance
        
    '''
        
    def _update_index(self):
        n, d = self.encoded_samples.shape
        faiss.normalize_L2(self.encoded_samples)
        
        # Use GPU resources for FAISS
        res = faiss.StandardGpuResources()  # use a single GPU
        
        # Using a flat inner product index (good for cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(d)
        
        # Make it a GPU index
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(self.encoded_samples)
    
    
    def find_least_similar_pair(self, exclude_checked=False):
        n = len(self.samples)
        min_ip = float('inf')
        least_similar_pair = None

        for i in range(n):
            # Search with k = n, which includes all other samples
            _, I = self.index.search(self.encoded_samples[i:i+1], n)
            for j in I[0][::-1]:  # Start from the end to get least similar first
                if i != j:
                    pair = tuple(sorted((i, j)))
                    if exclude_checked and pair in self.checked_pairs:
                        continue
                    ip = np.dot(self.encoded_samples[i], self.encoded_samples[j])
                    if ip < min_ip:
                        min_ip = ip
                        least_similar_pair = pair
                        break  # Found the least similar for this i, move to next i

        if least_similar_pair:
            self.checked_pairs.add(least_similar_pair)
            cosine_distance = 1 - min_ip
            return least_similar_pair, cosine_distance
        return None, None
    
    def find_least_similar_pair(self, exclude_checked=False):
        n = len(self.samples)
        min_ip = float('inf')
        least_similar_pair = None

        for i in range(n):
            _, I = self.index.search(self.encoded_samples[i:i+1], n)
            for j in I[0][::-1]:  # Start from the end to get least similar first
                if i != j:
                    pair = tuple(sorted((i, j)))
                    if exclude_checked and pair in self.checked_pairs:
                        continue
                    ip = np.dot(self.encoded_samples[i], self.encoded_samples[j])
                    if ip < min_ip:
                        min_ip = ip
                        least_similar_pair = pair
                        break  # Found the least similar for this i, move to next i

        if least_similar_pair:
            self.checked_pairs.add(least_similar_pair)
            cosine_distance = 1 - min_ip
            return least_similar_pair, cosine_distance
        return None, None
    '''
    def get_sample(self, index):
        return self.samples[index]
    
    def get_random_sample(self):
        """
        Get a random sample from the existing sample set.

        :return: A randomly selected sample.
        :raises IndexError: If there are no samples in the set.
        """
        if not self.samples:
            raise IndexError("Cannot get a random sample from an empty set.")
        return random.choice(self.samples)    
    def calculate_diversity_scores(self, n_gram_range: Tuple[int, int] = (1, 3)) -> dict:
        """
        Calculate diversity scores using APS and INGF.
        
        :param n_gram_range: Range of n-grams to consider for INGF, default is (1, 3)
        :return: Dictionary containing APS and INGF scores
        """
        aps_score = self._calculate_aps()
        ingf_score = self._calculate_ingf(n_gram_range)
        
        return {
            "APS": aps_score,
            "INGF": ingf_score
        }
    '''
    def _calculate_aps(self) -> float:
        """
        Calculate the Average Pairwise Similarity (APS) score.
        
        :return: APS score
        """
        n = len(self.samples)
        if n < 2:
            return 0.0  # Not enough samples to calculate similarity

        total_similarity = 0.0
        pair_count = 0

        for i in range(n):
            for j in range(i+1, n):
                similarity = np.dot(self.encoded_samples[i], self.encoded_samples[j])
                total_similarity += similarity
                pair_count += 1

        return total_similarity / pair_count if pair_count > 0 else 0.0
    '''
    
    def _calculate_aps(self) -> float:
        if len(self.encoded_samples) < 2:
            return 0.0

        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(self.encoded_samples, self.encoded_samples.T)
        np.fill_diagonal(similarity_matrix, 0)  # Zero diagonal

        # Only upper triangle needed since matrix is symmetric
        total_similarity = np.sum(np.triu(similarity_matrix, 1))
        pair_count = len(self.encoded_samples) * (len(self.encoded_samples) - 1) / 2

        return total_similarity / pair_count if pair_count > 0 else 0.0
    '''
    def _calculate_ingf(self, n_gram_range: Tuple[int, int]) -> float:
        """
        Calculate the Inter-sample N-gram Frequency (INGF) score.
        
        :param n_gram_range: Range of n-grams to consider
        :return: INGF score
        """
        all_ngrams = []
        for sample in self.samples:
            sample_ngrams = []
            for n in range(n_gram_range[0], n_gram_range[1] + 1):
                sample_ngrams.extend(ngrams(sample.split(), n))
            all_ngrams.extend(sample_ngrams)

        ngram_counts = Counter(all_ngrams)
        total_ngrams = sum(ngram_counts.values())
        
        ingf_score = sum((count / total_ngrams) ** 2 for count in ngram_counts.values())
        return ingf_score
    '''
    def _calculate_ingf(self, n_gram_range: Tuple[int, int]) -> float:
        """
        Calculate the Inter-sample N-gram Frequency (INGF) score using optimized methods.
        
        :param n_gram_range: Range of n-grams to consider
        :return: INGF score
        """
        def ngrams(seq, n):
            """ Generate n-grams from a list of tokens. """
            return zip(*(islice(seq, i, None) for i in range(n)))

        def get_ngrams(sample, n_gram_range):
            """ Extract all n-grams for a given range from a sample. """
            tokens = sample.split()
            return chain.from_iterable(ngrams(tokens, n) for n in range(n_gram_range[0], n_gram_range[1] + 1))

        # Flatten list of ngrams from all samples
        all_ngrams = list(chain.from_iterable(get_ngrams(sample, n_gram_range) for sample in self.samples))
        
        # Count frequencies of each ngram
        ngram_counts = Counter(all_ngrams)
        total_ngrams = sum(ngram_counts.values())
        
        # Calculate INGF score using NumPy for faster computation
        count_array = np.array(list(ngram_counts.values()))
        ingf_score = np.sum((count_array / total_ngrams) ** 2)
        
        return ingf_score
if __name__ == '__main__':
    # Usage example
    manager = SampleManager()

    # Add initial samples
    initial_samples = ["Sample 1", "Sample 2", "Sample 3"]
    manager.add_samples(initial_samples)

    # Find least similar pair
    pair, distance = manager.find_least_similar_pair()
    print(f"Least similar pair: {pair}, Distance: {distance}")
    print(f"Sample 1: {manager.get_sample(pair[0])}")
    print(f"Sample 2: {manager.get_sample(pair[1])}")

    # Add new samples
    new_samples = ["New sample 1", "New sample 2"]
    manager.add_samples(new_samples)

    # Find least similar pair again, excluding checked pairs
    pair, distance = manager.find_least_similar_pair(exclude_checked=True)
    print(f"New least similar pair: {pair}, Distance: {distance}")
    print(f"Sample 1: {manager.get_sample(pair[0])}")
    print(f"Sample 2: {manager.get_sample(pair[1])}")

    # Find least similar pair without excluding checked pairs
    pair, distance = manager.find_least_similar_pair(exclude_checked=False)
    print(f"Overall least similar pair: {pair}, Distance: {distance}")
    print(f"Sample 1: {manager.get_sample(pair[0])}")
    print(f"Sample 2: {manager.get_sample(pair[1])}")