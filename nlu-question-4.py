import math
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Function to clean text and extract words with 3+ characters
def tokenize(text):
    text = text.lower().strip()
    return re.findall(r'\b[a-z]{3,}\b', text)

class TFIDFModel:
    def __init__(self):
        # Initialize storage for IDF values, word list, and document count
        self.idf = {}
        self.vocabulary = []
        self.doc_count = 0

    def fit(self, documents):
        # Calculate document frequencies and build the vocabulary
        self.doc_count = len(documents)
        df = Counter()
        all_words = set()
        for doc in documents:
            words = set(tokenize(doc))
            for word in words:
                df[word] += 1
                all_words.add(word)
        
        # Sort vocabulary for consistent vector indexing
        self.vocabulary = sorted(list(all_words))
        
        # Calculate Inverse Document Frequency (IDF) for each word
        for word, count in df.items():
            self.idf[word] = math.log(self.doc_count / (1 + count))

    def transform(self, doc):
        # Convert text into a numerical vector using TF * IDF
        tokens = tokenize(doc)
        tf = Counter(tokens)
        return [tf.get(word, 0) * self.idf.get(word, 0) for word in self.vocabulary]

class NaiveBayes:
    def train(self, docs, labels):
        # Initialize class metadata, vocabulary, and frequency counters
        self.classes = list(set(labels))
        self.vocab = set()
        self.word_counts = {c: Counter() for c in self.classes}
        self.total_words = {c: 0 for c in self.classes}
        self.priors = {c: labels.count(c)/len(labels) for c in self.classes}
        
        # Aggregate word counts per category
        for doc, label in zip(docs, labels):
            tokens = tokenize(doc)
            for t in tokens:
                self.word_counts[label][t] += 1
                self.total_words[label] += 1
                self.vocab.add(t)

    def predict(self, text):
        # Calculate log-probabilities for each class to predict the label
        tokens = tokenize(text)
        results = {}
        for c in self.classes:
            score = math.log(self.priors[c])
            for t in tokens:
                if t in self.vocab:
                    # Apply Laplace smoothing to handle unseen words
                    prob = (self.word_counts[c][t] + 1) / (self.total_words[c] + len(self.vocab))
                    score += math.log(prob)
            results[c] = score
        # Return the class with the highest probability score
        return max(results, key=results.get)

class KNN:
    def __init__(self, k=3):
        # Set the number of neighbors to consider
        self.k = k

    def train(self, vectors, labels):
        # Store training vectors and their corresponding labels
        self.train_vectors = vectors
        self.train_labels = labels

    def _euclidean_dist(self, v1, v2):
        # Calculate the geometric distance between two vectors
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def predict(self, vec):
        # Calculate distances to all training points and find the closest k
        dists = [(self._euclidean_dist(vec, tv), tl) for tv, tl in zip(self.train_vectors, self.train_labels)]
        dists.sort(key=lambda x: x[0])
        neighbors = [label for dist, label in dists[:self.k]]
        # Return the most common label among the k-nearest neighbors
        return Counter(neighbors).most_common(1)[0][0]

class CosineClassifier:
    def train(self, vectors, labels):
        # Calculate the average vector (centroid) for each class
        self.centroids = {}
        for label in set(labels):
            class_vecs = [v for v, l in zip(vectors, labels) if l == label]
            self.centroids[label] = [sum(col)/len(col) for col in zip(*class_vecs)]

    def _cosine_sim(self, v1, v2):
        # Calculate similarity based on the cosine of the angle between vectors
        dot = sum(a*b for a, b in zip(v1, v2))
        m1 = math.sqrt(sum(a*a for a in v1))
        m2 = math.sqrt(sum(a*a for a in v2))
        return dot / (m1 * m2) if m1 * m2 > 0 else 0

    def predict(self, vec):
        # Assign label based on the highest cosine similarity to a centroid
        sims = {label: self._cosine_sim(vec, cent) for label, cent in self.centroids.items()}
        return max(sims, key=sims.get)

def calculate_accuracy(model, test_data, true_labels, is_vector=False):
    """
    Helper to calculate accuracy by comparing model predictions 
    to the ground truth labels.
    """
    correct = 0
    for doc, label in zip(test_data, true_labels):
        pred = model.predict(doc)
        if pred == label:
            correct += 1
    return correct / len(true_labels)

def show_accuracy_graph(nb_acc, knn_acc, cos_acc):
    # Generate a bar chart using dynamically calculated scores
    models = ['Naive Bayes', 'KNN (k=3)', 'Cosine Centroid']
    accuracies = [nb_acc, knn_acc, cos_acc]
    plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('ML Model Comparison - Sport vs Politics')
    plt.ylim(0, 1.1) # Set limit slightly above 1 for visibility
    plt.show()

def main():
    # Sample training data for two categories
    sport_wiki = """
    Sport is a physical activity or game, often competitive and organized. 
    Common sports include football, cricket, basketball, tennis, chess and rugby. 
    Athletes compete in the Olympic Games and World Cup tournaments. 
    Coaches train teams for matches, scores, and championships.
    """
    politics_wiki = """
    Politics involves decision making in groups and power relations. 
    Governments consist of legislature, executive, and judiciary branches. 
    Elections determine leaders through political parties and voting. 
    Diplomacy and law regulate sovereign states and public policy.
    """

    # Pre-process text into labeled document sets
    s_docs = [s.strip() for s in sport_wiki.split('.') if len(s) > 15]
    p_docs = [p.strip() for p in politics_wiki.split('.') if len(p) > 15]
    docs = s_docs + p_docs
    labels = ["Sport"] * len(s_docs) + ["Politics"] * len(p_docs)

    # Vectorize the documents using TF-IDF
    tfidf = TFIDFModel()
    tfidf.fit(docs)
    vectors = [tfidf.transform(d) for d in docs]

    # Initialize and train all three classification models
    nb = NaiveBayes(); nb.train(docs, labels)
    knn = KNN(k=3); knn.train(vectors, labels)
    cos = CosineClassifier(); cos.train(vectors, labels)

    # Dynamically calculate the accuracy of each model
    nb_acc = calculate_accuracy(nb, docs, labels)
    knn_acc = calculate_accuracy(knn, vectors, labels)
    cos_acc = calculate_accuracy(cos, vectors, labels)

    print("-" * 40)
    print("COMPARATIVE CLASSIFIER SYSTEM")
    print("-" * 40)
    
    # Display the comparison graph with real data
    show_accuracy_graph(nb_acc, knn_acc, cos_acc)

    # Interactive loop for user input testing
    while True:
        txt = input("\nEnter text: ").strip()
        if txt.lower() == 'exit': break
        if not txt: continue
        
        vec = tfidf.transform(txt)
        print(f"1. Naive Bayes Answer:      {nb.predict(txt)}")
        print(f"2. KNN (k=3) Answer:        {knn.predict(vec)}")
        print(f"3. Cosine Centroid Answer:  {cos.predict(vec)}")

if __name__ == "__main__":
    main()
