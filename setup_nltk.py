import nltk
import os

def setup_nltk():
    """Download required NLTK data packages."""
    # Create NLTK data directory if it doesn't exist
    nltk_data = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data, exist_ok=True)
    
    # Add NLTK data path
    nltk.data.path.append(nltk_data)
    
    print("Setting up NLTK data...")
    
    # List of required NLTK packages
    packages = [
        'punkt',           # For tokenization
        'wordnet',         # For lemmatization
        'averaged_perceptron_tagger',  # For POS tagging
        'stopwords',       # For stopword removal
        'omw-1.4',        # Open Multilingual WordNet (required for non-English support)
        'wordnet_ic',     # WordNet's Information Content
        'sentiwordnet'    # For sentiment analysis
    ]
    
    # Download each package
    for package in packages:
        try:
            print(f"Downloading NLTK package: {package}")
            nltk.download(package, download_dir=nltk_data, quiet=False)
        except Exception as e:
            print(f"Error downloading {package}: {e}")
    
    # Verify downloads
    print("\nVerifying NLTK data installation...")
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            print(f"✓ {package} is installed")
        except LookupError:
            print(f"✗ {package} is NOT installed")
    
    print("\nNLTK setup complete!")
    print(f"NLTK data is stored at: {nltk_data}")

if __name__ == "__main__":
    setup_nltk()
