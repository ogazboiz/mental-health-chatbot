import os
from zipfile import ZipFile

# 1. Create the target directory
nltk_dir = os.path.expanduser('~/nltk_data/tokenizers/punkt_tab')
os.makedirs(nltk_dir, exist_ok=True)

# 2. Extract the downloaded zip file there
with ZipFile('punkt.zip', 'r') as zip_ref:
    zip_ref.extractall(nltk_dir)

# 3. Verify
from nltk.tokenize import word_tokenize
print(word_tokenize("This should now work"))  # Should output ['This', 'should', 'now', 'work']