# Fund classification project
### ML Algorithms for investment fund classification

### Directory and files:
- `python\dati\new` -> data directory
- `stopwords.txt` -> stopwords used for text vectorization (in italian and english)
- `classificazione_fondi.yml` -> anaconda environment file
- `dataframe_iniziale.xlsx` -> quantitative and qualitative investment fund data
- `fondi_tradotti_in_italiano.xlsx` -> fund textual data translated in italian from english

### Python scripts:

- `unsupervised_learning_def`: clustering algorithms definitions (unsupervised methods) 
- `unsupervised_learning`: clustering applications

- `supervised_learning_def`: classification algorithms definitions (supervised methods)
- `supervised_learning`: classification algorithms applications

- `text_vectorizer_def`: text vectorization algorithms definitions
- `text_vectorizer`: text vectorization algorithms applications

- `data_management_def`: data management functions
- `data_management`: data management pipeline

- `supervised learning.ipynb`: notebook script
- `*.doc2vec`: neural networks text vectorization experiments esperimenti di vettorizzazione reti neurali

### Running order for classification: 
1. data_management (data_management.py, outlier management, missing data and visualization) 
2. text_vectorizer (text_vectorization_crossval.py, for nlp algorithm comparison)
3. supervised_learning (supervised_learning.py for classification algorithm comparison)

### Running order for clustering: 
1. data_management
2. unsupervised_learning

### Packages used
Name		Ver  
scikit-learn 	0.21.3  
pandas 		0.25.1  
numpy 		1.16.5  
pandas-ml 	0.6.1  
gensim		3.8.1  
nltk		3.4.5  
scipy		1.3.1  
matplotlib	3.1.1  
