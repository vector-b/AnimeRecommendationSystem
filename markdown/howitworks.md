

## Key Features

Our project has three main features:

### 1. Anime Recommendation

In the Anime Recommendation feature, users can enter the name of an anime they like or want to find similar animes. Based on the information of genres, synopsis, and title of the animes, the system calculates the similarity between the animes and returns a list of recommended animes that have similar characteristics. The user can specify the number of animes they want to receive as recommendations.


The similarity between two animes is calculated using the cosine similarity metric, which measures the directional similarity between two feature vectors. The calculation involves converting the anime information into feature vectors and then applying the cosine similarity formula.

### 2. User x Anime Comparison

In this feature, users provide their identification (ID) along with the identification (ID) of a specific anime. The system then searches for users who have similar preferences to the provided user, using cosine similarity between user feature vectors. Based on the ratings of these similar users and the corresponding cosine similarity, the system calculates a score that reflects how much the user may like the anime in question.


The similarity between users is calculated based on the ratings they provided for the same animes. From these ratings, we create feature vectors representing the preferences of each user. We then apply the cosine similarity formula to compare these vectors and find users with similar preferences.

### 3. Recommendation Based on Similar Users

In the Recommendation Based on Similar Users feature, users input their identification (ID) and specify the number of animes they want to receive as recommendations. The system finds other users who have similar preferences to the provided user and creates a list of recommended animes based on the ratings of these similar users. The animes are ranked according to a score that considers the ratings and the corresponding cosine similarity.


The recommendation is made using collaborative filtering metric. First, we find users most similar to the provided user, using cosine similarity between user feature vectors. Then, we select the animes rated by these similar users and calculate a score for each anime. This score is used to rank the animes in the recommendation list.

## Technologies Used

Our anime recommendation system was developed in Python and utilizes the following libraries:

- Pandas: For data manipulation and analysis.
- NumPy: For efficient numerical operations.
- difflib: For finding close matches of anime names.
- scikit-learn: For calculating cosine similarity between animes and users.
- TfidfVectorizer: For text vectorization based on TF-IDF.
- Streamlit: For creating the interactive web interface.
