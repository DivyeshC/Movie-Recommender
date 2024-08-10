import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\HP\pyprojects\Movie recommender\movie_dataset.csv")
df.head()
features = ['keywords','cast','genres','director']
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
for feature in features:
    df[feature] = df[feature].fillna('')

df["combined_features"] = df.apply(combine_features,axis=1)
cv = CountVectorizer() 
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
inp = input("Enter your movie name: ")
cap_inp = inp.title()
movie_index = get_index_from_title(cap_inp)
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
print("Top 5 similar movies to "+cap_inp+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>5:
        break

similar_movie_genres = []
for element in sorted_similar_movies[:10]:
    similar_movie_genres.extend(df['genres'][element[0]].split(' '))
genre_counts = pd.Series(similar_movie_genres).value_counts()
plt.figure(figsize=(8, 8))
genre_counts.plot(kind='pie', autopct='%1.1f%%', startangle=360)
plt.title('Genres of Similar Movies')
plt.show()

similarity_scores = [element[1] for element in sorted_similar_movies]
plt.figure(figsize=(10, 6))
sns.histplot(similarity_scores, bins=20, kde=True)
plt.title('Distribution of Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.show()

similar_movie_genres = []
for element in sorted_similar_movies[:10]:
    similar_movie_genres.extend(df['genres'][element[0]].split(' '))
plt.figure(figsize=(10, 6))
sns.countplot(y=similar_movie_genres, palette='viridis', order=pd.Series(similar_movie_genres).value_counts().index)
plt.title('Distribution of Genres in Similar Movies')
plt.xlabel('Frequency')
plt.ylabel('Genres')
plt.show()


