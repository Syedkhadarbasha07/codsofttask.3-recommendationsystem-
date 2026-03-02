import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------
data = pd.read_csv("movies.csv")

print("\nDataset:\n")
print(data)

# ---------------- CREATE USER-MOVIE MATRIX ----------------
user_movie_matrix = data.pivot_table(
    index="User",
    columns="Movie",
    values="Rating"
).fillna(0)

print("\nUser-Movie Matrix:\n")
print(user_movie_matrix)

# ---------------- CALCULATE SIMILARITY ----------------
similarity = cosine_similarity(user_movie_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("\nUser Similarity Matrix:\n")
print(similarity_df)


# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_movies(user):

    print(f"\nRecommendations for User {user}:")

    # Find similar users
    similar_users = similarity_df[user].sort_values(ascending=False)

    # Remove self similarity
    similar_users = similar_users.drop(user)

    # Most similar user
    top_user = similar_users.index[0]

    print("Most similar user:", top_user)

    user_movies = user_movie_matrix.loc[user]
    top_user_movies = user_movie_matrix.loc[top_user]

    recommendations = []

    for movie in user_movie_matrix.columns:
        if user_movies[movie] == 0 and top_user_movies[movie] >= 4:
            recommendations.append(movie)

    return recommendations


# ---------------- RUN SYSTEM ----------------
user_name = input("\nEnter User (A/B/C/D): ").upper()

if user_name in user_movie_matrix.index:
    recs = recommend_movies(user_name)
    print("Recommended Movies:", recs)
else:
    print("User not found!")