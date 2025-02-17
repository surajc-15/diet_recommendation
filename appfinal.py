# # # # # # import numpy as np
# # # # # # import pickle
# # # # # # import streamlit as st
# # # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # # import dill
# # # # # # import plotly.express as px  # Import Plotly for Visualization

# # # # # # # Load the Pickle File using Dill
# # # # # # with open('filtered_diet.pkl', 'rb') as f:
# # # # # #     data = dill.load(f)

# # # # # # filtered_diet = data['filtered_diet']

# # # # # # # Function to calculate BMR and TDEE
# # # # # # def calculate_bmr(weight, height, age, sex):
# # # # # #     if sex == "male":
# # # # # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
# # # # # #     else:
# # # # # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

# # # # # # def calculate_tdee(bmr, activity_level):
# # # # # #     activity_multipliers = {
# # # # # #         "sedentary": 1.2,
# # # # # #         "lightly_active": 1.375,
# # # # # #         "moderate": 1.55,
# # # # # #         "very_active": 1.725,
# # # # # #         "extra_active": 1.9
# # # # # #     }
# # # # # #     return bmr * activity_multipliers.get(activity_level, 1.2)

# # # # # # # Function to calculate Macronutrients
# # # # # # def calculate_macros(calories, goal):
# # # # # #     if goal == "weight_loss":
# # # # # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
# # # # # #     elif goal == "weight_gain":
# # # # # #         protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
# # # # # #     else:
# # # # # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
# # # # # #     protein = (calories * protein_ratio) / 4
# # # # # #     fat = (calories * fat_ratio) / 9
# # # # # #     carbs = (calories * carb_ratio) / 4
# # # # # #     return protein, fat, carbs

# # # # # # # User Inputs
# # # # # # st.title("Personalized Meal Recommendation System")
# # # # # # age = st.number_input('Age', min_value=1, max_value=120, value=25)
# # # # # # sex = st.selectbox('Sex', ['male', 'female'])
# # # # # # weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
# # # # # # height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
# # # # # # activity_level = st.selectbox('Activity Level', ['sedentary', 'lightly_active', 'moderate', 'very_active', 'extra_active'])
# # # # # # goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])
# # # # # # diet_preference = st.selectbox('Diet Preference', ['none', 'vegetarian', 'vegan'])
# # # # # # allergies = st.text_input('Allergies (comma separated)', '')

# # # # # # def recommend_meals(user_calories, user_fat, user_carbs, user_protein, top_n=5):
# # # # # #     user_profile_vector = np.array([[user_calories, user_fat, user_carbs, user_protein, 0, 0, 0, 0, 0]])
# # # # # #     similarities = cosine_similarity(user_profile_vector, filtered_diet.iloc[:, 1:10].values)
    
# # # # # #     filtered_diet["Similarity"] = similarities[0]
# # # # # #     top_recommendations = filtered_diet.sort_values(by="Similarity", ascending=False).head(top_n)
    
# # # # # #     recommendations = []
# # # # # #     for _, row in top_recommendations.iterrows():
# # # # # #         recommendation = {
# # # # # #             "Name": row["Name"],
# # # # # #             "Calories": row["Calories"],
# # # # # #             "ProteinContent": row["ProteinContent"],
# # # # # #             "FatContent": row["FatContent"],
# # # # # #             "CarbohydrateContent": row["CarbohydrateContent"],
# # # # # #             "SodiumContent": row["SodiumContent"],
# # # # # #             "RecipeInstructions": row["RecipeInstructions"]
# # # # # #         }
# # # # # #         recommendations.append(recommendation)
    
# # # # # #     return recommendations

# # # # # # # Button to Trigger Recommendation
# # # # # # if st.button('Get Recommendation'):
# # # # # #     # Calculate Nutritional Needs
# # # # # #     bmr = calculate_bmr(weight, height, age, sex)
# # # # # #     tdee = calculate_tdee(bmr, activity_level)
# # # # # #     target_calories = tdee - 500 if goal == "weight_loss" else tdee + 500 if goal == "weight_gain" else tdee
# # # # # #     target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

# # # # # #     # Filter Recipes Based on Diet Preferences & Allergies
# # # # # #     filtered_df = filtered_diet.copy()

# # # # # #     if diet_preference in ["vegetarian", "vegan"]:
# # # # # #         filtered_df = filtered_df[filtered_df["RecipeInstructions"].str.contains(diet_preference, case=False, na=False)]

# # # # # #     if allergies:
# # # # # #         for allergen in allergies.split(','):
# # # # # #             allergen = allergen.strip()
# # # # # #             filtered_df = filtered_df[~filtered_df["RecipeInstructions"].str.contains(allergen, case=False, na=False)]

# # # # # #     # Generate Meal Recommendations
# # # # # #     recommendations = recommend_meals(target_calories, target_fat, target_carbs, target_protein, top_n=5)

# # # # # #     # Display the Recommendations
# # # # # #     st.header("Top 5 Recommended Meals")
# # # # # #     if recommendations:
# # # # # #         names = []
# # # # # #         calories = []
# # # # # #         protein = []
# # # # # #         fat = []
# # # # # #         carbs = []
# # # # # #         sodium = []

# # # # # #         for idx, rec in enumerate(recommendations, start=1):
# # # # # #             st.subheader(f"Recipe {idx}")
# # # # # #             st.write(f"**Name:** {rec['Name']}")
# # # # # #             st.write(f"**Calories:** {rec['Calories']}")
# # # # # #             st.write(f"**Protein Content:** {rec['ProteinContent']}g")
# # # # # #             st.write(f"**Fat Content:** {rec['FatContent']}g")
# # # # # #             st.write(f"**Carbohydrate Content:** {rec['CarbohydrateContent']}g")
# # # # # #             st.write(f"**Sodium Content:** {rec['SodiumContent']}mg")
# # # # # #             st.write(f"**Recipe Instructions:** {rec['RecipeInstructions']}")
# # # # # #             st.markdown("---")

# # # # # #             # Collect data for Cumulative Graph
# # # # # #             names.append(rec['Name'])
# # # # # #             calories.append(rec['Calories'])
# # # # # #             protein.append(rec['ProteinContent'])
# # # # # #             fat.append(rec['FatContent'])
# # # # # #             carbs.append(rec['CarbohydrateContent'])
# # # # # #             sodium.append(rec['SodiumContent'])

# # # # # #         # Cumulative Graph for Nutritional Values
# # # # # #         st.subheader("Cumulative Nutritional Values")
# # # # # #         cum_df = {
# # # # # #             'Recipe': names,
# # # # # #             'Protein': protein,
# # # # # #             'Fat': fat,
# # # # # #             'Carbs': carbs,
# # # # # #             'Sodium': sodium
# # # # # #         }
# # # # # #         fig = px.bar(
# # # # # #             cum_df,
# # # # # #             x='Recipe',
# # # # # #             y=['Protein', 'Fat', 'Carbs', 'Sodium'],
# # # # # #             title="Nutritional Values in Recommended Meals",
# # # # # #             labels={'value': 'Nutritional Value', 'Recipe': 'Recipe Name'},
# # # # # #             barmode='stack'
# # # # # #         )
# # # # # #         st.plotly_chart(fig)
# # # # # #     else:
# # # # # #         st.warning("No recommendations available for the given criteria.")

# # # # # import numpy as np
# # # # # import pickle
# # # # # import streamlit as st
# # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # from sklearn.neighbors import KNeighborsClassifier
# # # # # from sklearn.ensemble import RandomForestClassifier
# # # # # import dill
# # # # # import pandas as pd
# # # # # import matplotlib.pyplot as plt

# # # # # # Load the Pickle File using Dill
# # # # # with open('filtered_diet.pkl', 'rb') as f:
# # # # #     data = dill.load(f)

# # # # # filtered_diet = data['filtered_diet']

# # # # # # Function to calculate BMR and TDEE
# # # # # def calculate_bmr(weight, height, age, sex):
# # # # #     if sex == "male":
# # # # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
# # # # #     else:
# # # # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

# # # # # def calculate_tdee(bmr, activity_level):
# # # # #     activity_multipliers = {
# # # # #         "sedentary": 1.2,
# # # # #         "lightly_active": 1.375,
# # # # #         "moderate": 1.55,
# # # # #         "very_active": 1.725,
# # # # #         "extra_active": 1.9
# # # # #     }
# # # # #     return bmr * activity_multipliers.get(activity_level, 1.2)

# # # # # # Function to calculate Macronutrients
# # # # # def calculate_macros(calories, goal):
# # # # #     if goal == "weight_loss":
# # # # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
# # # # #     elif goal == "weight_gain":
# # # # #         protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
# # # # #     else:
# # # # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
# # # # #     protein = (calories * protein_ratio) / 4
# # # # #     fat = (calories * fat_ratio) / 9
# # # # #     carbs = (calories * carb_ratio) / 4
# # # # #     return protein, fat, carbs

# # # # # # Function for Cosine Similarity Recommendations
# # # # # def recommend_cosine(user_vector, top_n=5):
# # # # #     similarities = cosine_similarity(user_vector, filtered_diet.iloc[:, 1:10].values)  
# # # # #     filtered_diet["Similarity"] = similarities[0]
# # # # #     return filtered_diet.sort_values(by="Similarity", ascending=False).head(top_n)

# # # # # # Function for KNN Recommendations
# # # # # def recommend_knn(user_vector, top_n=5):
# # # # #     knn = KNeighborsClassifier(n_neighbors=top_n)
# # # # #     X = filtered_diet.iloc[:, 1:10].values
# # # # #     y = filtered_diet['Name']
# # # # #     knn.fit(X, y)
# # # # #     distances, indices = knn.kneighbors(user_vector)
# # # # #     return filtered_diet.iloc[indices[0]]

# # # # # # Function for Random Forest Recommendations
# # # # # def recommend_random_forest(user_vector, top_n=5):
# # # # #     rf = RandomForestClassifier(n_estimators=100)
# # # # #     X = filtered_diet.iloc[:, 1:10].values
# # # # #     y = filtered_diet['Name']
# # # # #     rf.fit(X, y)
# # # # #     predictions = rf.predict(user_vector)
# # # # #     return filtered_diet[filtered_diet['Name'].isin(predictions)].head(top_n)

# # # # # # Function to Plot Cumulative Nutritional Values
# # # # # def plot_nutritional_values(recommendations):
# # # # #     nutrition_totals = recommendations[[ 'ProteinContent', 'FatContent', 'CarbohydrateContent']].sum()
# # # # #     plt.figure(figsize=(6,6))
# # # # #     plt.pie(nutrition_totals, labels=nutrition_totals.index, autopct='%1.1f%%')
# # # # #     plt.title(f'Cumulative Nutritional Values')
# # # # #     st.pyplot(plt)

# # # # # # User Inputs for Recommendation
# # # # # st.title("Personalized Meal Recommendation System")
# # # # # age = st.number_input('Age', min_value=1, max_value=120, value=25)
# # # # # sex = st.selectbox('Sex', ['male', 'female'])
# # # # # weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
# # # # # height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
# # # # # activity_level = st.selectbox('Activity Level', ['sedentary', 'lightly_active', 'moderate', 'very_active', 'extra_active'])
# # # # # goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])

# # # # # target_calories = calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level)
# # # # # target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

# # # # # # target_calories, target_protein, target_fat, target_carbs = calculate_macros(calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level), goal)
# # # # # user_vector = np.array([[target_calories, target_fat, target_carbs, target_protein, 0, 0, 0, 0, 0]])

# # # # # if st.button('Get Recommendations'):
# # # # #     rec_cosine = recommend_cosine(user_vector)
# # # # #     rec_knn = recommend_knn(user_vector)
# # # # #     rec_rf = recommend_random_forest(user_vector)
# # # # #     # st.subheader("Cosine Similarity Recommendations")
# # # # #     st.dataframe(rec_cosine)
# # # # #     plot_nutritional_values(rec_cosine)
# # # # #     # st.subheader("KNN Recommendations")
# # # # #     st.dataframe(rec_knn)
# # # # #     print(rec_knn.columns)
# # # # #     plot_nutritional_values(rec_knn)
# # # # #     # st.subheader("Random Forest Recommendations")
# # # # #     # st.dataframe(rec_rf)
# # # # #     # plot_nutritional_values(rec_rf, "Random Forest")


# # # # import numpy as np
# # # # import streamlit as st
# # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # from sklearn.neighbors import KNeighborsClassifier
# # # # from sklearn.ensemble import RandomForestClassifier
# # # # import dill
# # # # import matplotlib.pyplot as plt

# # # # # Load the Pickle File using Dill
# # # # with open('filtered_diet.pkl', 'rb') as f:
# # # #     data = dill.load(f)

# # # # filtered_diet = data['filtered_diet']

# # # # # Function to calculate BMR and TDEE
# # # # def calculate_bmr(weight, height, age, sex):
# # # #     if sex == "male":
# # # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
# # # #     else:
# # # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

# # # # def calculate_tdee(bmr, activity_level):
# # # #     activity_multipliers = {
# # # #         "sedentary": 1.2,
# # # #         "lightly_active": 1.375,
# # # #         "moderate": 1.55,
# # # #         "very_active": 1.725,
# # # #         "extra_active": 1.9
# # # #     }
# # # #     return bmr * activity_multipliers.get(activity_level, 1.2)

# # # # # Function to calculate Macronutrients
# # # # def calculate_macros(calories, goal):
# # # #     if goal == "weight_loss":
# # # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
# # # #     elif goal == "weight_gain":
# # # #         protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
# # # #     else:
# # # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
# # # #     protein = (calories * protein_ratio) / 4
# # # #     fat = (calories * fat_ratio) / 9
# # # #     carbs = (calories * carb_ratio) / 4
# # # #     return protein, fat, carbs

# # # # # Function for Cosine Similarity Recommendations
# # # # def recommend_knn(user_vector, top_n=5):
# # # #     X = filtered_diet.iloc[:, 1:10].values  # Feature vectors
# # # #     y = filtered_diet['Name'].values  # Assuming a 'Category' column for classification
# # # #     knn = KNeighborsClassifier(n_neighbors=top_n)
# # # #     knn.fit(X, y)
# # # #     distances, indices = knn.kneighbors(user_vector)
# # # #     recommendations = filtered_diet.iloc[indices[0]]
# # # #     return recommendations

# # # # # Function to Plot Cumulative Nutritional Values
# # # # def plot_nutritional_values(recommendations, title):
# # # #     nutrition_totals = recommendations[['ProteinContent', 'FatContent', 'CarbohydrateContent']].sum()
# # # #     plt.figure(figsize=(6,6))
# # # #     plt.pie(nutrition_totals, labels=nutrition_totals.index, autopct='%1.1f%%')
# # # #     plt.title(title)
# # # #     st.pyplot(plt)

# # # # # User Inputs for Recommendation
# # # # st.title("Personalized Meal Recommendation System")
# # # # age = st.number_input('Age', min_value=1, max_value=120, value=25)
# # # # sex = st.selectbox('Sex', ['male', 'female'])
# # # # weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
# # # # height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
# # # # activity_level = st.selectbox('Activity Level', ['sedentary', 'lightly_active', 'moderate', 'very_active', 'extra_active'])
# # # # goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])

# # # # target_calories = calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level)
# # # # target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

# # # # user_vector = np.array([[target_calories, target_fat, target_carbs, target_protein, 0, 0, 0, 0, 0]])

# # # # if st.button('Get Recommendations'):
# # # #     rec_knn = recommend_knn(user_vector)
# # # #     with st.expander("KNN Recommendations"):
# # # #         selected_recommendation = st.selectbox("Select a Meal", rec_knn['Name'].tolist())
# # # #         selected_meal = rec_knn[rec_knn['Name'] == selected_recommendation].iloc[0]
# # # #         st.write("### Meal Details")
# # # #         st.write("**Calories:**", selected_meal['Calories'])
# # # #         st.write("**Fat Content:**", selected_meal['FatContent'])
# # # #         st.write("**Saturated Fat Content:**", selected_meal['SaturatedFatContent'])
# # # #         st.write("**Cholesterol Content:**", selected_meal['CholesterolContent'])
# # # #         st.write("**Sodium Content:**", selected_meal['SodiumContent'])
# # # #         st.write("**Carbohydrate Content:**", selected_meal['CarbohydrateContent'])
# # # #         st.write("**Fiber Content:**", selected_meal['FiberContent'])
# # # #         st.write("**Sugar Content:**", selected_meal['SugarContent'])
# # # #         st.write("**Protein Content:**", selected_meal['ProteinContent'])
# # # #         st.write("**Recipe Instructions:**", selected_meal['RecipeInstructions'])
# # # #     plot_nutritional_values(rec_knn, "KNN Nutritional Values")


# # # import numpy as np
# # # import streamlit as st
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # from sklearn.neighbors import KNeighborsClassifier
# # # from sklearn.ensemble import RandomForestClassifier
# # # import dill
# # # import matplotlib.pyplot as plt

# # # # Load the Pickle File using Dill
# # # with open('filtered_diet.pkl', 'rb') as f:
# # #     data = dill.load(f)

# # # filtered_diet = data['filtered_diet']

# # # # Function to calculate BMR and TDEE
# # # def calculate_bmr(weight, height, age, sex):
# # #     if sex == "male":
# # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
# # #     else:
# # #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

# # # def calculate_tdee(bmr, activity_level):
# # #     activity_multipliers = {
# # #         1: 1.2,  # sedentary
# # #         2: 1.375,  # lightly_active
# # #         3: 1.55,  # moderate
# # #         4: 1.725,  # very_active
# # #         5: 1.9   # extra_active
# # #     }
# # #     return bmr * activity_multipliers.get(activity_level, 1.2)

# # # # Function to calculate Macronutrients
# # # def calculate_macros(calories, goal):
# # #     if goal == "weight_loss":
# # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
# # #     elif goal == "weight_gain":
# # #         protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
# # #     else:
# # #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
# # #     protein = (calories * protein_ratio) / 4
# # #     fat = (calories * fat_ratio) / 9
# # #     carbs = (calories * carb_ratio) / 4
# # #     return protein, fat, carbs

# # # # Function for Cosine Similarity Recommendations
# # # def recommend_knn(user_vector, top_n=5):
# # #     X = filtered_diet.iloc[:, 1:10].values  # Feature vectors
# # #     y = filtered_diet['Name'].values  # Assuming a 'Category' column for classification
# # #     knn = KNeighborsClassifier(n_neighbors=top_n)
# # #     knn.fit(X, y)
# # #     distances, indices = knn.kneighbors(user_vector)
# # #     recommendations = filtered_diet.iloc[indices[0]]
# # #     return recommendations

# # # # Function to Plot Cumulative Nutritional Values
# # # def plot_nutritional_values(recommendations, title):
# # #     nutrition_totals = recommendations[['ProteinContent', 'FatContent', 'CarbohydrateContent']].sum()
# # #     plt.figure(figsize=(6,6))
# # #     plt.pie(nutrition_totals, labels=nutrition_totals.index, autopct='%1.1f%%')
# # #     plt.title(title)
# # #     st.pyplot(plt)

# # # # User Inputs for Recommendation
# # # st.title("Personalized Meal Recommendation System")
# # # age = st.number_input('Age', min_value=1, max_value=120, value=25)
# # # sex = st.selectbox('Sex', ['male', 'female'])
# # # weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
# # # height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
# # # activity_level = st.slider('Activity Level', min_value=1, max_value=5, value=3, step=1)
# # # goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])

# # # target_calories = calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level)
# # # target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

# # # user_vector = np.array([[target_calories, target_fat, target_carbs, target_protein, 0, 0, 0, 0, 0]])

# # # rec_knn = None  # Initialize rec_knn
# # # if st.button('Get Recommendations'):
# # #     rec_knn = recommend_knn(user_vector)
# # #     st.write('### KNN Recommendations')
# # # st.markdown('---')
# # # if rec_knn is not None:
# # #     rec_knn.reset_index(drop=True, inplace=True)
# # # if rec_knn is not None:
# # #     rec_knn['RecipeInstructions'] = rec_knn['RecipeInstructions'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
# # # st.write('#### Recommended Meals')
# # # if rec_knn is not None:
# # #     cols = st.columns(len(rec_knn))
# # # if rec_knn is not None:
# # #     for i, col in enumerate(cols):
# # #         with col:
# # #             st.markdown(f"**{rec_knn['Name'][i]}**")
# # #             st.write(f"Calories: {rec_knn['Calories'][i]}")
# # #             st.write(f"Protein: {rec_knn['ProteinContent'][i]}")
# # #             st.write(f"Fat: {rec_knn['FatContent'][i]}")
# # #             st.write(f"Carbs: {rec_knn['CarbohydrateContent'][i]}")
# # #             st.selectbox('Recipe Instructions', [rec_knn['RecipeInstructions'][i]], key=f'recipe_{i}')
# # # st.markdown('---')
# # # if rec_knn is not None:
# # #     plot_nutritional_values(rec_knn, "KNN Nutritional Values")

# # import numpy as np
# # import streamlit as st
# # from sklearn.metrics.pairwise import cosine_similarity
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # import dill
# # import matplotlib.pyplot as plt

# # # Load the Pickle File using Dill
# # with open('filtered_diet.pkl', 'rb') as f:
# #     data = dill.load(f)

# # filtered_diet = data['filtered_diet']

# # # Function to calculate BMR and TDEE
# # def calculate_bmr(weight, height, age, sex):
# #     if sex == "male":
# #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
# #     else:
# #         return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

# # def calculate_tdee(bmr, activity_level):
# #     activity_multipliers = {
# #         1: 1.2,  # sedentary
# #         2: 1.375,  # lightly_active
# #         3: 1.55,  # moderate
# #         4: 1.725,  # very_active
# #         5: 1.9   # extra_active
# #     }
# #     return bmr * activity_multipliers.get(activity_level, 1.2)

# # # Function to calculate Macronutrients
# # def calculate_macros(calories, goal):
# #     if goal == "weight_loss":
# #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
# #     elif goal == "weight_gain":
# #         protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
# #     else:
# #         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
# #     protein = (calories * protein_ratio) / 4
# #     fat = (calories * fat_ratio) / 9
# #     carbs = (calories * carb_ratio) / 4
# #     return protein, fat, carbs

# # # Function for Cosine Similarity Recommendations
# # def recommend_knn(user_vector, top_n=5):
# #     X = filtered_diet.iloc[:, 1:10].values  # Feature vectors
# #     y = filtered_diet['Name'].values  # Assuming a 'Category' column for classification
# #     knn = KNeighborsClassifier(n_neighbors=top_n)
# #     knn.fit(X, y)
# #     distances, indices = knn.kneighbors(user_vector)
# #     recommendations = filtered_diet.iloc[indices[0]]
# #     return recommendations

# # # Function to Plot Cumulative Nutritional Values
# # def plot_nutritional_values(recommendations, title):
# #     nutrition_totals = recommendations[['ProteinContent', 'FatContent', 'CarbohydrateContent']].sum()
# #     plt.figure(figsize=(6,6))
# #     plt.pie(nutrition_totals, labels=nutrition_totals.index, autopct='%1.1f%%')
# #     plt.title(title)
# #     st.pyplot(plt)

# # # User Inputs for Recommendation
# # st.title("Personalized Meal Recommendation System")
# # age = st.number_input('Age', min_value=1, max_value=120, value=25)
# # sex = st.selectbox('Sex', ['male', 'female'])
# # weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
# # height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
# # activity_level = st.slider('Activity Level', min_value=1, max_value=5, value=3, step=1, format='%d')
# # activity_names = ['Sedentary', 'Lightly Active', 'Moderate', 'Very Active', 'Extra Active']
# # st.markdown(f"<p style='color: green;'>Activity Level: {activity_names[activity_level-1]}</p>", unsafe_allow_html=True)
# # goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])

# # target_calories = calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level)
# # target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

# # user_vector = np.array([[target_calories, target_fat, target_carbs, target_protein, 0, 0, 0, 0, 0]])

# # rec_knn = None  # Initialize rec_knn
# # if st.button('Get Recommendations'):
# #     rec_knn = recommend_knn(user_vector)
# #     st.write('### KNN Recommendations')
# # st.markdown('</div>', unsafe_allow_html=True)
# # st.markdown('---')
# # if rec_knn is not None:
# #     rec_knn.reset_index(drop=True, inplace=True)
# # if rec_knn is not None:
# #     rec_knn['RecipeInstructions'] = rec_knn['RecipeInstructions'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
# # st.write('#### Recommended Meals')
# # if rec_knn is not None:
# #     cols = st.columns(len(rec_knn))
# # if rec_knn is not None:
# #     for i, col in enumerate(cols):
# #         with col:
# #             st.markdown(f"<div style='border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #e6ffe6; margin-bottom: 10px;'>", unsafe_allow_html=True)
# #         with col:
# #             st.markdown(f"<h4 style='color: green;'>{rec_knn['Name'][i]}</h4>", unsafe_allow_html=True)
# #             st.write(f"Calories: {rec_knn['Calories'][i]}")
# #             st.write(f"Protein: {rec_knn['ProteinContent'][i]}")
# #             st.write(f"Fat: {rec_knn['FatContent'][i]}")
# #             st.write(f"Carbs: {rec_knn['CarbohydrateContent'][i]}")
# #             st.expander('Recipe Instructions').write(rec_knn['RecipeInstructions'][i])
# # st.markdown('---')
# # if rec_knn is not None:
# #     plot_nutritional_values(rec_knn, "KNN Nutritional Values")
# import numpy as np
# import streamlit as st
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# import dill
# import matplotlib.pyplot as plt

# # Load the Pickle File using Dill
# with open('filtered_diet.pkl', 'rb') as f:
#     data = dill.load(f)

# filtered_diet = data['filtered_diet']

# # Function to calculate BMR and TDEE
# def calculate_bmr(weight, height, age, sex):
#     if sex == "male":
#         return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
#     else:
#         return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

# def calculate_tdee(bmr, activity_level):
#     activity_multipliers = {
#         1: 1.2,  # sedentary
#         2: 1.375,  # lightly_active
#         3: 1.55,  # moderate
#         4: 1.725,  # very_active
#         5: 1.9   # extra_active
#     }
#     return bmr * activity_multipliers.get(activity_level, 1.2)

# # Function to calculate Macronutrients
# def calculate_macros(calories, goal):
#     if goal == "weight_loss":
#         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
#     elif goal == "weight_gain":
#         protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
#     else:
#         protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
#     protein = (calories * protein_ratio) / 4
#     fat = (calories * fat_ratio) / 9
#     carbs = (calories * carb_ratio) / 4
#     return protein, fat, carbs

# # Function for Cosine Similarity Recommendations
# def recommend_knn(user_vector, top_n=5):
#     X = filtered_diet.iloc[:, 1:10].values  # Feature vectors
#     y = filtered_diet['Name'].values  # Assuming a 'Category' column for classification
#     knn = KNeighborsClassifier(n_neighbors=top_n)
#     knn.fit(X, y)
#     distances, indices = knn.kneighbors(user_vector)
#     recommendations = filtered_diet.iloc[indices[0]]
#     return recommendations

# # Function to Plot Cumulative Nutritional Values
# def plot_nutritional_values(recommendations, title):
#     nutrition_totals = recommendations[['ProteinContent', 'FatContent', 'CarbohydrateContent']].sum()
#     plt.figure(figsize=(4,4))
#     plt.pie(nutrition_totals, labels=nutrition_totals.index, autopct='%1.1f%%')
#     plt.title(title)
#     st.pyplot(plt)

# # User Inputs for Recommendation
# st.title("Personalized Meal Recommendation System")
# age = st.number_input('Age', min_value=1, max_value=120, value=25)
# sex = st.selectbox('Sex', ['male', 'female'])
# weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
# height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
# activity_level = st.slider('Activity Level', min_value=1, max_value=5, value=3, step=1, format='%d')
# activity_names = ['Sedentary', 'Lightly Active', 'Moderate', 'Very Active', 'Extra Active']
# st.markdown(f"<p style='color: green;'>Activity Level: {activity_names[activity_level-1]}</p>", unsafe_allow_html=True)
# goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])

# target_calories = calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level)
# target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

# user_vector = np.array([[target_calories, target_fat, target_carbs, target_protein, 0, 0, 0, 0, 0]])

# rec_knn = None  # Initialize rec_knn
# if st.button('Get Recommendations'):
#     rec_knn = recommend_knn(user_vector)
#     st.write('### KNN Recommendations')
# st.markdown('</div>', unsafe_allow_html=True)
# st.markdown('---')
# if rec_knn is not None:
#     rec_knn.reset_index(drop=True, inplace=True)
# if rec_knn is not None:
#     rec_knn['RecipeInstructions'] = rec_knn['RecipeInstructions'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
# st.write('#### Recommended Meals 🌿')
# if rec_knn is not None:
#     cols = st.columns(len(rec_knn))
# if rec_knn is not None:
#     for i, col in enumerate(cols):
#         with col:  # Ensure proper alignment
#             col.markdown(f"<div style='border: 2px solid white; border-radius: 10px; background-color: #e6ffe6; margin-bottom: 10px;'>", unsafe_allow_html=True)
#         with col:
#             col.markdown(f"<h6 style='color: green; height: 60px; overflow: hidden; text-align: center;'>{rec_knn['Name'][i]}</h6>", unsafe_allow_html=True)
#             col.markdown(f"<p style='text-align: left; min-height: 30px;'>Calories: {rec_knn['Calories'][i]}</p>", unsafe_allow_html=True)
#             st.markdown(f"<p style='text-align: left;'>Protein: {rec_knn['ProteinContent'][i]}</p>", unsafe_allow_html=True)
#             st.markdown(f"<p style='text-align: left;'>Fat: {rec_knn['FatContent'][i]}</p>", unsafe_allow_html=True)
#             st.markdown(f"<p style='text-align: left;'>Carbs: {rec_knn['CarbohydrateContent'][i]}</p>", unsafe_allow_html=True)
#             col.expander('Recipe Instructions').write(rec_knn['RecipeInstructions'][i])
# st.markdown('---')
# if rec_knn is not None:
#     plot_nutritional_values(rec_knn, "KNN Nutritional Values")

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import dill
import matplotlib.pyplot as plt

# Load the Pickle File using Dill
with open('filtered_diet.pkl', 'rb') as f:
    data = dill.load(f)

filtered_diet = data['filtered_diet']

# Function to calculate BMR and TDEE
def calculate_bmr(weight, height, age, sex):
    if sex == "male":
        return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
    else:
        return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        1: 1.2,  # sedentary
        2: 1.375,  # lightly_active
        3: 1.55,  # moderate
        4: 1.725,  # very_active
        5: 1.9   # extra_active
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)

# Function to calculate Macronutrients
def calculate_macros(calories, goal):
    if goal == "weight_loss":
        protein_ratio, fat_ratio, carb_ratio = 0.26, 0.3, 0.4
    elif goal == "weight_gain":
        protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
    else:
        protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
    protein = (calories * protein_ratio) / 4
    fat = (calories * fat_ratio) / 9
    carbs = (calories * carb_ratio) / 4
    return protein, fat, carbs


def recommend_knn(user_vector, top_n=5):
    X = filtered_diet.iloc[:, 1:10].values  # Feature vectors
    y = filtered_diet['Name'].values 
    knn = KNeighborsClassifier(n_neighbors=top_n)
    knn.fit(X, y)
    distances, indices = knn.kneighbors(user_vector)
    recommendations = filtered_diet.iloc[indices[0]]
    return recommendations

# Function to Plot Cumulative Nutritional Values
def plot_nutritional_values(recommendations, title):
    nutrition_totals = recommendations[['ProteinContent', 'FatContent', 'CarbohydrateContent']].sum()
    plt.figure(figsize=(4,4))
    plt.pie(nutrition_totals, labels=nutrition_totals.index, autopct='%1.1f%%')
    plt.title(title)
    st.pyplot(plt)

# User Inputs for Recommendation
st.title("Personalized Meal Recommendation System")
st.markdown("<div style='text-align: right; margin-bottom: 10px;'><a href='https://multilingualchatbotapplication.streamlit.app/' target='_blank'><button style='background-color:#4CAF50;color:white;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:14px;box-shadow: 2px 2px 5px #888888;'>💬 Open Chatbot</button></a></div>", unsafe_allow_html=True)
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
activity_level = st.slider('Activity Level', min_value=1, max_value=5, value=3, step=1, format='%d')
activity_names = ['Sedentary', 'Lightly Active', 'Moderate', 'Very Active', 'Extra Active']
st.markdown(f"<p style='color: green;'>Activity Level: {activity_names[activity_level-1]}</p>", unsafe_allow_html=True)
goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])

target_calories = calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level)
target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

user_vector = np.array([[target_calories, target_fat, target_carbs, target_protein, 0, 0, 0, 0, 0]])

rec_knn = None  # Initialize rec_knn
if st.button('Get Recommendations'):
    rec_knn = recommend_knn(user_vector)
    st.write('### KNN Recommendations')
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('---')
if rec_knn is not None:
    rec_knn.reset_index(drop=True, inplace=True)
if rec_knn is not None:
    rec_knn['RecipeInstructions'] = rec_knn['RecipeInstructions'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
st.write('#### Recommended Meals 🌿')
if rec_knn is not None:
    cols = st.columns(len(rec_knn))
if rec_knn is not None:
    for i, col in enumerate(cols):
        with col:  # Ensure proper alignment
            col.markdown(f"<div style='border: 2px solid grey; border-radius: 10px;  background-color: #e6ffe6; margin-bottom: 10px; '>", unsafe_allow_html=True)
        with col:
            col.markdown(f"<h6 style='color: green; height: 60px; overflow: hidden; text-align: center;'>{rec_knn['Name'][i]} </h6>", unsafe_allow_html=True)
            col.markdown(f"<p style='text-align: left; min-height: 20px;'>Calories: {rec_knn['Calories'][i]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left;'>Protein: {rec_knn['ProteinContent'][i]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left;'>Fat: {rec_knn['FatContent'][i]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left;'>Carbs: {rec_knn['CarbohydrateContent'][i]}</p>", unsafe_allow_html=True)
            col.expander('Recipe Instructions').write(rec_knn['RecipeInstructions'][i])
        with col:  # Ensure proper alignment
            col.markdown(f"<div style='border: 2px solid grey; border-radius: 10px;  background-color: #e6ffe6; margin-bottom: 10px; '>", unsafe_allow_html=True)
st.markdown('---')
if rec_knn is not None:
    plot_nutritional_values(rec_knn, "Nutritional Values 🌿")
