# Anime Recommendation System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [About](#about)
- [How It Works](#how-it-works)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

In this project, we have developed an Anime Recommendation System based on user reviews from MyAnimeList (MAL). The system utilizes collaborative filtering and text processing techniques to provide personalized anime recommendations to users. For more detailed information, please check the [About Page](./about.md).

## How It Works

Our Anime Recommendation System has three main features:

- **Anime Recommendation:** Allows users to find animes similar to their favorite anime based on genres, synopsis, and title. Users can specify the number of animes they want to receive as recommendations.

- **User x Anime Comparison:** Enables users to compare their preferences with an anime and get a score indicating how much they might like the anime.

- **Recommendation Based on Similar Users:** Provides anime recommendations based on the preferences of users with similar tastes.

For more details on how the system works, please visit the [How It Works Page](markdown/howitworks.md).

## Features

- User-based anime recommendation system
- Collaborative filtering for user comparisons
- Text processing using TF-IDF for anime similarity
- Streamlit web interface for interactive user experience

## Technologies Used

- Python
- Pandas
- NumPy
- difflib
- scikit-learn
- TfidfVectorizer
- Streamlit

## Getting Started

To get started with the Anime Recommendation System, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/vector-b/AnimeRecommendationSystem.git
