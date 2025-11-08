# Restaurant Review Sentiment Analyzer Pro

An advanced NLP-powered business intelligence platform that analyzes restaurant reviews to provide actionable insights using ensemble machine learning models.

## 🌐 Live Demo

**[View Live Application](https://dev2943-restaurant-sentiment-analyzer.streamlit.app)**

## 📊 Overview

This project analyzes 750+ restaurant reviews using state-of-the-art Natural Language Processing techniques to deliver comprehensive sentiment analysis and business recommendations. The application features multi-restaurant comparison capabilities, real-time sentiment analysis, and automated insights generation.

## ✨ Key Features

- **Advanced NLP Pipeline**: Ensemble ML models using TextBlob, VADER, and scikit-learn algorithms
- **High Accuracy**: Achieves 85%+ classification accuracy
- **Multi-Restaurant Comparison**: Compare sentiment across multiple restaurant locations
- **Real-Time Analysis**: Instant sentiment scoring and classification
- **Topic Modeling**: Automatic extraction of key themes from reviews
- **Interactive Visualizations**: Word clouds, sentiment distributions, and trend analysis
- **Automated Business Insights**: Data-driven recommendations for restaurant performance improvement

## 🛠️ Built With

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **TextBlob** - Simple sentiment analysis
- **VADER** - Social media and short text sentiment analysis
- **Scikit-learn** - Machine learning algorithms (Naive Bayes, Logistic Regression, SVM)
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **WordCloud** - Visual representation of frequent terms

## 📈 Technical Highlights

- **Ensemble Learning**: Combines multiple NLP models for robust predictions
- **Feature Engineering**: Advanced text preprocessing and feature extraction
- **Scalable Architecture**: Handles large datasets efficiently
- **Interactive Dashboard**: User-friendly interface with real-time updates
- **Business Intelligence**: Converts raw sentiment into actionable recommendations

## 🚀 How It Works

1. **Data Input**: Upload restaurant reviews or use the sample dataset
2. **Text Preprocessing**: Clean and normalize review text
3. **Sentiment Analysis**: Apply ensemble ML models (TextBlob, VADER, classifiers)
4. **Topic Modeling**: Extract key themes and topics
5. **Visualization**: Generate interactive charts, word clouds, and metrics
6. **Insights Generation**: Produce automated business recommendations

## 📊 Analysis Features

- Sentiment distribution across positive, negative, and neutral reviews
- Rating correlation analysis
- Topic extraction and frequency analysis
- Word clouds for positive and negative feedback
- Time-series sentiment trends (if date data available)
- Restaurant comparison dashboards
- Key driver analysis for customer satisfaction

## 🎯 Use Cases

- **Restaurant Owners**: Understand customer feedback and improve service
- **Marketing Teams**: Identify strengths and weaknesses for campaigns
- **Operations Managers**: Prioritize areas for operational improvements
- **Business Analysts**: Extract insights from customer reviews at scale

## 📂 Project Structure

```
restaurant-sentiment-analyzer/
├── app.py                  # Main Streamlit application
├── data/                   # Sample review datasets
├── models/                 # Pre-trained ML models
├── utils/                  # Helper functions for NLP
├── visualizations/         # Plotting and charting utilities
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🔧 Local Installation

```bash
# Clone the repository
git clone https://github.com/Dev2943/restaurant-sentiment-analyzer.git

# Navigate to project directory
cd restaurant-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
textblob
vaderSentiment
matplotlib
seaborn
wordcloud
plotly
```

## 🎓 Technical Details

### Machine Learning Models
- **Naive Bayes Classifier**: Fast and efficient for text classification
- **Logistic Regression**: Provides probabilistic sentiment scores
- **Support Vector Machine**: High-dimensional text classification
- **Ensemble Voting**: Combines predictions for improved accuracy

### NLP Techniques
- Tokenization and lemmatization
- Stop word removal
- TF-IDF vectorization
- N-gram analysis
- Part-of-speech tagging

### Evaluation Metrics
- Accuracy: 85%+
- Precision, Recall, F1-Score
- Confusion Matrix Analysis
- Cross-validation scores

## 🌟 Future Enhancements

- [ ] Multi-language support
- [ ] Aspect-based sentiment analysis
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Real-time review scraping from APIs
- [ ] Competitive analysis features
- [ ] Email alert system for negative reviews
- [ ] Mobile app version

## 👨‍💻 Author

**Dev Golakiya**
- Email: devgolakiya31@gmail.com
- LinkedIn: [Dev Golakiya](https://www.linkedin.com/in/devgolakiya)
- GitHub: [@Dev2943](https://github.com/Dev2943)
- Portfolio: [View Portfolio](https://your-portfolio-link.netlify.app)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with Streamlit for rapid prototyping
- NLP libraries: TextBlob, VADER, and scikit-learn
- Dataset sources and restaurant review platforms
- Open source community for tools and inspiration

---

⭐ **If you find this project useful, please consider giving it a star on GitHub!**

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue on GitHub
- Email: devgolakiya31@gmail.com
- Connect on LinkedIn

---

*Last Updated: April 2025*
