# 🧠 Pneumonia Misinformation Simulation

A comprehensive simulation platform that analyzes the impact of misinformation on patient care-seeking behavior in pneumonia cases. This project combines machine learning, social media analysis, and agent-based modeling to understand how misinformation affects healthcare decisions.

## 🌟 Features

### 🔬 **Machine Learning Models**
- **Logistic Regression** and **XGBoost** classifiers for pneumonia detection
- Image preprocessing with data augmentation
- Comprehensive model evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization

### 📱 **Multi-Source Data Collection**
- **Twitter** - Real-time tweet analysis using Twitter API v2
- **News Articles** - Current news coverage via NewsAPI
- **Reddit** - Community discussions and posts
- **Web Search** - Comprehensive web results via Tavily API
- **Wikipedia** - Academic and reference information
- **Hacker News** - Tech community discussions
- **PubMed** - Medical research publications
- **CrossRef** - Academic research metadata

### 🤖 **Agent-Based Simulation**
- **Patient Agents** - Model individual patient behavior and decision-making
- **Clinician Agents** - Represent healthcare provider interactions
- **Misinformation Impact** - Simulate how misinformation affects care-seeking behavior
- **Dynamic Environment** - Multi-grid spatial simulation with random activation

### 📊 **Advanced Analytics**
- Sentiment analysis using TextBlob
- Misinformation detection algorithms
- Real-time data visualization
- Interactive Streamlit dashboard
- Comprehensive reporting and metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Streamlit
- Required API keys (see Configuration section)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pneumonia-misinfo-simulation.git
   cd pneumonia-misinfo-simulation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file with your API keys
   BEARER_TOKEN=your_twitter_bearer_token
   NEWS_API_KEY=your_news_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_pneumonia.py
   ```

## ⚙️ Configuration

### API Keys Required

| Service | Environment Variable | Description |
|---------|---------------------|-------------|
| Twitter | `BEARER_TOKEN` | Twitter API v2 Bearer Token |
| News API | `NEWS_API_KEY` | NewsAPI.org API Key |
| Tavily | `TAVILY_API_KEY` | Tavily Search API Key |

### Optional Settings

- **Tweet Count**: Number of tweets to analyze (1-100)
- **Reddit Posts**: Number of Reddit posts to collect (1-100)
- **Web Results**: Number of Tavily search results (1-50)
- **Wikipedia**: Number of Wikipedia articles (1-50)
- **Hacker News**: Number of HN stories (1-50)
- **PubMed**: Number of medical publications (1-50)
- **CrossRef**: Number of academic papers (1-50)

## 📁 Project Structure

```
pneumonia/
├── streamlit_pneumonia.py    # Main application file
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables (create this)
├── train/                   # Training image dataset
│   ├── NORMAL/             # Normal chest X-rays
│   └── PNEUMONIA/          # Pneumonia chest X-rays
└── test/                    # Test image dataset
    ├── NORMAL/             # Normal chest X-rays
    └── PNEUMONIA/          # Pneumonia chest X-rays
```

## 🔧 Usage

### 1. **Data Collection**
- Configure your search query (default: "Pneumonia")
- Adjust data collection parameters in the sidebar
- Click "Run Simulation" to start data collection

### 2. **Model Training**
- The system automatically processes chest X-ray images
- Trains Logistic Regression and XGBoost models
- Displays performance metrics and confusion matrices

### 3. **Misinformation Analysis**
- Collects data from multiple sources
- Performs sentiment analysis on collected text
- Calculates misinformation rates per source

### 4. **Simulation**
- Runs agent-based simulation with configurable parameters
- Models patient behavior under misinformation influence
- Visualizes care-seeking behavior patterns

### 5. **Results & Visualization**
- Interactive charts and graphs
- Source-wise misinformation analysis
- Sentiment distribution plots
- Simulation trends over time

## 📊 Output Examples

- **Model Performance**: Accuracy metrics, confusion matrices
- **Misinformation Rates**: Per-source breakdown of potential misinformation
- **Sentiment Analysis**: TextBlob polarity scores across sources
- **Simulation Results**: Patient behavior patterns and trends
- **Care-Seeking Impact**: How misinformation affects healthcare decisions

## 🛠️ Technical Details

### Machine Learning Pipeline
- **Image Preprocessing**: Resize to 64x64, normalize to [0,1], data augmentation
- **Feature Extraction**: Flatten images for traditional ML models
- **Model Training**: Logistic Regression and XGBoost with hyperparameter tuning
- **Evaluation**: Comprehensive metrics and visualization

### Agent-Based Modeling
- **Patient Agents**: Individual decision-making based on symptoms and misinformation exposure
- **Clinician Agents**: Healthcare provider representation
- **Spatial Environment**: MultiGrid with random agent activation
- **Data Collection**: Real-time metrics collection during simulation

### Data Sources Integration
- **REST APIs**: Twitter, News, Tavily, PubMed, CrossRef
- **Web Scraping**: Reddit, Wikipedia, Hacker News
- **Rate Limiting**: Built-in timeout and error handling
- **Data Cleaning**: HTML tag removal, text preprocessing

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the interactive web framework
- **Mesa** for agent-based modeling capabilities
- **TextBlob** for sentiment analysis
- **Scikit-learn** and **XGBoost** for machine learning models
- **Matplotlib** and **Seaborn** for data visualization
- **Kaggle Dataset** - [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney for the chest X-ray image dataset

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/pneumonia-misinfo-simulation/issues) page
2. Create a new issue with detailed description
3. Include your environment details and error logs

## 🔬 Research Applications

This simulation platform is designed for:
- **Healthcare Research**: Understanding misinformation's impact on patient behavior
- **Public Health**: Analyzing social media's role in health decisions
- **Policy Making**: Informing healthcare communication strategies
- **Academic Studies**: Research on digital health literacy and misinformation

---

**⚠️ Disclaimer**: This tool is for research and educational purposes. It should not be used for medical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.

**Made with ❤️ for better healthcare understanding**
