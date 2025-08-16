# Cognitive Health Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for predicting cognitive health scores (PHQ-8) using multimodal data analysis and misinformation spread simulation. This project combines advanced ML techniques with network analysis to provide both predictive insights and resource allocation recommendations.

## 🚀 Features

### Core Functionality
- **Multimodal Data Processing**: Handles audio, facial keypoints, gaze confidence, pose confidence, and text data
- **PHQ-8 Score Prediction**: Machine learning pipeline for depression severity assessment
- **Feature Engineering**: Intelligent feature summarization and dimensionality reduction
- **Model Explainability**: SHAP and LIME integration for interpretable predictions
- **Resource Allocation**: Priority-based treatment capacity management

### Advanced Analytics
- **Misinformation Spread Simulation**: Network-based SIR model for risk assessment
- **Dynamic Risk Adjustment**: Real-time risk scoring with misinformation factors
- **Interactive Visualizations**: Streamlit-based dashboard with real-time updates
- **Performance Metrics**: Comprehensive evaluation with MAE, RMSE, and R² scores

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended for large datasets
- GPU support optional (CPU-based processing supported)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cognitive-prediction.git
   cd cognitive-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
cognitive-prediction/
├── predictive_cognitive.py    # Main application file
├── data/                      # Data directory
│   ├── data_csv/             # CSV label files
│   ├── train/                # Training data features
│   ├── valid/                # Validation data features
│   └── test/                 # Test data features
├── artifacts/                 # Generated outputs
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Usage

### Streamlit Web Application (Recommended)

Launch the interactive web interface:

```bash
streamlit run predictive_cognitive.py
```

**Features:**
- Real-time model training and evaluation
- Interactive parameter adjustment
- Dynamic visualizations and charts
- Patient-level explanations
- Resource allocation simulation

### Command Line Interface

Run the pipeline in CLI mode:

```bash
python predictive_cognitive.py --mode cli --capacity 20 --steps 30
```

**CLI Options:**
- `--mode`: Choose between `cli` or `app` (default: `app`)
- `--trans-prob`: Misinformation transmission probability (default: 0.2)
- `--rec-prob`: Recovery probability (default: 0.1)
- `--steps`: Simulation steps (default: 20)
- `--capacity`: Treatment capacity (default: 10)
- `--output-dir`: Output directory for artifacts (default: `artifacts`)

## 🔧 Configuration

### Environment Variables

Set the data directory path:

```bash
export PREDICT_DATA_DIR="/path/to/your/data"
```

### Data Format Requirements

**Dataset Download:**
Download the required dataset from: [TramCam DAIC-WOZ-E Dataset on Kaggle](https://www.kaggle.com/datasets/trilism/tramcam-daic-woz-e?resource=download)

**CSV Files:**
- `train_split_Depression_AVEC2017.csv`
- `dev_split_Depression_AVEC2017.csv`
- `full_test_split.csv`

**Required Columns:**
- `Participant_ID`: Unique participant identifier
- `PHQ8_Score`: Depression severity score (0-24)

**Feature Files:**
- Format: `.npy` files
- Naming: `{split}_ft_{modality}_{pid}.npy`
- Modalities: audio, fkps, gaze_conf, pose_conf, text

## 📊 Model Architecture

### Feature Processing Pipeline
1. **Multimodal Loading**: Automatic detection and loading of available modalities
2. **Feature Summarization**: Statistical aggregation and dimensionality reduction
3. **Alignment**: Consistent feature matrix construction across participants
4. **Scaling**: StandardScaler normalization for optimal model performance

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Hyperparameters**: 500 estimators, optimized for regression tasks
- **Cross-validation**: Train/validation/test split strategy
- **Feature Selection**: Automatic handling of missing modalities

### Explainability Methods
- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Instance-specific explanations

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Mean Absolute Error (MAE)**: Average prediction deviation
- **Root Mean Square Error (RMSE)**: Standard deviation of prediction errors
- **R² Score**: Coefficient of determination (0-1, higher is better)

## 🌐 Misinformation Simulation

### SIR Model Implementation
- **Susceptible (S)**: Participants not exposed to misinformation
- **Infected (I)**: Participants currently affected by misinformation
- **Recovered (R)**: Participants who have overcome misinformation

### Network Properties
- **Topology**: Barabási-Albert scale-free network
- **Dynamics**: Time-evolving state transitions
- **Risk Assessment**: Real-time misinformation spread calculation

## 📁 Output Artifacts

Generated files are saved in the `artifacts/` directory:

- `severity_model.pkl`: Trained model and scaler
- `validation_plot.png`: Predicted vs actual PHQ-8 scores
- `risk_heatmap_cli.png`: Risk distribution visualization
- `explain_*.png`: SHAP/LIME explanation plots

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings for new functions
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AVEC 2017 Dataset**: Depression assessment data
- **SHAP**: Model explainability framework
- **LIME**: Local interpretability methods
- **Streamlit**: Interactive web application framework
- **NetworkX**: Network analysis and visualization

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/yourusername/cognitive-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cognitive-prediction/discussions)
- **Email**: your.email@example.com

## 🔮 Future Roadmap

- [ ] Deep learning model integration
- [ ] Real-time data streaming support
- [ ] Advanced network analysis algorithms
- [ ] Mobile application development
- [ ] API endpoint for external integrations
- [ ] Multi-language support

---

**Note**: This system is designed for research and educational purposes. Always consult healthcare professionals for clinical decisions.

**Last Updated**: August 2025
**Version**: 1.0.0
