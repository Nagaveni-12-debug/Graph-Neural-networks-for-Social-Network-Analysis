# Graph Neural Network for Social Network Analysis

A deep learning based social network analysis system using Heterogeneous Graph Neural Networks for Fake News Detection and User Interaction Analysis. This project models relationships between news articles and tweets using graph structures and applies GraphSAGE and Graph Attention Networks to classify fake and real news with improved accuracy.

## Features

* Fake News Detection using Graph Neural Networks
* Heterogeneous Graph Construction
* Graph Attention Network (GAT) implementation
* GraphSAGE based neighborhood aggregation
* Training and Validation Accuracy Visualization
* Confusion Matrix Generation
* User Post Interaction Network Visualization
* Performance Evaluation using Precision, Recall, F1 Score, and Accuracy

## Project Workflow

1. Data Collection from FakeNewsNet Dataset
2. Text Preprocessing and Cleaning
3. Feature Extraction using TF IDF
4. Heterogeneous Graph Construction
5. Graph Neural Network Training
6. Model Evaluation and Visualization

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* NetworkX
* Matplotlib
* Scikit Learn
* Pandas
* NumPy

## Model Architecture

The proposed model combines:

* GraphSAGE Convolution Layer
* Graph Attention Network Layer
* Fully Connected Classification Layer

The system learns node representations by aggregating information from neighboring nodes and captures relationships between news articles and tweets. 

## Dataset

The project uses the FakeNewsNet dataset containing:

* Real News
* Fake News
* Tweet Interactions
* User Engagement Information

Data preprocessing includes text cleaning, normalization, and TF IDF feature extraction. 

## Experimental Results

The model achieved:

* Accuracy: 92.5%
* Precision: 91.2%
* Recall: 93.1%
* F1 Score: 92.1%

## Generated Outputs

* Training and Validation Accuracy Graph
* Confusion Matrix
* Network Graph Visualization

## Project Structure

```bash
├── engine.py
├── preprocessing.py
├── graph_builder.py
├── gnn_model.py
├── evaluation.py
├── visualize.py
├── training_validation_graph.png
├── confusion_matrix.png
├── network_graph.png
└── README.md
```

## How to Run

```bash
python engine.py
```

## Output Visualizations

The system generates:

* Accuracy vs Epoch graph
* Confusion Matrix
* Heterogeneous Network Graph

## Future Improvements

* Real time fake news detection
* Dynamic graph processing
* Transformer integrated Graph Neural Networks
* Large scale social network deployment

