# Quick Start Guide for MNIST Digit Predictor API

Welcome to the MNIST Digit Predictor API! This quick start guide will help you get the API up and running with minimal setup.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/mnist-digit-predictor-api.git
   cd mnist-digit-predictor-api
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the API

1. **Set the environment variables:**

   You need to set the `MODEL_PATH` and `PORT` environment variables. You can do this in your terminal:

   ```bash
   export MODEL_PATH="path/to/your/model.pth"
   export PORT=8000  # Optional, defaults to 8000
   ```

2. **Start the API server:**

   Run the following command to start the FastAPI server:

   ```bash
   uvicorn demo.app:app --host 0.0.0.0 --port $PORT --reload
   ```

3. **Access the API:**

   Open your web browser and navigate to `http://localhost:8000` to see the API running. You can also access the interactive API documentation at `http://localhost:8000/docs`.

## Making Predictions

You can make predictions by sending a POST request to the `/predict/image` or `/predict/base64` endpoints with the appropriate data.

Refer to the [API Endpoints](../api/endpoints.md) documentation for more details on how to use these endpoints.

## Conclusion

You are now ready to use the MNIST Digit Predictor API! For more information, check out the other sections of the documentation.
