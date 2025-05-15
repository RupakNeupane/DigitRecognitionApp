# Installation Instructions for MNIST Digit Predictor API

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- Python 3.7 or higher
- pip (Python package installer)

## Installation Steps

1. **Clone the Repository**

   Start by cloning the MNIST Digit Predictor API repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/mnist-digit-predictor-api.git
   cd mnist-digit-predictor-api
   ```

2. **Create a Virtual Environment**

   It is recommended to create a virtual environment to manage dependencies:

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**

   Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**

   Set the necessary environment variables. You can create a `.env` file in the root directory of the project and add the following:

   ```
   MODEL_PATH=dump/version_1/model.pth
   PORT=8000
   ```

5. **Run the API**

   You can now run the FastAPI application:

   ```bash
   uvicorn demo.app:app --host 0.0.0.0 --port $PORT --reload
   ```

   The API should now be accessible at `http://localhost:8000`.

## Conclusion

You have successfully installed the MNIST Digit Predictor API and are ready to start using it. For further instructions, refer to the [Quickstart Guide](quickstart.md).
