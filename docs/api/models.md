# File: /mnist-api-docs/mnist-api-docs/docs/api/models.md

# MNIST Digit Predictor API Models

This document outlines the data models used in the MNIST Digit Predictor API. The API utilizes specific data structures for requests and responses to ensure consistency and clarity.

## Request Models

### Image Upload Model

The model for uploading an image file for digit prediction is defined as follows:

- **file**: `UploadFile`
  - Type: File
  - Description: An image file containing a handwritten digit.

### Base64 Image Model

The model for sending a base64 encoded image for digit prediction is defined as follows:

- **data**: `dict`
  - **image**: `str`
    - Type: Base64 encoded string
    - Description: A base64 encoded image string representing a handwritten digit.

## Response Models

### Prediction Response Model

The response model for the prediction endpoint is defined as follows:

- **digit**: `int`
  - Type: Integer
  - Description: The predicted digit (0-9).

- **confidence**: `float`
  - Type: Float
  - Description: The confidence score of the prediction, ranging from 0 to 1.

### Health Check Response Model

The response model for the health check endpoint is defined as follows:

- **status**: `str`
  - Type: String
  - Description: The health status of the API (e.g., "healthy", "unhealthy").

- **model_loaded**: `bool`
  - Type: Boolean
  - Description: Indicates whether the model is loaded and ready for predictions.
