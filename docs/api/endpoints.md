# MNIST Digit Predictor API Endpoints

This document outlines the various API endpoints available in the MNIST Digit Predictor API, including details about request methods, parameters, and response formats.

## Root Endpoint

### `GET /`

- **Description**: Checks if the API is running.
- **Response**:
  - **Status Code**: 200 OK
  - **Content**:
    ```json
    {
      "message": "MNIST Digit Predictor API is running"
    }
    ```

## Health Check Endpoint

### `GET /health`

- **Description**: Checks the health status of the API and whether the model is loaded.
- **Response**:
  - **Status Code**: 200 OK
  - **Content**:
    ```json
    {
      "status": "healthy",
      "model_loaded": true
    }
    ```
  - **Status Code**: 503 Service Unavailable
  - **Content**:
    ```json
    {
      "status": "unhealthy",
      "model_loaded": false
    }
    ```

## Predict from Image Endpoint

### `POST /predict/image`

- **Description**: Predicts the digit from an uploaded image file.
- **Request**:
  - **Parameters**:
    - `file` (form-data): An image file containing a handwritten digit.
- **Response**:
  - **Status Code**: 200 OK
  - **Content**:
    ```json
    {
      "prediction": <predicted_digit>,
      "confidence": <confidence_score>
    }
    ```
  - **Status Code**: 500 Internal Server Error
  - **Content**:
    ```json
    {
      "detail": "Prediction error: <error_message>"
    }
    ```

## Predict from Base64 Endpoint

### `POST /predict/base64`

- **Description**: Predicts the digit from a base64 encoded image.
- **Request**:
  - **Parameters**:
    - `data` (application/json): JSON object with an 'image' field containing base64 encoded image.
- **Response**:
  - **Status Code**: 200 OK
  - **Content**:
    ```json
    {
      "prediction": <predicted_digit>,
      "confidence": <confidence_score>
    }
    ```
  - **Status Code**: 400 Bad Request
  - **Content**:
    ```json
    {
      "detail": "No image data provided"
    }
    ```
  - **Status Code**: 500 Internal Server Error
  - **Content**:
    ```json
    {
      "detail": "Prediction error: <error_message>"
    }
    ```
