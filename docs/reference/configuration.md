# Configuration Options for MNIST Digit Predictor API

This document outlines the configuration options available for the MNIST Digit Predictor API. You can customize the behavior of the API by setting various environment variables.

## Environment Variables

### `MODEL_PATH`
- **Description**: Specifies the path to the trained model file.
- **Default**: `dump/version_1/model.pth`
- **Usage**: Set this variable to point to the location of your model file.

### `PORT`
- **Description**: Defines the port on which the API server will run.
- **Default**: `8000`
- **Usage**: Change this variable to run the API on a different port.

### `CORS_ORIGINS`
- **Description**: A comma-separated list of allowed origins for CORS requests.
- **Default**: `*` (allows all origins)
- **Usage**: Set this variable to restrict access to specific domains.

### `LOG_LEVEL`
- **Description**: Sets the logging level for the application.
- **Default**: `INFO`
- **Usage**: Options include `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

## Example Configuration

To run the API with a custom model path and port, you can set the environment variables as follows:

```bash
export MODEL_PATH="path/to/your/model.pth"
export PORT=8080
```

Make sure to adjust these settings according to your deployment environment and requirements.
