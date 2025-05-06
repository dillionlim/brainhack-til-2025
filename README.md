# Brainhack Repository for TIL-AI 2025

## Prerequisites

*   Python (v3.13 recommended)
*   UV (Python package and project installer)

This project makes use of [UV](https://github.com/astral-sh/uv), an extremely fast Python package and project manager.

For installation of UV, the following is reproduced for convenience:

### UV Installation

Install uv with UV's standalone installers:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If installed via the standalone installer, uv can update itself to the latest version:
```bash
uv self update
```

## Directory Structure

```
root/
|
├── README.md            # This file
├── asr                  # Example (ASR Folder)
└── cv                   # Example (CV Folder)
    └── README.md        # More notes about running CV independently
```

## Version Control

Do make heavy use of the branches to test and experiment with new models.

The `main` branch should be the finalised branch.

## Setup

### Backend (Python FastAPI)

1.  **Create and activate a virtual environment (Recommended):**
    Make sure to install a Python 3.13 virtual environment!
    ```bash
    uv venv --python 3.13 # using uv
    source .venv/bin/activate # On Windows use `venv\\Scripts\\activate`
    ```

2.  **Install dependencies:**
    From the root folder:
    ```bash
    uv sync
    ```

3.  **Environment Variables:**
    *   Ensure you have a `.env` file in the root directory or have the necessary environment variables set globally. Key variables likely include:
        *   `TEAM_NAME` for the team name
        *   `TEAM_TRACK` for the team track
        and so on...

    Do look at the `.env.example` file for an example environment file. 

## Running the Applications

1.  **Start the Competition Server:**
    *   Make sure you are in the root directory and your virtual environment is activated.
    *   Run the FastAPI application using Uvicorn:
        For UV,

        ```bash
        uv run uvicorn ...:app --reload --port 8000
        ```
    *   The server will be available at ...

2.  **Call the Server**
    * ...

## Run dockerised container

1. Run ....
