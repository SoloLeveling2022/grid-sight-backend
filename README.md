# GridSight Services

This mono-repository hosts the backend and machine learning services for the **GridSight** parking violation intelligence platform. It operates alongside the Next.js frontend dashboard.

## Services Overview

### 1. `backend/` (Node.js & Express)
* **Description**: Coordinates all data requests, handles telemetry logging, and manages API integrations for violation analytics and voice agent session flows.
* **Setup**:
  ```bash
  cd backend
  npm install
  npm run dev
  ```
* **Configuration**: Configured via [backend/.env](./backend/.env) (default port `3001`).

### 2. `ml-service/` (Python & FastAPI)
* **Description**: Holds congestion model predictions, recommendation algorithms, and discovery clustering modules.
* **Setup**:
  ```bash
  cd ml-service
  python -m venv venv
  source venv/bin/activate  # On Windows: .\venv\Scripts\activate
  pip install -r requirements.txt
  python app/main.py
  ```
* **Configuration**: Configured via [ml-service/.env](./ml-service/.env) (default port `8000`).

### 3. `shared/` (JSON schemas & constants)
* **Description**: Hosts shared validation schemas and key constants, making it easy to enforce schema validation across both JavaScript/Express backend models and Python Pydantic input models.

---
*For frontend setup, refer to the sibling folder `gridsight-app/`.*
