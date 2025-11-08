# DebateLab

A web application for practicing debate with an AI opponent. Create structured debates with multiple rounds, submit turns via text or audio transcription, and get AI-generated responses.

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation and models
- **OpenAI API** - GPT-4o-mini for debate responses and Whisper-1 for audio transcription
- **python-dotenv** - Environment variable management

### Frontend
- **React** - UI library
- **Vite** - Build tool and dev server
- **Fetch API** - For HTTP requests

## Prerequisites

- Python 3.9 or higher
- Node.js 16+ and npm (or yarn)
- pip (Python package manager)
- OpenAI API key (optional, but required for full functionality)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-Debate-Tutor
```

### 2. Install Python Dependencies

Install the required packages:

```bash
pip install fastapi uvicorn openai pydantic python-dotenv
```

Or if you prefer using a virtual environment (recommended):

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows
pip install fastapi uvicorn openai pydantic python-dotenv
```

## Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```bash
cd backend
touch .env
```

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** 
- The variable name must be `OPENAI_API_KEY` (not `OPEN_API_KEY`)
- The `.env` file is already in `.gitignore` to keep your API key secure
- The app will work without an API key but will use stub responses for AI turns

## Running the Application

### 1. Start the Backend Server

Navigate to the `backend` directory and start the server:

```bash
cd backend
python3 -m uvicorn app.main:app --reload --port 8000
```

**Note:** Use `python3 -m uvicorn` instead of just `uvicorn` if the command is not found in your PATH.

The server will start at `http://localhost:8000`. You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 2. Install Frontend Dependencies

Navigate to the `frontend` directory and install Node.js dependencies:

```bash
cd frontend
npm install
```

### 3. Start the Frontend Development Server

```bash
npm run dev
```

The React app will start at `http://localhost:3000` and should automatically open in your browser.

The frontend is configured to connect to `http://localhost:8000` by default. You can change the API base URL in the UI if needed.

## API Endpoints

- `GET /v1/health` - Health check
- `POST /v1/debates` - Create a new debate
- `GET /v1/debates/{debate_id}` - Get debate state and messages
- `POST /v1/debates/{debate_id}/turns` - Submit a turn
- `POST /v1/debates/{debate_id}/auto-turn` - Generate AI assistant turn
- `POST /v1/debates/{debate_id}/finish` - Finish debate early
- `POST /v1/transcribe` - Transcribe audio file

## Troubleshooting

### "command not found: uvicorn"

Use `python3 -m uvicorn` instead of just `uvicorn`:
```bash
python3 -m uvicorn app.main:app --reload --port 8000
```

### "Address already in use" (Port 8000)

Kill the process using port 8000:
```bash
lsof -ti:8000 | xargs kill
```

Or use a different port:
```bash
python3 -m uvicorn app.main:app --reload --port 8001
```
Then update the API base URL in the frontend UI.

### Python Version Compatibility

This project requires Python 3.9+. If you encounter syntax errors with `str | None`, ensure you're using Python 3.9 or higher. The code uses `Optional[str]` for compatibility.

### API Key Not Working

- Verify the `.env` file is in the `backend` directory
- Check that the variable name is exactly `OPENAI_API_KEY` (case-sensitive)
- Restart the server after creating or modifying the `.env` file
- Remove quotes if your API key is wrapped in quotes (some systems handle this differently)

## Development

### Frontend Development

The frontend uses Vite for fast development with hot module replacement. The app will automatically reload when you make changes.

To build for production:
```bash
cd frontend
npm run build
```

The built files will be in the `frontend/dist` directory.

### Backend Development

The backend uses in-memory storage, so data is not persisted between server restarts. This is suitable for development and testing.

For production deployment, consider:
- Adding a database (PostgreSQL, MongoDB, etc.)
- Implementing user authentication
- Adding rate limiting
- Setting up proper CORS configuration
- Using environment-specific configuration

## License

[Add your license here]

