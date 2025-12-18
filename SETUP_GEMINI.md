# DigiPal - Google Gemini API Setup

## Migration to Google Gemini API

This project has been updated to use Google's Gemini API instead of Ollama's llama3.2:1b model.

### Prerequisites

1. **Google AI Studio API Key**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and replace `your_google_api_key_here` with your actual Google API key:
     ```
     GOOGLE_API_KEY=your_actual_api_key_here
     GOOGLE_MODEL_NAME=gemini-pro
     ```

3. **For Production Deployment**:
   - Set the `GOOGLE_API_KEY` environment variable in your production environment
   - Do not commit your `.env` file to version control

### Changes Made

- **Model**: Replaced `llama3.2:1b` (Ollama) with `gemini-pro` (Google Gemini)
- **Embeddings**: Replaced `nomic-embed-text` (Ollama) with `models/embedding-001` (Google)
- **Dependencies**: Replaced `langchain-ollama` with `langchain-google-genai`
- **Environment Variables**: Added support for `.env` file with API key configuration

### Running the Application

1. **Command Line Interface**:
   ```bash
   python chat.py
   ```

2. **Web UI**:
   ```bash
   python UI.py
   ```

### Security Notes

- Keep your Google API key secure and never commit it to version control
- Use environment variables for production deployments
- The `.env` file is included in `.gitignore` to prevent accidental commits

### API Usage and Costs

- Google Gemini API has usage limits and may incur costs
- Monitor your usage in the [Google AI Studio](https://makersuite.google.com/)
- Consider implementing rate limiting for production use