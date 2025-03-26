# PixelCut AI - Advanced Image Processing Web Application

An AI-powered web application for advanced image processing, featuring background removal, image enhancement, and AI filters.

## Features

- Background Removal (automatic + manual refinement)
- Image Enhancement (upscaling, denoising, sharpening)
- AI Filters (artistic styles, vintage effects)
- Real-time Preview
- User-friendly Interface

## Tech Stack

### Frontend
- Next.js 14 with React
- TypeScript
- Tailwind CSS
- Zustand (State Management)
- React Dropzone
- HTML5 Canvas

### Backend
- FastAPI (Python)
- rembg for Background Removal
- Real-ESRGAN for Image Enhancement
- ONNX Runtime
- PyTorch

### Infrastructure
- Docker
- AWS/Render (Backend)
- Vercel (Frontend)
- Firebase Storage/AWS S3

## Setup Instructions

### Prerequisites
- Node.js 18+
- Python 3.9+
- Docker

### Frontend Setup
1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

### Backend Setup
1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the server:
```bash
uvicorn main:app --reload
```

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_STORAGE_BUCKET=your-bucket-name
```

### Backend (.env)
```
MODEL_PATH=./models
REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

## Docker Deployment
```bash
docker-compose up --build
```

## License
MIT 