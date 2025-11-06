# Deployment Guide

## Architecture Overview

This melanoma detection system is now split into two parts:

1. **Backend API** (FastAPI) - Handles ML predictions and data management
2. **Frontend** (Next.js) - User interface that consumes the API

## Deployment Options

### Option 1: Railway (Recommended for Backend)

Railway is perfect for ML applications with larger resource requirements.

#### Backend Deployment on Railway:

1. **Create Railway Account**: Go to [railway.app](https://railway.app)

2. **Deploy from GitHub**:
   ```bash
   # Connect your GitHub repo to Railway
   # Railway will automatically detect the Dockerfile
   ```

3. **Set Environment Variables**:
   ```
   DATABASE_URL=postgresql://...  (Railway provides this)
   SECRET_KEY=your-secret-key-here
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ALLOWED_ORIGINS=https://your-frontend.vercel.app
   ```

4. **Upload Model Files**:
   - Upload your `.pkl` model files to the Railway project
   - Or store them in a cloud storage service like AWS S3

#### Frontend Deployment on Vercel:

1. **Connect to Vercel**:
   ```bash
   cd frontend
   npm install -g vercel
   vercel
   ```

2. **Set Environment Variables**:
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app
   ```

### Option 2: Full Stack on Vercel (Limited)

⚠️ **Note**: Vercel has limitations for ML models due to:
- 50MB deployment size limit
- 10-second function timeout
- Limited memory

If your models are small enough:

1. **Create `api/` folder in your Next.js project**
2. **Move FastAPI endpoints to Vercel functions**
3. **Use Vercel Edge Runtime for better performance**

### Option 3: Docker Deployment

For self-hosting on any cloud provider:

```bash
# Build and run the API
cd api
docker build -t melanoma-api .
docker run -p 8000:8000 melanoma-api

# Build and run the frontend
cd frontend
npm run build
npm start
```

## Environment Setup

### Backend (.env):
```env
DATABASE_URL=postgresql://user:pass@host:5432/db
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

### Frontend (.env.local):
```env
NEXT_PUBLIC_API_URL=https://your-api-domain.com
```

## Database Setup

### PostgreSQL (Recommended for Production):

1. **Get a PostgreSQL instance**:
   - Railway PostgreSQL (automatic)
   - Neon.tech (free tier)
   - Supabase (free tier)
   - AWS RDS

2. **Update DATABASE_URL** in your backend environment

### SQLite (Development Only):
```env
DATABASE_URL=sqlite:///./melanoma_api.db
```

## Model Optimization

### Model Size Reduction:
```python
# Compress models before deployment
import joblib
from sklearn.ensemble import VotingClassifier

# Load your trained model
model = joblib.load('voting_ensemble.pkl')

# Save with compression
joblib.dump(model, 'voting_ensemble_compressed.pkl', compress=3)
```

### Lazy Loading:
```python
# Load models only when needed
@lru_cache(maxsize=1)
def get_model(model_type: str):
    return joblib.load(f'{model_type}_model.pkl')
```

## Security Considerations

1. **API Keys**: Use strong secret keys
2. **CORS**: Configure allowed origins properly
3. **Rate Limiting**: Add rate limiting to prevent abuse
4. **Input Validation**: Validate all user inputs
5. **HTTPS**: Always use HTTPS in production

## Monitoring

### Health Checks:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len([m for m in models.values() if m is not None])
    }
```

### Logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Performance Tips

1. **Use async/await** for I/O operations
2. **Implement caching** for frequently accessed data
3. **Optimize image processing** pipeline
4. **Use CDN** for static assets
5. **Enable gzip compression**

## Scaling

- **Horizontal Scaling**: Deploy multiple API instances
- **Load Balancing**: Use Railway's automatic load balancing
- **Database Scaling**: Use read replicas for heavy read workloads
- **CDN**: Use Vercel's global CDN for frontend assets

## Costs Estimation

### Railway (Backend):
- Hobby Plan: $5/month (512MB RAM, 1GB storage)
- Pro Plan: $20/month (8GB RAM, 100GB storage)

### Vercel (Frontend):
- Hobby: Free (100GB bandwidth)
- Pro: $20/month (1TB bandwidth)

### Database:
- Neon/Supabase: Free tier available
- Railway PostgreSQL: Included in plans

## Troubleshooting

### Common Issues:

1. **Model Loading Errors**: Check file paths and permissions
2. **CORS Issues**: Verify allowed origins configuration
3. **Database Connection**: Check DATABASE_URL format
4. **Image Processing**: Ensure OpenCV dependencies are installed
5. **Memory Issues**: Consider model compression or upgrade plan

### Debug Commands:
```bash
# Check API health
curl https://your-api-domain.com/health

# Test model loading
python -c "import joblib; print(joblib.load('voting_ensemble.pkl'))"

# Check dependencies
pip freeze | grep -E "(fastapi|uvicorn|scikit-learn)"
```

## Next Steps

1. Deploy backend to Railway
2. Deploy frontend to Vercel
3. Set up monitoring and alerts
4. Configure custom domain
5. Set up CI/CD pipeline
6. Add more ML models
7. Implement user analytics
8. Add email notifications