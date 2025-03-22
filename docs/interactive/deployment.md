# Deploying the Interactive Demo

This page provides guidance on deploying the DAPS interactive demo to various platforms.

## Streamlit Cloud (Recommended)

The easiest way to deploy the demo is using [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your DAPS repository to GitHub
2. Sign up for a Streamlit Cloud account
3. Connect your GitHub repository
4. Configure the app:
   - Main file path: `interactive/app.py`
   - Python version: 3.8 or later
   - Requirements: Add both `interactive/requirements.txt` and the root package

Streamlit Cloud will automatically deploy your app and provide a public URL (like `https://daps-demo.streamlit.app`).

## Heroku Deployment

To deploy on Heroku:

1. Create a new Heroku app
2. Set up a `Procfile` in the repository root:
   ```
   web: cd interactive && streamlit run app.py --server.port=$PORT
   ```
3. Configure the buildpacks:
   ```bash
   heroku buildpacks:add heroku/python
   ```
4. Create a `runtime.txt` with your Python version:
   ```
   python-3.9.16
   ```
5. Deploy your app:
   ```bash
   git push heroku main
   ```

## Docker Deployment

For Docker-based deployment:

1. Create a `Dockerfile` in the repository root:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Copy repository
   COPY . .
   
   # Install DAPS and requirements
   RUN pip install -e .
   RUN pip install -r interactive/requirements.txt
   
   # Expose port
   EXPOSE 8501
   
   # Set working directory to interactive
   WORKDIR /app/interactive
   
   # Run the app
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t daps-demo .
   docker run -p 8501:8501 daps-demo
   ```

## Self-Hosted Deployment

To deploy on your own server:

1. Clone the repository:
   ```bash
   git clone https://github.com/sethuiyer/DAPS.git
   cd DAPS
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   pip install -r interactive/requirements.txt
   ```

3. Configure a reverse proxy (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name daps-demo.yourdomain.com;
   
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header Host $host;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 86400;
       }
   }
   ```

4. Run the app with a process manager like Supervisor:
   ```ini
   [program:daps-demo]
   command=/path/to/venv/bin/streamlit run app.py
   directory=/path/to/DAPS/interactive
   autostart=true
   autorestart=true
   stderr_logfile=/var/log/daps-demo.err.log
   stdout_logfile=/var/log/daps-demo.out.log
   user=your_user
   ```

## Performance Considerations

When deploying the demo:

1. **Memory Usage**: The demo can use significant memory when evaluating complex functions or with high-resolution grids.
2. **CPU Load**: 3D visualizations and function evaluations can be CPU-intensive.
3. **Scaling**: For high traffic, consider using a load balancer with multiple instances.

## Security Considerations

When deploying publicly:

1. Enable CORS protection in the Streamlit config
2. Use HTTPS for all traffic
3. Consider adding authentication if needed
4. Limit the maximum resolution of prime numbers to prevent resource exhaustion

## Monitoring

Once deployed, monitor:

1. CPU and memory usage
2. Response times
3. Error rates
4. User engagement

Most platforms provide built-in monitoring tools, or you can use services like Datadog, Prometheus, or New Relic. 