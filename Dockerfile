FROM python:3.11-slim

# Create a non-root user for security
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy dependency specification and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Install the application and its dependencies
RUN pip install --no-cache-dir .

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "interview_practice_partner.main:app", "--host", "0.0.0.0", "--port", "8000"]
