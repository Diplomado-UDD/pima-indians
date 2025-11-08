#!/bin/bash
# Production deployment script for diabetes prediction model

set -e

VERSION=${1:-latest}
REGISTRY=${DOCKER_REGISTRY:-local}

echo "=========================================="
echo "Diabetes Prediction Model Deployment"
echo "Version: $VERSION"
echo "=========================================="

# Build Docker image
echo "Building Docker image..."
docker build -t diabetes-batch-predictor:$VERSION .

# Tag for registry
if [ "$REGISTRY" != "local" ]; then
    echo "Tagging for registry: $REGISTRY"
    docker tag diabetes-batch-predictor:$VERSION $REGISTRY/diabetes-batch-predictor:$VERSION
    docker tag diabetes-batch-predictor:$VERSION $REGISTRY/diabetes-batch-predictor:latest

    echo "Pushing to registry..."
    docker push $REGISTRY/diabetes-batch-predictor:$VERSION
    docker push $REGISTRY/diabetes-batch-predictor:latest
fi

# Run smoke test
echo "Running smoke test..."
docker run --rm \
    -v $(pwd)/data/incoming:/data/incoming:ro \
    -v $(pwd)/data/predictions:/data/predictions \
    -v $(pwd)/models/production:/app/models/production:ro \
    diabetes-batch-predictor:$VERSION \
    --input /data/incoming/test_batch.csv \
    --model-dir /app/models/production

if [ $? -eq 0 ]; then
    echo "✓ Smoke test passed"
    echo "✓ Deployment successful!"
    echo ""
    echo "To run in production:"
    echo "  docker run --rm -v /path/to/data:/data diabetes-batch-predictor:$VERSION \\"
    echo "    --input /data/incoming/batch.csv"
else
    echo "✗ Smoke test failed"
    exit 1
fi
