#!/bin/bash
# Quick script to force CPU-only mode

echo "Setting Docling to CPU-only mode..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env first:"
    echo "  cp .env.example .env"
    exit 1
fi

# Add or update DOCLING_GPU_PREFERRED setting
if grep -q "^DOCLING_GPU_PREFERRED=" .env; then
    # Update existing line
    sed -i 's/^DOCLING_GPU_PREFERRED=.*/DOCLING_GPU_PREFERRED=false/' .env
    echo "✓ Updated DOCLING_GPU_PREFERRED=false in .env"
else
    # Add new line at the top after the first comment
    sed -i '2a\\n# GPU/CPU Settings\nDOCLING_GPU_PREFERRED=false' .env
    echo "✓ Added DOCLING_GPU_PREFERRED=false to .env"
fi

echo ""
echo "Current setting:"
grep "^DOCLING_GPU_PREFERRED=" .env

echo ""
echo "To apply changes, restart the service:"
echo "  docker compose restart"
