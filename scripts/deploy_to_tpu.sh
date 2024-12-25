#!/bin/bash
set -e

echo "Running full deployment sequence..."

# Run each deployment stage
./scripts/deploy_env.sh
./scripts/deploy_repo.sh
./scripts/deploy_setup.sh

echo "Full deployment complete!"
