#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o integration_tests_%j.out
cd ~/projects/kitt/kitt
pytest -m slow tests/integration/test_ml.py::TestIntegrationTests::test_ml_infer_segmentation
