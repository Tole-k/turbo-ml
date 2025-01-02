#!/bin/sh
#SBATCH --job-name=tb-ml
#SBATCH --partition=obl
#SBATCH --nodes=2
#SBATCH --ntasks=150
#SBATCH --time=7-00:00:00
#SBATCH --comment="Turbo-ML project being done with prof. Dariusz Brzeziński"

VENV_DIR="../.venv"
PYTHON_EXEC="$VENV_DIR/bin/python"
PREFECT_EXEC="$VENV_DIR/bin/prefect"
$PYTHON_EXEC -m turbo_ml.workflow
