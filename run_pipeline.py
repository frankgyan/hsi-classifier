import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.pipeline_orchestrator import PipelineOrchestrator

MODEL_PATH = "./models/model.keras"
CONFIG_PATH = "./config/pipeline_config.yaml"

orchestrator = PipelineOrchestrator(CONFIG_PATH)
orchestrator.initialize(MODEL_PATH)

print("=" * 60)
print("STARTING PRODUCTION PIPELINE")
print("=" * 60)

stats = orchestrator.run_batch()

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"Total processed: {stats['total_processed']}")
print(f"Successful: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"QA failed: {stats['qa_failed']}")
print(f"Results saved to: ./output/")