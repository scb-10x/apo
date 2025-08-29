import argparse
from dotenv import load_dotenv
from loguru import logger
from tasks.task2_runner import Task2Runner

load_dotenv()

# Configure loguru for better console output
logger.remove()  # Remove default handler
logger.add(
    sink=lambda message: print(message, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="results/task2_responses.json")
    parser.add_argument("--data_path", type=str, default="data/task2_sample.json")
    parser.add_argument(
        "--no_evaluation", action="store_true", help="Disable evaluation"
    )
    args = parser.parse_args()

    logger.info("🚀 Starting Task 2 execution")
    logger.info(f"📂 Data path: {args.data_path}")
    logger.info(f"💾 Save path: {args.save_path}")
    logger.info(f"📊 Evaluation enabled: {not args.no_evaluation}")

    try:
        runner = Task2Runner()
        runner.process_conversations(
            data_path=args.data_path,
            save_path=args.save_path,
            enable_evaluation=not args.no_evaluation,
        )
        logger.success("✅ Task 2 completed successfully!")
    except Exception as e:
        logger.error(f"❌ Task 2 failed: {e}")
        raise
