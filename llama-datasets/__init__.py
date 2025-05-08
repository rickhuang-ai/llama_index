import asyncio

from llama_index.core.llama_dataset import download_llama_dataset

async def download_all_datasets():
    datasets = [
        "Uber10KDataset2021",
        "BlockchainSolanaDataset",
        "BraintrustCodaHelpDeskDataset",
        "CovidQaDataset",
        "DocugamiKgRagSec10Q",
        "EvaluatingLlmSurveyPaperDataset",
        "HistoryOfAlexnetDataset",
        "Llama2PaperDataset",
        "MiniCovidQaDataset",
        "MiniEsgBenchDataset",
        "MiniSquadV2Dataset",
        "MiniTruthfulQADataset",
        "MtBenchHumanJudgementDataset",
        "OriginOfCovid19Dataset",
        "PatronusAIFinanceBenchDataset",
        "PaulGrahamEssayDataset",
    ]

    for dataset in datasets:
        print(f"Downloading {dataset}...")
        await download_llama_dataset(dataset, "./data")
        print(f"Downloaded {dataset} to ./data")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_all_datasets())