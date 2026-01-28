import argparse
import pandas as pd

from compat import ensure_supported_python

ensure_supported_python()

from event_extractor import EventExtractor


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ProjektNLP demo")
    p.add_argument(
        "--ollama",
        action="store_true",
        help="Użyj lokalnej Ollamy do klasyfikacji typu zdarzenia (bez trenowania sklearn).",
    )
    p.add_argument(
        "--ollama-model",
        default="qwen2.5:7b-instruct",
        help="Nazwa modelu w Ollamie (np. qwen2.5:7b-instruct).",
    )
    p.add_argument(
        "--ollama-host",
        default="",
        help="Host Ollamy (domyślnie z OLLAMA_HOST albo http://localhost:11434).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if args.ollama:
        from ollama_event_classifier import OllamaEventClassifier
        from ollama_relation_extractor import OllamaRelationExtractor

        extractor = EventExtractor(
            classifier=OllamaEventClassifier(model=args.ollama_model, host=args.ollama_host),
            relations=OllamaRelationExtractor(model=args.ollama_model, host=args.ollama_host),
        )
    else:
        extractor = EventExtractor()
        extractor.train(
            "datasets/id_and_headline_first_sentence (1).csv",
            "datasets/tagged.csv",
        )

    sentences = [
        "Napastnik pobił ochroniarza przed klubem.",
        "Samochód uderzył w drzewo w lesie.",
        "Pożar zniszczył halę magazynową w Poznaniu.",
        "Eksperci komentują sytuację gospodarczą."
    ]

    for s in sentences:
        print("\nZDANIE:", s)
        event = extractor.extract_event(s)
        print(event)


    # print("\n=== TEST NA ZBIORZE TESTOWYM ===")
    #
    # for _, row in test_df.iterrows():
    #     sentence = row["sentence"]
    #
    #     print("\nZDANIE:", sentence)
    #
    #     event = extractor.extract_event(sentence)
    #     print("PREDYKCJA:")
    #     print(event)
    #
    #     print("")
    #     print(f"KTO: {row['who']}")
    #     print(f"TRIGGER: {row['trigger']}")
    #     print(f"CO: {row['what']}")
    #
    #     print("-" * 50)


if __name__ == "__main__":
    main()
