import pandas as pd

from compat import ensure_supported_python

ensure_supported_python()

from event_extractor import EventExtractor


def main():
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
