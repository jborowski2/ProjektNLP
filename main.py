import pandas as pd

from event_extractor import EventExtractor


def main():
    extractor = EventExtractor()
    extractor.train("datasets/training_data.csv")

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

    #test_df = pd.read_csv("datasets/test_relations.csv")

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
