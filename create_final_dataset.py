#!/usr/bin/env python3
"""
Create a comprehensive dataset with 1000+ examples for event classification.
This script combines existing data with new synthetic examples.
"""

import pandas as pd
import random

# Read existing data
existing_df = pd.read_csv('datasets/training_data.csv')
print(f"Starting with {len(existing_df)} existing examples")

# Read our extended data
extended_df = pd.read_csv('datasets/training_data_extended.csv')
print(f"Extended dataset has {len(extended_df)} examples")

# Now add more examples to each category to reach our target
# We need approximately 50+ examples per category (20 categories * 50 = 1000)

additional_examples = []

# Helper function to add examples
def add_examples(label, sentences):
    for sentence in sentences:
        additional_examples.append({'sentence': sentence, 'label': label})

# Add more examples for each category

# PRZESTĘPSTWO (more examples)
add_examples('PRZESTĘPSTWO', [
    "Mężczyzna okradł sklep jubilerski.",
    "Sprawca włamał się do samochodu na parkingu.",
    "Złodzieje ukradli towar ze sklepu.",
    "Oszust wyłudził dane karty kredytowej.",
    "Wandale zniszczyli witrynę sklepową.",
    "Przestępca zaatakował przypadkową osobę.",
    "Rabuś napadł na bank z bronią w ręku.",
    "Gang dokonał napadu na konwój.",
    "Złodziej ukradł portfel w autobusie.",
    "Włamywacz dostał się przez balkon.",
    "Sprawca pobił przechodnia na ulicy.",
    "Oszuści wyłudzili pieniądze przez telefon.",
    "Przestępcy porwali biznesmen dla okupu.",
    "Złodzieje ukradli motocykl z garażu.",
    "Wandal zniszczył monument w parku.",
    "Rabuś napadł na stację benzynową.",
    "Gang ukradł samochody luksusowe.",
    "Włamywacz okradł biuro w nocy.",
    "Sprawca zaatakował nożem w metrze.",
    "Oszuści wyłudzili kredyty bankowe.",
])

# WYPADEK (more examples)
add_examples('WYPADEK', [
    "Auto uderzyło w słup na drodze.",
    "Motocykl wpadł pod samochód osobowy.",
    "Ciężarówka zderzyła się z busem.",
    "Kierowca zjechał na czerwonym świetle.",
    "Rowerzysta wpadł pod tramwaj.",
    "Samochód rozbił się o barierę ochronną.",
    "Pociąg zderzył się z samochodem na przejeździe.",
    "Kierowca stracił kontrolę na zakręcie.",
    "Auto wpadło do kanału.",
    "Pojazd uderzył w przydrożne drzewo.",
    "Motocykl przewrócił się na mokrej drodze.",
    "Samochód wjechał w tył autobusu.",
    "Kierowca nie ustąpił pierwszeństwa.",
    "Auto zderzyło się z ciężarówką czołowo.",
    "Pojazd spadł z mostu do rzeki.",
    "Samochód uderzył w budynek mieszkalny.",
    "Kierowca zjechał z drogi na ślisko.",
    "Auto uderzyło w znak drogowy.",
    "Motocykl wpadł pod samochód dostawczy.",
    "Kierowca nie zauważył pieszego na przejściu.",
])

# POŻAR (more examples)
add_examples('POŻAR', [
    "Ogień pojawił się w bloku mieszkalnym.",
    "Pożar zniszczył hale magazynową.",
    "Płomienie objęły fabrykę mebli.",
    "Ogień strawił restaurację w centrum.",
    "Pożar wybuchł w hali produkcyjnej.",
    "Płomienie zniszczyły stodołę z sianem.",
    "Ogień objął budynek biurowy.",
    "Pożar strawił las iglasty.",
    "Płomienie zniszczyły sklep odzieżowy.",
    "Ogień wybuchł w garażu podziemnym.",
    "Pożar objął dom jednorodzinny.",
    "Płomienie strawiły budynek gospodarczy.",
    "Ogień zniszczył warsztat mechaniczny.",
    "Pożar wybuchł w kamienicy przy rynku.",
    "Płomienie objęły stację benzynową.",
    "Ogień strawił tartак drewna.",
    "Pożar zniszczył hotel przy plaży.",
    "Płomienie objęły zakład produkcyjny.",
    "Ogień wybuchł w sklepie spożywczym.",
    "Pożar strawił budynek szkoły.",
])

# POLITYKA (more examples)
add_examples('POLITYKA', [
    "Sejm uchwalił nowe prawo wyborcze.",
    "Minister przedstawił projekt ustawy.",
    "Rząd przyjął strategię rozwoju kraju.",
    "Prezydent spotkał się z kanclerzem.",
    "Parlament odrzucił propozycję opozycji.",
    "Minister zdrowia ogłosił reformę systemu.",
    "Sejm przyjął uchwałę o stanie klęski.",
    "Rząd zapowiedział zmiany w podatkach.",
    "Prezydent odwołał ambasadora.",
    "Parlament debatował nad budżetem państwa.",
    "Minister finansów przedstawił plan oszczędności.",
    "Sejm uchwalił ustawę o ochronie środowiska.",
    "Rząd podpisał umowę międzynarodową.",
    "Prezydent rozpoczął konsultacje z partiami.",
    "Parlament przyjął nowelizację kodeksu.",
    "Minister spraw zagranicznych odbył wizytę.",
    "Sejm odrzucił wniosek o wotum nieufności.",
    "Rząd powołał nową komisję.",
    "Prezydent wziął udział w szczycie.",
    "Parlament uchwalił prawo o referendum.",
])

# SPORT (more examples)
add_examples('SPORT', [
    "Zawodnik zdobył medal na olimpiadzie.",
    "Drużyna awansowała do półfinału.",
    "Piłkarz strzelił dwa gole w meczu.",
    "Siatkarze wygrali trzeciego seta.",
    "Lekkoatleta pobił rekord Polski.",
    "Reprezentacja zremisowała z rywalem.",
    "Koszykarz zdobył 25 punktów.",
    "Hokeista asystował przy trzech bramkach.",
    "Pływak osiągnął czas kwalifikacyjny.",
    "Kolarz wygrał sprint na mecie.",
    "Tenisista przegrał w trzecim secie.",
    "Skoczek narciarski zajął trzecie miejsce.",
    "Piłkarka została MVP turnieju.",
    "Drużyna zdobyła Puchar Polski.",
    "Zawodnik zakwalifikował się do finału.",
    "Reprezentacja przegrała po dogrywce.",
    "Koszykarka zdobyła double-double.",
    "Hokeista został zawieszony na mecz.",
    "Pływaczka ustanowiła nowy rekord świata.",
    "Kolarz dojechał w czołówce stawki.",
])

# EKONOMIA (more examples)
add_examples('EKONOMIA', [
    "Spółka wypłaciła dywidendę akcjonariuszom.",
    "Bank ogłosił wyniki finansowe za kwartał.",
    "Giełda zamknęła się na plusie.",
    "Firma uruchomiła nową linię produkcyjną.",
    "Akcje spółki wzrosły o dziesięć procent.",
    "Bank obniżył oprocentowanie kredytów.",
    "Koncern przeprowadził restrukturyzację.",
    "Giełda odnotowała rekordowe wolumeny.",
    "Przedsiębiorstwo zainwestowało w robotyzację.",
    "Kurs dolara osiągnął szczyt.",
    "Firma ogłosiła program zwolnień.",
    "Bank wprowadził nową ofertę kredytową.",
    "Spółka zwiększyła wydatki na rozwój.",
    "Giełda reaguje na dane makroekonomiczne.",
    "Przedsiębiorstwo podpisało kontrakt eksportowy.",
    "Akcje spółki technologicznej spadły.",
    "Bank centralny utrzymał stopy bez zmian.",
    "Firma przejęła konkurencyjne przedsiębiorstwo.",
    "Giełda zamknęła tydzień ze spadkami.",
    "Spółka wypłaciła premie pracownikom.",
])

# NAUKA (more examples)
add_examples('NAUKA', [
    "Badacze odkryli nową metodę syntezy.",
    "Naukowcy opublikowali artykuł w Nature.",
    "Zespół naukowy otrzymał nagrodę.",
    "Eksperci odkryli nowy minerał.",
    "Laboratorium przetestowało prototyp.",
    "Badacze zmapowali strukturę białka.",
    "Naukowcy zaobserwowali nową gwiazdę.",
    "Instytut zakupił teleskop kosmiczny.",
    "Uczeni odkryli przyczynę zjawiska.",
    "Badacze opracowali innowacyjny algorytm.",
    "Naukowcy stworzyli nowy materiał.",
    "Zespół naukowy przeprowadził eksperyment.",
    "Eksperci odkryli fossile dinozaura.",
    "Laboratorium opracowało nową terapię.",
    "Badacze udowodnili hipotezę naukową.",
    "Naukowcy sklonowali komórki macierzyste.",
    "Instytut badawczy uruchomił akcelerator.",
    "Uczeni zidentyfikowali nowy gen.",
    "Badacze odkryli egzoplanetę.",
    "Naukowcy stworzyli szczepionkę mRNA.",
])

# KLĘSKA_ŻYWIOŁOWA (more examples)
add_examples('KLĘSKA_ŻYWIOŁOWA', [
    "Rzeka wylała i zalała pola.",
    "Trzęsienie ziemi uszkodziło budynki.",
    "Tornado zniszczyło gospodarstwa.",
    "Osuwisko ziemi zablokowało drogę.",
    "Huragan zniszczył infrastrukturę.",
    "Susza dotknęła region rolniczy.",
    "Lawina zasypała szlak górski.",
    "Fala powodziowa zalała miasto.",
    "Erupcja wulkanu wyrzuciła popiół.",
    "Burza zniszczyła uprawy.",
    "Tsunami uderzyło w wybrzeże.",
    "Grad zniszczył sady owocowe.",
    "Ziemia osunęła się pod domami.",
    "Cyklon przeszedł przez wyspę.",
    "Ulewa spowodowała powódź błyskawiczną.",
    "Mróz zniszczył plantacje warzyw.",
    "Wichura powaliła drzewa w lesie.",
    "Deszcz spowodował podtopienia.",
    "Oblodzenie zablokowało drogi.",
    "Burza zniszczyła sieć elektryczną.",
])

# Continue with more categories...
# MEDYCYNA
add_examples('MEDYCYNA', [
    "Lekarz zdiagnozował rzadką chorobę.",
    "Szpital przyjął ofiary katastrofy.",
    "Chirurg przeprowadził operację serca.",
    "Pacjent przeszedł transplantację wątroby.",
    "Ratownicy uratowali tonącego.",
    "Lekarka wykryła nowotwór wcześnie.",
    "Szpital otworzył oddział intensywnej terapii.",
    "Chirurg wykonał operację mózgu.",
    "Pacjent rozpoczął chemioterapię.",
    "Lekarz przepisał nowe leczenie.",
    "Szpital zakupił tomograf komputerowy.",
    "Ratownicy udzielili pomocy poszkodowanym.",
    "Lekarka pomogła urodzić bliźnięta.",
    "Chirurg naprawił uszkodzony narząd.",
    "Pacjent przeszedł długą rehabilitację.",
    "Szpital uruchomił program profilaktyczny.",
    "Lekarz wykrył wczesne objawy choroby.",
    "Ratownicy przetransportowali pacjenta.",
    "Chirurg usunął guz nowotworowy.",
    "Pacjent wybudził się ze śpiączki.",
])

# PROTESTY
add_examples('PROTESTY', [
    "Górnicy protestowali przed ministerstwem.",
    "Studenci zorganizowali protest.",
    "Rolnicy zablokowali główną drogę.",
    "Pracownicy strajkują o podwyżki.",
    "Manifestacja przeszła ulicami miasta.",
    "Związki zawodowe ogłosiły strajk.",
    "Obywatele demonstrowali przeciwko ustawie.",
    "Protest klimatyczny zgromadził młodzież.",
    "Nauczyciele strajkują o wyższe płace.",
    "Demonstranci zablokowali wejście do budynku.",
    "Protest przeciw dyskryminacji.",
    "Strajk sparaliżował przemysł.",
    "Manifestacja domagała się reform.",
    "Rolnicy protestują przeciw importowi.",
    "Pracownicy ochrony zdrowia strajkują.",
    "Demonstracja wsparcia dla uchodźców.",
    "Protest przeciwko korupcji.",
    "Strajkujący zablokowali magistralę.",
    "Manifestacja żądała dymisji ministra.",
    "Protest studentów przeciw opłatom za studia.",
])

# KULTURA
add_examples('KULTURA', [
    "Reżyser nakręcił nowy film fabularny.",
    "Malarz zaprezentował obrazy w galerii.",
    "Muzyk zagrał koncert na żywo.",
    "Teatr wystawił klasyczną sztukę.",
    "Pisarz wydał nową powieść.",
    "Festiwal przyciągnął tysiące widzów.",
    "Artysta odsłonił rzeźbę w parku.",
    "Koncert symfoniczny odbył się w operze.",
    "Fotograf wystawił swoje prace.",
    "Aktor otrzymał nagrodę za rolę.",
    "Muzeum otworzyło wystawę sztuki.",
    "Spektakl baletowy zachwycił publiczność.",
    "Galeria kupiła dzieło sztuki.",
    "Autor książki spotkał się z czytelnikami.",
    "Festiwal filmowy przyznał nagrody.",
    "Muzyk wydał nowy album.",
    "Teatr zorganizował premiery spektakl.",
    "Malarz otrzymał prestiżowe wyróżnienie.",
    "Koncert rockowy zgromadził fanów.",
    "Wystawa fotografii trwa do końca miesiąca.",
])

# TECHNOLOGIA
add_examples('TECHNOLOGIA', [
    "Startup opracował nową aplikację mobilną.",
    "Firma wprowadziła sztuczną inteligencję.",
    "Programiści wydali aktualizację oprogramowania.",
    "Koncern zaprezentował nowy laptop.",
    "Naukowcy stworzyli chip kwantowy.",
    "Firma technologiczna uruchomiła chmurę.",
    "Startup otrzymał inwestycję.",
    "Programiści naprawili błędy bezpieczeństwa.",
    "Koncern wprowadził nowy system operacyjny.",
    "Firma opracowała technologię blockchain.",
    "Startup stworzył platformę e-learning.",
    "Programiści zintegrowali systemy.",
    "Koncern kupił firmę technologiczną.",
    "Naukowcy testują komputer kwantowy.",
    "Firma wprowadziła sieć 5G.",
    "Startup opracował chatbota AI.",
    "Programiści stworzyli framework open-source.",
    "Koncern zainwestował w badania AI.",
    "Firma uruchomiła centrum danych.",
    "Startup rozwija platformę IoT.",
])

# PRAWO
add_examples('PRAWO', [
    "Sędzia ogłosił wyrok skazujący.",
    "Prokurator przedstawił dowody w sprawie.",
    "Trybunał rozpatrzył skargę konstytucyjną.",
    "Sąd nakazał zwrot nieruchomości.",
    "Prawnik reprezentował oskarżonego.",
    "Sędzia oddalił apelację.",
    "Prokurator wniósł akt oskarżenia.",
    "Trybunał wydał wyrok.",
    "Sąd orzekł odszkodowanie dla poszkodowanego.",
    "Prawnik sporządził testament.",
    "Sędzia zawiesił wyrok warunkowo.",
    "Prokurator przesłuchał podejrzanego.",
    "Trybunał uchylił decyzję sądu.",
    "Sąd rozpatrzył sprawę o zniesławienie.",
    "Prawnik wniósł kasację.",
    "Sędzia uniewinni oskarżonego.",
    "Prokurator umorzył postępowanie.",
    "Trybunał rozstrzygnął spór kompetencyjny.",
    "Sąd nakazał areszt tymczasowy.",
    "Prawnik negocjował ugodę.",
])

# BEZPIECZEŃSTWO
add_examples('BEZPIECZEŃSTWO', [
    "Wojsko przeprowadziło ćwiczenia taktyczne.",
    "Służby zabezpieczyły szczyt.",
    "Straż graniczna wzmocniła patrole.",
    "Agencja wykryła zagrożenie terrorystyczne.",
    "Kontrwywiad zatrzymał szpiega.",
    "Policja rozbiła gang narkotykowy.",
    "Służby przeprowadziły nalot.",
    "Wojsko uczestniczy w misji międzynarodowej.",
    "Agencja ostrzegła przed atakiem.",
    "Straż graniczna przechwyciła przemytników.",
    "Kontrwywiad monitoruje podejrzanych.",
    "Służby specjalne udaremniły zamach.",
    "Wojsko patroluje strefę przygraniczną.",
    "Agencja współpracuje z NATO.",
    "Policja zabezpieczyła wydarzenie masowe.",
    "Służby przeszkoliły funkcjonariuszy.",
    "Straż graniczna kontroluje ruch.",
    "Kontrwywiad wykrył komórkę szpiegowską.",
    "Wojsko testuje nową broń.",
    "Agencja analizuje zagrożenia.",
])

# ADMINISTRACJA
add_examples('ADMINISTRACJA', [
    "Urząd wprowadził nowe procedury.",
    "Gmina zatwierdziła plan inwestycyjny.",
    "Burmistrz podpisał umowę z wykonawcą.",
    "Rada miejska przyjęła uchwałę budżetową.",
    "Urząd wojewódzki wydał pozwolenie.",
    "Samorząd zrealizował projekt unijny.",
    "Prezydent miasta otworzył szkołę.",
    "Rada gminy przyjęła program rozwoju.",
    "Urząd wydał decyzję o odmowie.",
    "Wojewoda nadzoruje działania gminy.",
    "Samorząd modernizuje drogi.",
    "Burmistrz spotkał się z radnymi.",
    "Rada powiatu przyjęła strategię.",
    "Urząd miejski wydał zaświadczenie.",
    "Gmina otrzymała dotację.",
    "Prezydent miasta podpisał porozumienie.",
    "Rada odrzuciła projekt uchwały.",
    "Urząd wprowadził e-usługi.",
    "Wojewoda rozwiązał komisję.",
    "Gmina zmodernizowała plac miejski.",
])

# SPOŁECZEŃSTWO
add_examples('SPOŁECZEŃSTWO', [
    "Organizacja wsparła rodziny w kryzysie.",
    "Wolontariusze pomagali bezdomnym.",
    "Fundacja przekazała sprzęt dla szpitala.",
    "Społeczność lokalna zorganizowała piknik.",
    "Obywatele podpisali petycję.",
    "Organizacja prowadzi program integracyjny.",
    "Wolontariusze sprzątali plaże.",
    "Fundacja pomaga dzieciom z rodzin zastępczych.",
    "Społeczność lokalna remontuje plac zabaw.",
    "Obywatele wsparli lokalny biznes.",
    "Organizacja niesie pomoc powodzianom.",
    "Wolontariusze odwiedzili szpital dziecięcy.",
    "Fundacja finansuje kursy zawodowe.",
    "Społeczność lokalna organizuje jarmark.",
    "Obywatele wspierają seniorów.",
    "Organizacja prowadzi linię wsparcia.",
    "Wolontariusze pomagają w schronisku.",
    "Fundacja wspiera osoby z autyzmem.",
    "Społeczność lokalna tworzy kooperatywę.",
    "Obywatele tworzą straż sąsiedzką.",
])

# INFRASTRUKTURA
add_examples('INFRASTRUKTURA', [
    "Miasto buduje nową przeprawę mostową.",
    "Inwestor rozbudowuje galerie handlową.",
    "Gmina modernizuje ulice osiedlowe.",
    "Władze otwierają węzeł komunikacyjny.",
    "Firma buduje park przemysłowy.",
    "Miasto rozbudowuje kanalizację sanitarną.",
    "Inwestor konstruuje biurowiec.",
    "Gmina buduje ścieżki rowerowe.",
    "Władze modernizują dworzec autobusowy.",
    "Firma rozbudowuje port morski.",
    "Miasto remontuje place i chodniki.",
    "Inwestor buduje kompleks sportowy.",
    "Gmina rozbudowuje wodociągi.",
    "Władze otwierają centrum przesiadkowe.",
    "Firma buduje halę widowiskowo-sportową.",
    "Miasto modernizuje system odwodnienia.",
    "Inwestor buduje osiedle domków.",
    "Gmina remontuje budynek urzędu.",
    "Władze otwierają obwodnicę miasta.",
    "Firma buduje centrum logistyczne.",
])

# EKOLOGIA
add_examples('EKOLOGIA', [
    "Aktywiści protestują przeciwko kopalni.",
    "Organizacja sprząta teren parku narodowego.",
    "Ekologowie monitorują populację zwierząt.",
    "Miasto wprowadza strefy czystego transportu.",
    "Działacze ratują zagrożone gatunki roślin.",
    "Organizacja sadzi las kompensacyjny.",
    "Ekologowie badają zanieczyszczenie wody.",
    "Miasto zakazuje spalania śmieci.",
    "Działacze chronią mokradła.",
    "Organizacja ratuje ptaki z wycieku ropy.",
    "Ekologowie protestują przeciwko zabudowie.",
    "Miasto tworzy tereny zielone.",
    "Działacze edukują o ochronie przyrody.",
    "Organizacja chroni dziki puszczy.",
    "Ekologowie monitorują jakość gleby.",
    "Miasto inwestuje w panele słoneczne.",
    "Działacze protestują przeciwko wycinki.",
    "Organizacja wspiera różnorodność biologiczną.",
    "Ekologowie badają efekt zmian klimatu.",
    "Miasto tworzy rezerwat przyrody.",
])

# EDUKACJA
add_examples('EDUKACJA', [
    "Szkoła uruchomiła program stypendialny.",
    "Uniwersytet przyjął zagranicznych wykładowców.",
    "Nauczyciel prowadzi warsztaty naukowe.",
    "Szkoła zorganizowała wyjazd edukacyjny.",
    "Uniwersytet otworzył bibliotekę cyfrową.",
    "Uczeń zdobył nagrodę w konkursie.",
    "Szkoła zakupiła tablice interaktywne.",
    "Uniwersytet organizuje szkolenia dla nauczycieli.",
    "Nauczyciele wprowadzili nową metodykę.",
    "Szkoła organizuje koła zainteresowań.",
    "Uniwersytet przyjmuje studentów Erasmus.",
    "Uczeń reprezentuje szkołę na olimpiadzie.",
    "Szkoła współpracuje z lokalnymi firmami.",
    "Uniwersytet organizuje drzwi otwarte.",
    "Nauczyciel otrzymał nagrodę pedagogiczną.",
    "Szkoła wprowadza zajęcia programowania.",
    "Uniwersytet otwiera nową specjalizację.",
    "Uczeń przygotowuje projekt naukowy.",
    "Szkoła organizuje zajęcia wyrównawcze.",
    "Uniwersytet podpisuje umowy z zagranicznymi uczelniami.",
])

# BRAK_ZDARZENIA (more neutral statements)
add_examples('BRAK_ZDARZENIA', [
    "W sklepie są świeże warzywa.",
    "Park jest otwarty codziennie.",
    "Autobus kursuje według rozkładu.",
    "W bibliotece panuje cisza.",
    "Sklep ma szeroką ofertę produktów.",
    "W parku jest wiele alejek.",
    "Budynek jest dobrze oświetlony.",
    "Na parkingu jest dużo miejsc.",
    "W kinie są wygodne fotele.",
    "Restauracja serwuje lunch.",
    "W sklepie pracują miłe osoby.",
    "Park ma dużą fontannę.",
    "Na przystanku czeka kilka osób.",
    "W muzeum są cenne eksponaty.",
    "Sklep jest czysty i przestronny.",
    "W parku rosną stare dęby.",
    "Budynek ma nowoczesną architekturę.",
    "Na balkonie jest miejsce na meble.",
    "W kawiarni pachnie świeżą kawą.",
    "Sklep ma długie godziny otwarcia.",
    "W parku są ławki do siedzenia.",
    "Budynek ma windę i schody.",
    "Na ulicy rosną kasztany.",
    "W sklepie jest klimatyzacja.",
    "Park jest popularny wśród mieszkańców.",
    "Budynek znajduje się blisko centrum.",
    "Na placu jest pomnik.",
    "W sklepie można płacić kartą.",
    "Park ma plac zabaw dla dzieci.",
    "Budynek ma parking podziemny.",
])

# Create dataframe from all examples
all_examples_df = pd.DataFrame(additional_examples)

# Combine with extended dataset
print(f"\\nCombining {len(extended_df)} extended + {len(all_examples_df)} additional examples...")
final_df = pd.concat([extended_df, all_examples_df], ignore_index=True)

# Remove duplicates
before_dedup = len(final_df)
final_df = final_df.drop_duplicates(subset=['sentence'], keep='first')
after_dedup = len(final_df)
print(f"Removed {before_dedup - after_dedup} duplicates")

# Shuffle
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
final_df.to_csv('datasets/training_data_extended.csv', index=False)

print(f"\\n=== Final Dataset Created ===")
print(f"Total examples: {len(final_df)}")
print(f"\\nLabel distribution:")
print(final_df['label'].value_counts().sort_index())

print(f"\\nDataset saved to: datasets/training_data_extended.csv")
