#!/usr/bin/env python3
"""
Script to generate an extended training dataset for event classification.
Creates 1000+ examples including BRAK_ZDARZENIA (NO_EVENT) category.
"""

import pandas as pd
import random

def generate_event_sentences():
    """Generate synthetic sentences for various event types"""
    
    # PRZESTĘPSTWO - Crime
    przestepstwo = [
        "Napastnik pobił ochroniarza przed klubem.",
        "Złodziej ukradł samochód z parkingu.",
        "Bandyta zaatakował przypadkową osobę na ulicy.",
        "Włamywacz wszedł do domu przez okno.",
        "Rabuś napadł na sklep z bronią.",
        "Policja zatrzymała podejrzanego o kradzież.",
        "Mężczyzna pobił swoją żonę w domu.",
        "Sprawca uprowadził dziecko z przedszkola.",
        "Gang okradł bank w centrum miasta.",
        "Terrorysta zdetonował bombę w budynku.",
        "Przestępca zastrzelił policjanta.",
        "Hakerzy włamali się do systemu bankowego.",
        "Oszust wyłudził pieniądze od starszej osoby.",
        "Wandal zniszczył mienie publiczne.",
        "Pirat drogowy spowodował kolizję.",
        "Złodzieje okradli mieszkanie podczas wakacji.",
        "Sprawca zaatakował nożem przechodnia.",
        "Podpalacz zniszczył budynek przemysłowy.",
        "Przestępcy dokonali napadu na jubilera.",
        "Oszuści wyłudzili kredyty na fałszywe dane.",
    ]
    
    # WYPADEK - Accident
    wypadek = [
        "Samochód uderzył w drzewo.",
        "Dwa pojazdy zderzyły się na skrzyżowaniu.",
        "Motocyklista stracił panowanie nad pojazdem.",
        "Ciężarówka przewróciła się na autostradzie.",
        "Kierowca potrącił pieszego na przejściu.",
        "Rowerzysta upadł na oblodzonej drodze.",
        "Samolot awaryjnie lądował na lotnisku.",
        "Statek uderzył w nabrzeże w porcie.",
        "Autobus zjechał z drogi do rowu.",
        "Tramwaj wypadł z szyn w centrum miasta.",
        "Samochód wpadł do rzeki z mostu.",
        "Pociąg wykoleił się na zakręcie.",
        "Kierowca stracił panowanie na śliskiej drodze.",
        "Dwa autobusy zderzyły się czołowo.",
        "Motocykl zderzył się z samochodem osobowym.",
        "Ciężarówka uderzyła w barierki na autostradzie.",
        "Samochód dostawczy zjechał z jezdni.",
        "Karambol na autostradzie zablokował ruch.",
        "Pojazd uderzył w przystanek autobusowy.",
        "Kierowca zjechał do rowu unikając zwierzęcia.",
    ]
    
    # POŻAR - Fire
    pozar = [
        "Pożar strawił budynek mieszkalny.",
        "Ogień wybuchł w fabryce chemicznej.",
        "Płomienie zniszczyły las sosnowy.",
        "Budynek biurowy stanął w płomieniach.",
        "Pożar wybuchł w centrum handlowym.",
        "Ogień objął całe piętro kamienicy.",
        "Płonący las zagrażał wiosce.",
        "Pożar zniszczył magazyn towarów.",
        "Ogień strawił stodołę z ziarnem.",
        "Płomienie objęły dach kościoła.",
        "Pożar zniszczył warsztat samochodowy.",
        "Ogień wybuchł w restauracji.",
        "Płonący śmietnik zapalił samochód.",
        "Pożar objął hale produkcyjną.",
        "Ogień zniszczył domek letniskowy.",
        "Płomienie strawiły drewniany most.",
        "Pożar wybuchł w bloku mieszkalnym.",
        "Ogień objął elektrownię.",
        "Płonący budynek zagrażał sąsiednim domom.",
        "Pożar zniszczył zabytkowy kościół.",
    ]
    
    # POLITYKA - Politics
    polityka = [
        "Premier ogłosił nowe przepisy.",
        "Prezydent podpisał ustawę budżetową.",
        "Parlament uchwalił reformę sądownictwa.",
        "Minister przedstawił plan gospodarczy.",
        "Sejm przyjął uchwałę o referendum.",
        "Rząd zaproponował nowe prawo podatkowe.",
        "Prezydent zawetował ustawę o mediach.",
        "Koalicja rządowa rozpadła się w parlamencie.",
        "Opozycja złożyła wniosek o wotum nieufności.",
        "Minister podał się do dymisji po skandalu.",
        "Parlament debatował nad zmianami w konstytucji.",
        "Rząd ogłosił nowy program społeczny.",
        "Prezydent rozpoczął wizytę w Niemczech.",
        "Sejm odrzucił projekt ustawy opozycji.",
        "Minister zapowiedział reformę edukacji.",
        "Parlament wybrał nowego marszałka.",
        "Rząd przedłużył stan wyjątkowy.",
        "Prezydent ułaskawił skazanych więźniów.",
        "Koalicja uzgodniła kompromis w sprawie budżetu.",
        "Minister spotkał się z przedstawicielami związków.",
    ]
    
    # SPORT - Sports
    sport = [
        "Piłkarz strzelił zwycięskiego gola.",
        "Drużyna wygrała mecz finałowy.",
        "Reprezentacja pokonała rywali 3-0.",
        "Siatkarz zdobył mistrzostwo świata.",
        "Lekkoatleta pobił rekord kraju.",
        "Tenisista wygrał turniej wielkoszlemowy.",
        "Koszykarz zdobył 30 punktów w meczu.",
        "Bokser nokautował przeciwnika.",
        "Pływak zdobył złoty medal olimpijski.",
        "Kolarz wygrał wyścig Tour de France.",
        "Skoczek narciarski ustanowił nowy rekord.",
        "Hokeista zdobył hat-tricka w meczu.",
        "Piłkarka została najlepszą strzelczynią ligi.",
        "Zawodnik zdobył medale na mistrzostwach.",
        "Drużyna awansowała do finału mistrzostw.",
        "Tenisistka wygrała turniej WTA.",
        "Siatkarz został wybrany MVP rozgrywek.",
        "Reprezentacja zakwalifikowała się do mundialu.",
        "Kolarz wygrał etap górski.",
        "Bokser obronił tytuł mistrza świata.",
    ]
    
    # EKONOMIA - Economy
    ekonomia = [
        "Bank centralny podniósł stopy procentowe.",
        "Firma ogłosiła zwolnienia grupowe pracowników.",
        "Giełda odnotowała spadki indeksów.",
        "Rząd zwiększył budżet na inwestycje.",
        "Przedsiębiorstwo otworzyło nową fabrykę.",
        "Akcje spółki wzrosły o 20 procent.",
        "Bank udzielił kredytu na inwestycje.",
        "Koncern zapowiedział fuzję z konkurentem.",
        "Inflacja wzrosła do najwyższego poziomu.",
        "Bezrobocie spadło do minimum.",
        "Spółka ogłosiła upadłość po stratach.",
        "Inwestor kupił akcje za miliony złotych.",
        "Firma zatrudniła 500 nowych pracowników.",
        "Eksport wzrósł o 15 procent.",
        "Bank wprowadził nowe opłaty za usługi.",
        "Producent obniżył ceny produktów.",
        "Giełda osiągnęła rekordowe obroty.",
        "Rząd zapowiedział obniżkę podatków.",
        "Przedsiębiorstwo zainwestowało w nową technologię.",
        "Kurs euro osiągnął najwyższy poziom.",
    ]
    
    # NAUKA - Science
    nauka = [
        "Naukowcy odkryli nowy gatunek zwierzęcia.",
        "Badacze opracowali szczepionkę na wirusa.",
        "Uczeni znaleźli dowody na istnienie planety.",
        "Laboratorium przeprowadziło przełomowe badania.",
        "Eksperci stworzyli innowacyjną technologię.",
        "Zespół naukowy opublikował wyniki badań.",
        "Badacze odkryli metodę leczenia choroby.",
        "Naukowcy zidentyfikowali nowe cząstki elementarne.",
        "Instytut badawczy otrzymał grant na projekt.",
        "Uczeni udowodnili teorię fizyczną.",
        "Badacze zmapowali genom organizmu.",
        "Naukowcy skonstruowali nowy mikroskop.",
        "Eksperci odkryli starożytne artefakty.",
        "Laboratorium przetestowało nowy lek.",
        "Badacze opracowali model klimatyczny.",
        "Naukowcy obserwowali nowe zjawisko astronomiczne.",
        "Zespół naukowy zdobył Nagrodę Nobla.",
        "Badacze odkryli nowe właściwości materiału.",
        "Uczeni stworzyli sztuczną inteligencję.",
        "Instytut przeprowadził eksperyment na stacji kosmicznej.",
    ]
    
    # KLĘSKA_ŻYWIOŁOWA - Natural Disaster
    kleska_zywiolowa = [
        "Powódź zniszczyła setki domów.",
        "Trzęsienie ziemi nawiedziło region.",
        "Tornado przeszło przez miasto.",
        "Lawina błotna zasypała wioskę.",
        "Huragan zniszczył wybrzeże.",
        "Susza dotknęła obszary rolnicze.",
        "Lawina śnieżna zablokowała przełęcz.",
        "Tsunami uderzyło w brzeg oceanu.",
        "Erupcja wulkanu zagrażała okolicznym miastom.",
        "Burza piaskowa ogarnęła pustynię.",
        "Fala upałów przetoczyła się przez kraj.",
        "Grad zniszczył uprawy rolników.",
        "Ziemia osunęła się zagrażając zabudowaniom.",
        "Cyklon tropikalny uderzył w wyspę.",
        "Ulewa spowodowała podtopienia.",
        "Mróz zniszczył plantacje.",
        "Silny wiatr powalił drzewa.",
        "Gradobicie zniszczyło dachy budynków.",
        "Oblodzenie sparaliżowało ruch drogowy.",
        "Burza zniszczyła linie energetyczne.",
    ]
    
    # MEDYCYNA - Medicine
    medycyna = [
        "Lekarz przeprowadził skomplikowaną operację.",
        "Szpital przyjął pacjentów po wypadku.",
        "Pacjent wyzdrowiał po długim leczeniu.",
        "Chirurg przeszczepił nerkę choremu.",
        "Ratownicy medyczni udzielili pierwszej pomocy.",
        "Szpital wprowadził nową metodę leczenia.",
        "Lekarka rozpoznała rzadką chorobę.",
        "Zespół medyczny uratował życie pacjenta.",
        "Szpital otrzymał nowoczesny sprzęt.",
        "Pacjent przeszedł rehabilitację po urazie.",
        "Lekarz odkrył przyczynę schorzenia.",
        "Klinika otworzyła nowy oddział kardiologiczny.",
        "Chirurg wykonał przełomową operację.",
        "Szpital uruchomił program przeszczepów.",
        "Lekarka pomogła urodzić dziecko.",
        "Zespół medyczny walczył o życie poszkodowanego.",
        "Pacjent otrzymał eksperymentalną terapię.",
        "Szpital przeprowadził badania przesiewowe.",
        "Lekarz wykrył chorobę we wczesnym stadium.",
        "Klinika wdrożyła nowy protokół leczenia.",
    ]
    
    # PROTESTY - Protests
    protesty = [
        "Manifestanci zablokowali główną ulicę.",
        "Demonstracja przeszła przez centrum miasta.",
        "Związkowcy zorganizowali strajk generalny.",
        "Protest przeciwko reformie zebrał tysiące osób.",
        "Obywatele demonstrowali przed parlamentem.",
        "Strajk sparaliżował komunikację miejską.",
        "Rolnicy zablokowali drogi krajowe.",
        "Protest studencki objął uniwersytet.",
        "Manifestacja przeciw podwyżkom cen.",
        "Górnicy rozpoczęli strajk w kopalni.",
        "Protest klimatyczny zgromadził aktywistów.",
        "Demonstranci żądali dymisji rządu.",
        "Strajk nauczycieli zamknął szkoły.",
        "Protest przeciw cenzurze mediów.",
        "Manifestacja wsparcia dla imigrantów.",
        "Strajk pracowników ochrony zdrowia.",
        "Protest rolników przed ministerstwem.",
        "Demonstracja przeciwko przemocy.",
        "Strajk kierowców sparaliżował transport.",
        "Protest studentów przeciw opłatom.",
    ]
    
    # KULTURA - Culture
    kultura = [
        "Artysta otworzył wystawę w galerii.",
        "Film zdobył nagrodę na festiwalu.",
        "Muzyk wydał nowy album studyjny.",
        "Teatr wystawił premierę spektaklu.",
        "Pisarz otrzymał literacką nagrodę.",
        "Opera zorganizowała koncert galowy.",
        "Malarz zaprezentował swoje dzieła.",
        "Festiwal muzyczny zgromadził fanów.",
        "Muzeum otworzyło nową ekspozycję.",
        "Aktor zagrał główną rolę w filmie.",
        "Koncert klasyczny odbył się w filharmonii.",
        "Wystawa fotografii przyciągnęła tłumy.",
        "Spektakl teatralny odniósł sukces.",
        "Artysta otrzymał nagrodę za całokształt twórczości.",
        "Festiwal filmowy zaprezentował nowe produkcje.",
        "Muzyk wystąpił w prestiżowej sali koncertowej.",
        "Galeria kupiła obraz za miliony.",
        "Premiera książki przyciągnęła czytelników.",
        "Teatr zorganizował warsztaty dla dzieci.",
        "Festiwal jazzowy zgromadził muzyków.",
    ]
    
    # TECHNOLOGIA - Technology
    technologia = [
        "Firma zaprezentowała nowy smartfon.",
        "Programiści opracowali innowacyjną aplikację.",
        "Koncern wprowadził na rynek nowy procesor.",
        "Startup otrzymał finansowanie na rozwój.",
        "Naukowcy stworzyli kwantowy komputer.",
        "Firma technologiczna ogłosiła nowy produkt.",
        "Programiści wydali aktualizację systemu.",
        "Koncern zainwestował w sztuczną inteligencję.",
        "Startup opracował aplikację do nauki.",
        "Firma wprowadziła chmurę obliczeniową.",
        "Programiści naprawili krytyczne błędy.",
        "Koncern kupił startup za miliardy.",
        "Firma uruchomiła sieć 5G w mieście.",
        "Naukowcy testują kwantowe szyfrowanie.",
        "Producent zaprezentował elektryczny samochód.",
        "Programiści stworzyli platform do e-commerce.",
        "Firma technologiczna otworzyła centrum R&D.",
        "Startup opracował system rozpoznawania twarzy.",
        "Koncern wprowadził usługę streamingu.",
        "Firma zainwestowała w rozwój AI.",
    ]
    
    # PRAWO - Law
    prawo = [
        "Sąd wydał wyrok w głośnej sprawie.",
        "Prokurator postawił zarzuty oskarżonemu.",
        "Trybunał orzekł o niezgodności ustawy.",
        "Sędzia skazał przestępcę na karę więzienia.",
        "Prawnik obronił klienta w sądzie.",
        "Sąd uniewinnił oskarżonego z zarzutów.",
        "Prokurator wszczął śledztwo w sprawie.",
        "Trybunał uchylił wcześniejszy wyrok.",
        "Sąd nakazał wypłatę odszkodowania.",
        "Prawnik złożył apelację od wyroku.",
        "Sędzia oddalił pozew powoda.",
        "Sąd rozpatrzył sprawę o alimenty.",
        "Prokurator przesłuchał świadków.",
        "Trybunał wydał nakaz aresztowania.",
        "Sąd orzekł rozwód małżonków.",
        "Prawnik sporządził umowę dla klienta.",
        "Sędzia zawiesił wykonanie kary.",
        "Sąd nakazał eksmisję lokatorów.",
        "Prokurator zażądał kary maksymalnej.",
        "Trybunał rozstrzygnął spór graniczny.",
    ]
    
    # BEZPIECZEŃSTWO - Security
    bezpieczenstwo = [
        "Straż graniczna zatrzymała przemytników.",
        "Służby specjalne przeprowadziły operację.",
        "Agencja bezpieczeństwa ostrzegła przed zagrożeniem.",
        "Wojsko rozpoczęło manewry wojskowe.",
        "Kontrwywiad wykrył szpiegów.",
        "Policja rozbiła gang przestępczy.",
        "Służby specjalne zatrzymały terrorystę.",
        "Agencja zabezpieczyła szczyt dyplomatyczny.",
        "Wojsko patroluje granicę kraju.",
        "Kontrwywiad śledzi podejrzanych.",
        "Straż graniczna wzmocniła kontrole.",
        "Służby przeprowadziły akcję antyterrorystyczną.",
        "Agencja ostrzegła przed cyberatakiem.",
        "Wojsko uczestniczy w misji pokojowej.",
        "Kontrwywiad udaremnił spisek.",
        "Policja zabezpieczyła imprezę masową.",
        "Służby specjalne przeszkoliły funkcjonariuszy.",
        "Agencja współpracuje z zagranicą.",
        "Wojsko testuje nowy sprzęt.",
        "Kontrwywiad monitoruje zagrożenia.",
    ]
    
    # ADMINISTRACJA - Administration
    administracja = [
        "Urząd wydał decyzję administracyjną.",
        "Gmina przyjęła budżet na nowy rok.",
        "Burmistrz otworzył nową inwestycję.",
        "Rada miejska uchwaliła plan zagospodarowania.",
        "Urząd wojewódzki zatwierdził projekt.",
        "Samorząd zainwestował w infrastrukturę.",
        "Prezydent miasta podpisał umowę.",
        "Rada gminy przyjęła nowe przepisy.",
        "Urząd skarbowy przeprowadził kontrolę.",
        "Wojewoda wydał zarządzenie.",
        "Gmina rozbudowała sieć wodociągową.",
        "Burmistrz spotkał się z mieszkańcami.",
        "Rada powiatu przyjęła strategię rozwoju.",
        "Urząd miejski wydał pozwolenie na budowę.",
        "Samorząd dofinansował projekty lokalne.",
        "Prezydent miasta ogłosił nowy program.",
        "Rada gminy odrzuciła uchwałę.",
        "Urząd wprowadził elektroniczne usługi.",
        "Wojewoda rozwiązał radę gminy.",
        "Gmina zmodernizowała oświetlenie ulic.",
    ]
    
    # SPOŁECZEŃSTWO - Society
    spoleczenstwo = [
        "Organizacja charytatywna pomogła potrzebującym.",
        "Wolontariusze zorganizowali zbiórkę darów.",
        "Fundacja przekazała dary dla dzieci.",
        "Społeczność lokalna wsparła sąsiadów.",
        "Obywatele zainicjowali petycję.",
        "Organizacja pozarządowa rozpoczęła kampanię.",
        "Wolontariusze sprzątali park miejski.",
        "Fundacja otworzyła schronisko dla bezdomnych.",
        "Społeczność lokalna zorganizowała festyn.",
        "Obywatele protestowali przeciw niesprawiedliwości.",
        "Organizacja wspiera ofiary przemocy.",
        "Wolontariusze odwiedzili dom opieki.",
        "Fundacja finansuje stypendia dla uczniów.",
        "Społeczność lokalna remontuje świetlicę.",
        "Obywatele pomagają uchodźcom.",
        "Organizacja prowadzi program edukacyjny.",
        "Wolontariusze sadzą drzewa w lesie.",
        "Fundacja wspiera osoby niepełnosprawne.",
        "Społeczność lokalna organizuje dożynki.",
        "Obywatele tworzą inicjatywę sąsiedzką.",
    ]
    
    # INFRASTRUKTURA - Infrastructure
    infrastruktura = [
        "Rząd rozpoczął budowę nowej autostrady.",
        "Miasto remontuje most nad rzeką.",
        "Inwestor buduje centrum handlowe.",
        "Gmina modernizuje oczyszczalnię ścieków.",
        "Władze otwierają nową linię metra.",
        "Firma buduje elektrownię wiatrową.",
        "Miasto rozbudowuje sieć tramwajową.",
        "Inwestor konstruuje osiedle mieszkaniowe.",
        "Gmina buduje nowy chodnik.",
        "Władze modernizują dworzec kolejowy.",
        "Firma rozbudowuje terminal lotniska.",
        "Miasto remontuje ulice w centrum.",
        "Inwestor buduje hotel na wybrzeżu.",
        "Gmina rozbudowuje kanalizację.",
        "Władze otwierają nowy szpital.",
        "Firma buduje fabrykę samochodów.",
        "Miasto modernizuje bibliotekę publiczną.",
        "Inwestor buduje park technologiczny.",
        "Gmina remontuje szkołę podstawową.",
        "Władze otwierają stadion sportowy.",
    ]
    
    # EKOLOGIA - Ecology
    ekologia = [
        "Aktywiści protestują przeciwko wycince lasu.",
        "Organizacja sadzi drzewa w parku.",
        "Ekologowie ostrzegają przed zanieczyszczeniem.",
        "Miasto wprowadza segregację śmieci.",
        "Działacze chronią zagrożone gatunki.",
        "Organizacja sprząta brzeg jeziora.",
        "Ekologowie monitorują jakość powietrza.",
        "Miasto zakazuje plastikowych toreb.",
        "Działacze walczą o czystość rzek.",
        "Organizacja ratuje zwierzęta w lasach.",
        "Ekologowie protestują przeciw zanieczyszczeniom.",
        "Miasto tworzy nowy park narodowy.",
        "Działacze edukują o recyklingu.",
        "Organizacja chroni dziką przyrodę.",
        "Ekologowie badają skutki zmian klimatu.",
        "Miasto inwestuje w energię odnawialną.",
        "Działacze sprzeciwiają się budowie zapory.",
        "Organizacja wspiera bioróżnorodność.",
        "Ekologowie monitorują populację ptaków.",
        "Miasto tworzy korytarze ekologiczne.",
    ]
    
    # EDUKACJA - Education
    edukacja = [
        "Szkoła rozpoczęła nowy rok szkolny.",
        "Uniwersytet przyjął rekordową liczbę studentów.",
        "Nauczyciel otrzymał nagrodę za osiągnięcia.",
        "Szkoła zorganizowała konkurs dla uczniów.",
        "Uniwersytet otworzył nowy kierunek studiów.",
        "Uczen wygrał olimpiadę przedmiotową.",
        "Szkoła zmodernizowała sale lekcyjne.",
        "Uniwersytet podpisał umowę z zagranicą.",
        "Nauczyciele zorganizowali wycieczkę edukacyjną.",
        "Szkoła wprowadza nowy program nauczania.",
        "Uniwersytet organizuje konferencję naukową.",
        "Uczeń zdobył stypendium naukowe.",
        "Szkoła zakupiła nowoczesny sprzęt.",
        "Uniwersytet otwiera centrum badawcze.",
        "Nauczyciel prowadzi innowacyjne zajęcia.",
        "Szkoła organizuje zajęcia pozalekcyjne.",
        "Uniwersytet przyjmuje zagranicznych studentów.",
        "Uczeń reprezentował szkołę w konkursie.",
        "Szkoła współpracuje z przedsiębiorstwami.",
        "Uniwersytet organizuje wykłady otwarte.",
    ]
    
    # BRAK_ZDARZENIA - No Event (neutral/descriptive statements)
    brak_zdarzenia = [
        "Dzisiaj jest piękna pogoda.",
        "Niebo jest zachmurzone nad miastem.",
        "Temperatura wynosi dwadzieścia stopni.",
        "W parku rosną duże drzewa.",
        "Rzeka płynie przez centrum miasta.",
        "Budynek ma dziesięć pięter.",
        "Sklep jest otwarty od rana.",
        "Autobus jeździ co pół godziny.",
        "W bibliotece jest dużo książek.",
        "Park znajduje się przy ulicy Głównej.",
        "Most łączy dwa brzegi rzeki.",
        "Kawiarnia serwuje kawę i herbatę.",
        "Szkoła mieści się w starym budynku.",
        "Droga prowadzi do lasu.",
        "W ogrodzie rosną kwiaty.",
        "Samochód stoi na parkingu.",
        "W oknie wisi biała firanka.",
        "Pies śpi na podwórku.",
        "Kot siedzi na parapecie.",
        "Dzieci bawią się na placu zabaw.",
        "Sklep spożywczy jest przy rynku.",
        "W pokoju stoi duża szafa.",
        "Lampa świeci jasnym światłem.",
        "Zegar pokazuje piątą godzinę.",
        "Kwiat ma czerwone płatki.",
        "Stół jest wykonany z drewna.",
        "W lodówce jest mleko i jajka.",
        "Fotel stoi przy oknie.",
        "Na ścianie wisi obraz.",
        "Ławka znajduje się pod drzewem.",
        "Fontanna jest w centrum parku.",
        "W mieście jest wiele sklepów.",
        "Mieszkanie ma trzy pokoje.",
        "W kuchni stoi kuchenka gazowa.",
        "Rower ma dwa koła.",
        "Na biurku leży długopis.",
        "W szklanej wazie stoją kwiaty.",
        "Kapelusz wisi na wieszaku.",
        "Okulary leżą na stoliku.",
        "W szufladzie są dokumenty.",
        "Dywan pokrywa podłogę.",
        "Na balkonie rosną rośliny.",
        "W akwarium pływają ryby.",
        "Książka leży na półce.",
        "Na drzewie siedzi ptak.",
        "W kuchni pachnie kawa.",
        "Telefon leży na ładowarce.",
        "W ogrodzie jest altanka.",
        "Ulica jest wyłożona kostką brukową.",
        "W porcie stoją statki.",
        "Las rozciąga się na horyzoncie.",
        "Pole jest pokryte śniegiem.",
        "Wiatrak obraca się na wietrze.",
        "W sklepie jest muzyka w tle.",
        "Tablica informacyjna wisi na ścianie.",
        "W pokoju jest przyjemna temperatura.",
        "Krzesło ma miękkie oparcie.",
        "Na stole leży gazeta.",
        "W wazonie są świeże kwiaty.",
        "Morze jest spokojne dziś.",
        # Additional BRAK_ZDARZENIA examples
        "Ściana jest pomalowana na biało.",
        "W skrzynce pocztowej są listy.",
        "Okno wychodzi na północ.",
        "Drzwi są wykonane z dębu.",
        "Podłoga jest wyłożona parkietem.",
        "W szafie wiszą ubrania.",
        "Lustro odbija światło słońca.",
        "Firanki są koloru kremowego.",
        "W garażu stoi narzędzia.",
        "Ogród ma powierzchnię stu metrów.",
        "Żyrandol wisi nad stołem.",
        "W piwnicy przechowywane są rzeczy.",
        "Na tarasie stoją doniczki.",
        "Schodek prowadzi do piwnicy.",
        "W garażu jest miejsce na samochód.",
        "Dach pokryty jest dachówką.",
        "Płot oddziela posesje.",
        "W altanie stoją ławki.",
        "Kominek znajduje się w salonie.",
        "Basen jest w ogrodzie.",
        "Grządka jest pełna warzyw.",
        "Na podwórku rośnie jabłoń.",
        "W szopie trzymane są narzędzia.",
        "Ścieżka prowadzi do furtki.",
        "Brama jest otwarta.",
        "W skrzynce rosną pelargonie.",
        "Żaluzje są opuszczone.",
        "W komórce leży węgiel.",
        "Antena jest na dachu.",
        "W budzie mieszka pies.",
    ]
    
    # Combine all categories
    all_data = []
    
    categories = {
        'PRZESTĘPSTWO': przestepstwo,
        'WYPADEK': wypadek,
        'POŻAR': pozar,
        'POLITYKA': polityka,
        'SPORT': sport,
        'EKONOMIA': ekonomia,
        'NAUKA': nauka,
        'KLĘSKA_ŻYWIOŁOWA': kleska_zywiolowa,
        'MEDYCYNA': medycyna,
        'PROTESTY': protesty,
        'KULTURA': kultura,
        'TECHNOLOGIA': technologia,
        'PRAWO': prawo,
        'BEZPIECZEŃSTWO': bezpieczenstwo,
        'ADMINISTRACJA': administracja,
        'SPOŁECZEŃSTWO': spoleczenstwo,
        'INFRASTRUKTURA': infrastruktura,
        'EKOLOGIA': ekologia,
        'EDUKACJA': edukacja,
        'BRAK_ZDARZENIA': brak_zdarzenia,
    }
    
    for label, sentences in categories.items():
        for sentence in sentences:
            all_data.append({'sentence': sentence, 'label': label})
    
    return all_data


def generate_variations(sentence, num_variations=2):
    """Generate simple variations of a sentence"""
    variations = [sentence]
    
    # Simple transformations (these maintain the same meaning but vary the text)
    replacements = {
        'Napastnik': ['Agresor', 'Sprawca', 'Atakujący'],
        'Złodziej': ['Rabuś', 'Przestępca', 'Sprawca kradzieży'],
        'Policja': ['Funkcjonariusze', 'Policjanci', 'Służby porządkowe'],
        'Samochód': ['Pojazd', 'Auto', 'Automobil'],
        'Minister': ['Przedstawiciel rządu', 'Członek gabinetu', 'Polityk'],
        'Naukowcy': ['Badacze', 'Uczeni', 'Zespół naukowy'],
        'bardzo': ['niezwykle', 'wyjątkowo', 'nader'],
        'duży': ['wielki', 'ogromny', 'spory'],
        'nowy': ['świeży', 'nowoczesny', 'aktualny'],
    }
    
    # Try to create variations (but don't force it if it doesn't make sense)
    for _ in range(min(num_variations - 1, 1)):
        varied = sentence
        for old, new_options in replacements.items():
            if old in varied:
                varied = varied.replace(old, random.choice(new_options), 1)
                if varied != sentence and varied not in variations:
                    variations.append(varied)
                    break
    
    return variations[:num_variations]


def extend_existing_dataset(existing_path, output_path):
    """Read existing dataset and extend it with new examples"""
    
    # Read existing data
    print(f"Reading existing dataset from {existing_path}...")
    existing_df = pd.read_csv(existing_path)
    print(f"Existing dataset has {len(existing_df)} examples")
    
    # Generate new data
    print("Generating new examples...")
    new_data = generate_event_sentences()
    new_df = pd.DataFrame(new_data)
    
    # Generate variations to reach 1000+ examples
    print("Generating variations to reach target size...")
    all_data = new_data.copy()
    
    # Calculate how many more examples we need
    current_total = len(existing_df) + len(new_data)
    target = 1100  # Target a bit more than 1000
    
    if current_total < target:
        needed = target - current_total
        print(f"Need {needed} more examples to reach {target}")
        
        # Generate variations from existing examples
        variation_count = 0
        for item in new_data[:needed]:  # Take only what we need
            variations = generate_variations(item['sentence'], num_variations=2)
            for var in variations[1:]:  # Skip the original
                if variation_count < needed:
                    all_data.append({'sentence': var, 'label': item['label']})
                    variation_count += 1
    
    new_df = pd.DataFrame(all_data)
    
    # Combine datasets
    print("Combining datasets...")
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['sentence'], keep='first')
    after_dedup = len(combined_df)
    print(f"Removed {before_dedup - after_dedup} duplicate sentences")
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to file
    combined_df.to_csv(output_path, index=False)
    print(f"\nExtended dataset saved to {output_path}")
    print(f"Total examples: {len(combined_df)}")
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    label_counts = combined_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    
    return combined_df


if __name__ == "__main__":
    # Paths
    existing_path = "datasets/training_data.csv"
    output_path = "datasets/training_data_extended.csv"
    
    # Generate extended dataset
    df = extend_existing_dataset(existing_path, output_path)
    
    print("\n=== Dataset Generation Complete ===")
    print(f"The extended dataset contains {len(df)} examples")
    print(f"It includes the new BRAK_ZDARZENIA category with {len(df[df['label'] == 'BRAK_ZDARZENIA'])} examples")
