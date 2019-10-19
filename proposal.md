# Odporúcanie produktov v elektronickom obchode
TODO

## Existujúce riešenia
V práci [[2]](https://dl.acm.org/citation.cfm?id=3052569) autori spojili maticovú faktorizáciu a hlbokú neurónovú siet. Ich model pozostáva z dvoch paralelných vetiev, modelov. V jednom modely pomocou neurónovej siete vykonávajú maticovú faktorizáciu a druhý model je hlboká neurónová siet. Každý model sa predtrénuje samostatne a pri predikcii spoja tieto dva modely vo výstupnej vrstve. Spojením týchto dvoch modelov a ich samostatným predtrénovaním dostali pri experimentoch najlepšie výsledky. Vstupom do modelu je one hot encoding používatela a položky a výstupom je pravdepodobnost, že používatel kúpi danú položku. Metódu overili na dátových množinách MovieLens 1M a Pinterest..

V práci [[3]](https://arxiv.org/abs/1605.09477) používajú na odporúcanie prístup pomocou neurónovej autoregresie. Tento prístup je podobný rekurentnej neurónovej sieti. Model na základe toho ako používatel ohodnotil položky vypocíta rank(ocakávané hodnotenie) všetkých položiek, ktorý použije na výber položiek, ktoré sa budú používatelovi najviac páci. Model vyhodnocovali na dátových množinách Netflix, MovieLens 1M a 10M. Pri experimentoch zistili, že hlbšie neurónové siete dosahujú lepší výsledok.

## Dátové množiny
Na trénovanie a vyhodnotenie modelu budeme používat dátovú množinu, ktorá pozostáva z logov aktivity používatela v elektronickom obchode, teda aké položky si zobrazil, pridal do košíka a ktoré z nich nakúpil.

Z dostupných množín sme vybrali dve z prostredia elektronického obchodu. Prvou je Retailrocket, ktorá bola zverejnená na Kaggle a druhou (záložnou) je Yoochoose, ktorá bola zverejnená v rámci Recsys Challenge 2015. Jej nevýhodou však je, že neobsahuje ID používatela ale iba sedenia, co neumožnuje skúmat dlhodobé preferencie používatela.

### Retailrocket ([link](https://www.kaggle.com/retailrocket/ecommerce-dataset))
Dátová množina obsahuje súbor s akciami, ktoré používatel vykonal. Akcie sa delia na zobrazenie produktu, vloženie produktu do košíka a nákup produktu. Tabulka obsahuje cas vykonania aktivity, ID používatela, ID produktu a typ akcie. K dispozícii je aj súbor, ktorý obsahuje informácie o produktoch - kategóriu, dostupnost a anonymizované atribúty produktu. Dátová množina obsahuje 2,7 milióna akcii, z toho bolo 69 tisíc pridaní do košíka a a 22 tisíc nákupov.

### Yoochoose ([link](https://2015.recsyschallenge.com/challenge.html))
Táto dátová množina obsahuje dáta o kliknutiach a nákupoch používatelov v elektronickom obchode v rámci jedného sedenia. Je rozdelená do dvoch súborov, podla akcie na kliknutia a nákupy. Tabulka s kliknutiami obsahuje ID sedenia, cas akcie, ID produktu a kategóriu produktu. Tabulka s nákupmi obsahuje ID sedenia, cas akcie, ID produktu, cenu produktu a množstvo nakúpeného produktu. Dátová sada obsahuje 9 miliónov sedení, 33 miliónov kliknutí a 1,1 milióna nákupov.

## Návrh
TODO

## Zdroje
* HE, Xiangnan, et al. Neural collaborative filtering. In: Proceedings of the 26th international conference on world wide web. International World Wide Web Conferences Steering Committee, 2017. p. 173-182.
* ZHENG, Yin, et al. A neural autoregressive approach to collaborative filtering. arXiv preprint arXiv:1605.09477, 2016.
