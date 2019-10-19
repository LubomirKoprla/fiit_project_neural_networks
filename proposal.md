# Odpor�canie produktov v elektronickom obchode
TODO

## Existuj�ce rie�enia
V pr�ci [[2]](https://dl.acm.org/citation.cfm?id=3052569) autori spojili maticov� faktoriz�ciu a hlbok� neur�nov� siet. Ich model pozost�va z dvoch paraleln�ch vetiev, modelov. V jednom modely pomocou neur�novej siete vykon�vaj� maticov� faktoriz�ciu a druh� model je hlbok� neur�nov� siet. Ka�d� model sa predtr�nuje samostatne a pri predikcii spoja tieto dva modely vo v�stupnej vrstve. Spojen�m t�chto dvoch modelov a ich samostatn�m predtr�novan�m dostali pri experimentoch najlep�ie v�sledky. Vstupom do modelu je one hot encoding pou��vatela a polo�ky a v�stupom je pravdepodobnost, �e pou��vatel k�pi dan� polo�ku. Met�du overili na d�tov�ch mno�in�ch MovieLens 1M a Pinterest..

V pr�ci [[3]](https://arxiv.org/abs/1605.09477) pou��vaj� na odpor�canie pr�stup pomocou neur�novej autoregresie. Tento pr�stup je podobn� rekurentnej neur�novej sieti. Model na z�klade toho ako pou��vatel ohodnotil polo�ky vypoc�ta rank(ocak�van� hodnotenie) v�etk�ch polo�iek, ktor� pou�ije na v�ber polo�iek, ktor� sa bud� pou��vatelovi najviac p�ci. Model vyhodnocovali na d�tov�ch mno�in�ch Netflix, MovieLens 1M a 10M. Pri experimentoch zistili, �e hlb�ie neur�nov� siete dosahuj� lep�� v�sledok.

## D�tov� mno�iny
Na tr�novanie a vyhodnotenie modelu budeme pou��vat d�tov� mno�inu, ktor� pozost�va z logov aktivity pou��vatela v elektronickom obchode, teda ak� polo�ky si zobrazil, pridal do ko��ka a ktor� z nich nak�pil.

Z dostupn�ch mno��n sme vybrali dve z prostredia elektronick�ho obchodu. Prvou je Retailrocket, ktor� bola zverejnen� na Kaggle a druhou (z�lo�nou) je Yoochoose, ktor� bola zverejnen� v r�mci Recsys Challenge 2015. Jej nev�hodou v�ak je, �e neobsahuje ID pou��vatela ale iba sedenia, co neumo�nuje sk�mat dlhodob� preferencie pou��vatela.

### Retailrocket ([link](https://www.kaggle.com/retailrocket/ecommerce-dataset))
D�tov� mno�ina obsahuje s�bor s akciami, ktor� pou��vatel vykonal. Akcie sa delia na zobrazenie produktu, vlo�enie produktu do ko��ka a n�kup produktu. Tabulka obsahuje cas vykonania aktivity, ID pou��vatela, ID produktu a typ akcie. K dispoz�cii je aj s�bor, ktor� obsahuje inform�cie o produktoch - kateg�riu, dostupnost a anonymizovan� atrib�ty produktu. D�tov� mno�ina obsahuje 2,7 mili�na akcii, z toho bolo 69 tis�c pridan� do ko��ka a a 22 tis�c n�kupov.

### Yoochoose ([link](https://2015.recsyschallenge.com/challenge.html))
T�to d�tov� mno�ina obsahuje d�ta o kliknutiach a n�kupoch pou��vatelov v elektronickom obchode v r�mci jedn�ho sedenia. Je rozdelen� do dvoch s�borov, podla akcie na kliknutia a n�kupy. Tabulka s kliknutiami obsahuje ID sedenia, cas akcie, ID produktu a kateg�riu produktu. Tabulka s n�kupmi obsahuje ID sedenia, cas akcie, ID produktu, cenu produktu a mno�stvo nak�pen�ho produktu. D�tov� sada obsahuje 9 mili�nov seden�, 33 mili�nov kliknut� a 1,1 mili�na n�kupov.

## N�vrh
TODO

## Zdroje
* HE, Xiangnan, et al. Neural collaborative filtering. In: Proceedings of the 26th international conference on world wide web. International World Wide Web Conferences Steering Committee, 2017. p. 173-182.
* ZHENG, Yin, et al. A neural autoregressive approach to collaborative filtering. arXiv preprint arXiv:1605.09477, 2016.
