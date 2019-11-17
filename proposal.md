# Odpor��anie produktov v elektronickom obchode

Elektronick� obchod sa stal neoddelite�nou s��as�ou �ivota takmer ka�d�ho z n�s. V takomto obchode si mo�no dnes be�ne k�pi� tis�ce r�znych produktov. Ned� sa teda o�ak�va�, �e pou��vate� bude prech�dza� v�etky produkty a� k�m nen�jde ten spr�vny. Je tie� dok�zan�, �e ke� pou��vate� do ist�ho okamihu na webovom s�dle neobjav� relevantn� inform�cie, od�de. V pr�pade elektronick�ho obchodu mo�no de facto hovori� o strate z�kazn�ka, ktor� si produkt k�pi u konkurencie.

Pre predajcov je preto k���ov� z�kazn�kovi v elektronickom obchode zjednodu�i� prechod k produktom, o ktor� by potenci�lne mohol ma� z�ujem. Ide teda o personalizovan� odpor��anie produktov z�kazn�kom. Toto sa zvykne rie�i� r�znymi odpor��ac�mi algoritmami. Medzi najroz��renej�ie pr�stupy k odpor��aniu patr� obsahov� a kolaborat�vne filtrovanie. V posledn�ch rokoch sa v�ak uk�zalo, �e tento probl�m mo�no rie�i� aj s vyu�it�m hlbok�ch neur�nov�ch siet�, ktor� v niektor�ch pr�padoch dosahuj� signifikantne lep�ie v�sledky ako tradi�n� pr�stupy k odpor��aniu.


## Existuj�ce rie�enia
Autori v pr�ci [[1]](https://arxiv.org/abs/1608.07400) uk�zali, �e pomocou rekurentnej neur�novej siete je mo�n� dosiahnu� lep�ie v�sledky ako state-of-the-art pr�stupy pri kr�tkodobom odpor��an�. Pou�it�m LSTM dok�zali na z�klade sekvencie interakci� pou��vate�a s polo�kami predikova� jeho nasleduj�ce interakcie. Met�du overili na d�tov�ch mno�in�ch MovieLens 1M a Netflix. Okrem v�razn�ho zlep�enia v kr�tkodob�ch predikci�ch sa v pr�pade dom�ny MovieLens podarilo dosiahnu� aj zlep�enie v �plnosti (angl. recall), ktor� mo�no interpretova� ako dlhodob� predikciu. Do�lo tie� k v�razn�mu zv��eniu diverzity spr�vne odporu�en�ch polo�iek (v oboch dom�nach).


V pr�ci [[2]](https://dl.acm.org/citation.cfm?id=3052569) autori spojili maticov� faktoriz�ciu a hlbok� neur�nov� sie�. Ich model pozost�va z dvoch paraleln�ch vetiev, modelov. V jednom modely pomocou neur�novej siete vykon�vaj� maticov� faktoriz�ciu a druh� model je hlbok� neur�nov� sie�. Ka�d� model sa predtr�nuje samostatne a pri predikcii spoja tieto dva modely vo v�stupnej vrstve. Spojen�m t�chto dvoch modelov a ich samostatn�m predtr�novan�m dostali pri experimentoch najlep�ie v�sledky. Vstupom do modelu je one hot encoding pou��vate�a a polo�ky a v�stupom je pravdepodobnos�, �e pou��vate� k�pi dan� polo�ku. Met�du overili na d�tov�ch mno�in�ch MovieLens 1M a Pinterest..

 
V pr�ci [[3]](https://arxiv.org/abs/1605.09477) pou��vaj� na odpor��anie pr�stup pomocou neur�novej autoregresie. Tento pr�stup je podobn� rekurentnej neur�novej sieti. Model na z�klade toho ako pou��vate� ohodnotil polo�ky vypo��ta rank(o�ak�van� hodnotenie) v�etk�ch polo�iek, ktor� pou�ije na v�ber polo�iek, ktor� sa bud� pou��vate�ovi najviac p��i. Model vyhodnocovali na d�tov�ch mno�in�ch Netflix, MovieLens 1M a 10M. Pri experimentoch zistili, �e hlb�ie neur�nov� siete dosahuj� lep�� v�sledok.

Kombin�ciou dlhodob�ch a kr�tkodob�ch preferenci� pou��vate�a z�skan�ch dvomi LSTM neur�nov�mi sie�ami sa autorom v pr�ci [[4]](https://dl.acm.org/citation.cfm?id=3220014) podarilo prekona� met�du HRNN Init, ktor� bola pova�ovan� za state-of-the-art v �lohe odpor��ania nasleduj�cej polo�ky. Navrhnut� met�da BINN pozost�va z dvoch hlavn�ch komponentov: vnorenie polo�iek (angl. item embedding) a u�enie spr�vania pou��vate�a. Prv� �as� predstavuje item2vec obohaten� o zoh�adnenie frekvencie polo�ky ako v�hov�ho faktora. Druh� �as� tvoria dve LSTM siete ur�en� na zachytenie historicky stabiln�ch dlhodob�ch preferenci� pou��vate�a a kr�tkodob�ho z�meru pou��vate�a v r�mci s��asn�ho sedenia. Met�da bola overen� na dvoch e-commerce d�tov�ch mno�in�ch JD a Tianchi.

## D�tov� mno�iny
Na tr�novanie a vyhodnotenie modelu budeme pou��va� d�tov� mno�inu, ktor� pozost�va z logov aktivity pou��vate�a v elektronickom obchode, teda ak� polo�ky si zobrazil, pridal do ko��ka a ktor� z nich nak�pil.

Z dostupn�ch mno��n sme vybrali dve z prostredia elektronick�ho obchodu. Prvou je Retailrocket, ktor� bola zverejnen� na Kaggle a druhou (z�lo�nou) je Yoochoose, ktor� bola zverejnen� v r�mci Recsys Challenge 2015. Jej nev�hodou v�ak je, �e neobsahuje ID pou��vate�a ale iba sedenia, �o neumo��uje sk�ma� dlhodob� preferencie pou��vate�a.

### Retailrocket ([link](https://www.kaggle.com/retailrocket/ecommerce-dataset))
D�tov� mno�ina obsahuje s�bor s akciami, ktor� pou��vate� vykonal. Akcie sa delia na zobrazenie produktu, vlo�enie produktu do ko��ka a n�kup produktu. Tabu�ka obsahuje �as vykonania aktivity, ID pou��vate�a, ID produktu a typ akcie. K dispoz�cii je aj s�bor, ktor� obsahuje inform�cie o produktoch - kateg�riu, dostupnos� a anonymizovan� atrib�ty produktu. D�tov� mno�ina obsahuje 2,7 mili�na akcii, z toho bolo 69 tis�c pridan� do ko��ka a a 22 tis�c n�kupov.

### Yoochoose ([link](https://2015.recsyschallenge.com/challenge.html))
T�to d�tov� mno�ina obsahuje d�ta o kliknutiach a n�kupoch pou��vate�ov v elektronickom obchode v r�mci jedn�ho sedenia. Je rozdelen� do dvoch s�borov, pod�a akcie na kliknutia a n�kupy. Tabu�ka s kliknutiami obsahuje ID sedenia, �as akcie, ID produktu a kateg�riu produktu. Tabu�ka s n�kupmi obsahuje ID sedenia, �as akcie, ID produktu, cenu produktu a mno�stvo nak�pen�ho produktu. D�tov� sada obsahuje 9 mili�nov seden�, 33 mili�nov kliknut� a 1,1 mili�na n�kupov.


## N�vrh

Nako�ko sa uk�zalo, �e kombin�ciou dlhodob�ch a kr�tkodob�ch preferenci� je mo�n� dosiahnu� zauj�mav� v�sledky, chceli by sme sa tomuto venova� aj v na�ej pr�ci.
Mo�nou architekt�rou teda s� dve samostatn� rekurentn� siete (napr. LSTM). Prv� z nich m� na vstupe v�etky historick� interakcie pou��vate�a s polo�kami a druh� zoh�ad�uje len posledn�ch N akci� (bli��ie ur��me na z�klade anal�zy d�t a konzult�ci�). N�sledne by sme porovnali v�sledky dosiahnut� jednotliv�mi sie�ami ako aj ich vhodnej kombin�cie.


## N�vrh modelu

![N�vrh modelu](model.png)

## Zdroje

[1] DEVOOGHT, Robin; BERSINI, Hugues. Collaborative filtering with recurrent neural networks. arXiv preprint arXiv:1608.07400, 2016.

[2] HE, Xiangnan, et al. Neural collaborative filtering. In: Proceedings of the 26th international conference on world wide web. International World Wide Web Conferences Steering Committee, 2017. p. 173-182.

[3] ZHENG, Yin, et al. A neural autoregressive approach to collaborative filtering. arXiv preprint arXiv:1605.09477, 2016.

[4] LI, Zhi, et al. Learning from history and present: Next-item recommendation via discriminatively exploiting user behaviors. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018. p. 1734-1743.

