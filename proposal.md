# Odporúčanie produktov v elektronickom obchode

Elektronický obchod sa stal neoddeliteľnou súčasťou života takmer každého z nás. V takomto obchode si možno dnes bežne kúpiť tisíce rôznych produktov. Nedá sa teda očakávať, že používateľ bude prechádzať všetky produkty až kým nenájde ten správny. Je tiež dokázané, že keď používateľ do istého okamihu na webovom sídle neobjaví relevantné informácie, odíde. V prípade elektronického obchodu možno de facto hovoriť o strate zákazníka, ktorý si produkt kúpi u konkurencie.

Pre predajcov je preto kľúčové zákazníkovi v elektronickom obchode zjednodušiť prechod k produktom, o ktoré by potenciálne mohol mať záujem. Ide teda o personalizované odporúčanie produktov zákazníkom. Toto sa zvykne riešiť rôznymi odporúčacími algoritmami. Medzi najrozšírenejšie prístupy k odporúčaniu patrí obsahové a kolaboratívne filtrovanie. V posledných rokoch sa však ukázalo, že tento problém možno riešiť aj s využitím hlbokých neurónových sietí, ktoré v niektorých prípadoch dosahujú signifikantne lepšie výsledky ako tradičné prístupy k odporúčaniu.


## Existujúce riešenia
Autori v práci [[1]](https://arxiv.org/abs/1608.07400) ukázali, že pomocou rekurentnej neurónovej siete je možné dosiahnuť lepšie výsledky ako state-of-the-art prístupy pri krátkodobom odporúčaní. Použitím LSTM dokázali na základe sekvencie interakcií používateľa s položkami predikovať jeho nasledujúce interakcie. Metódu overili na dátových množinách MovieLens 1M a Netflix. Okrem výrazného zlepšenia v krátkodobých predikciách sa v prípade domény MovieLens podarilo dosiahnuť aj zlepšenie v úplnosti (angl. recall), ktorú možno interpretovať ako dlhodobú predikciu. Došlo tiež k výraznému zvýšeniu diverzity správne odporučených položiek (v oboch doménach).


V práci [[2]](https://dl.acm.org/citation.cfm?id=3052569) autori spojili maticovú faktorizáciu a hlbokú neurónovú sieť. Ich model pozostáva z dvoch paralelných vetiev, modelov. V jednom modely pomocou neurónovej siete vykonávajú maticovú faktorizáciu a druhý model je hlboká neurónová sieť. Každý model sa predtrénuje samostatne a pri predikcii spoja tieto dva modely vo výstupnej vrstve. Spojením týchto dvoch modelov a ich samostatným predtrénovaním dostali pri experimentoch najlepšie výsledky. Vstupom do modelu je one hot encoding používateľa a položky a výstupom je pravdepodobnosť, že používateľ kúpi danú položku. Metódu overili na dátových množinách MovieLens 1M a Pinterest..

 
V práci [[3]](https://arxiv.org/abs/1605.09477) používajú na odporúčanie prístup pomocou neurónovej autoregresie. Tento prístup je podobný rekurentnej neurónovej sieti. Model na základe toho ako používateľ ohodnotil položky vypočíta rank(očakávané hodnotenie) všetkých položiek, ktorý použije na výber položiek, ktoré sa budú používateľovi najviac páči. Model vyhodnocovali na dátových množinách Netflix, MovieLens 1M a 10M. Pri experimentoch zistili, že hlbšie neurónové siete dosahujú lepší výsledok.

Kombináciou dlhodobých a krátkodobých preferencií používateľa získaných dvomi LSTM neurónovými sieťami sa autorom v práci [[4]](https://dl.acm.org/citation.cfm?id=3220014) podarilo prekonať metódu HRNN Init, ktorá bola považovaná za state-of-the-art v úlohe odporúčania nasledujúcej položky. Navrhnutá metóda BINN pozostáva z dvoch hlavných komponentov: vnorenie položiek (angl. item embedding) a učenie správania používateľa. Prvá časť predstavuje item2vec obohatený o zohľadnenie frekvencie položky ako váhového faktora. Druhú časť tvoria dve LSTM siete určené na zachytenie historicky stabilných dlhodobých preferencií používateľa a krátkodobého zámeru používateľa v rámci súčasného sedenia. Metóda bola overená na dvoch e-commerce dátových množinách JD a Tianchi.

## Dátové množiny
Na trénovanie a vyhodnotenie modelu budeme používať dátovú množinu, ktorá pozostáva z logov aktivity používateľa v elektronickom obchode, teda aké položky si zobrazil, pridal do košíka a ktoré z nich nakúpil.

Z dostupných množín sme vybrali dve z prostredia elektronického obchodu. Prvou je Retailrocket, ktorá bola zverejnená na Kaggle a druhou (záložnou) je Yoochoose, ktorá bola zverejnená v rámci Recsys Challenge 2015. Jej nevýhodou však je, že neobsahuje ID používateľa ale iba sedenia, čo neumožňuje skúmať dlhodobé preferencie používateľa.

### Retailrocket ([link](https://www.kaggle.com/retailrocket/ecommerce-dataset),[analýza](https://github.com/LubomirKoprla/fiit_project_neural_networks/blob/master/notebooks/analysis-ecommerce.ipynb))
Dátová množina obsahuje súbor s akciami, ktoré používateľ vykonal. Akcie sa delia na zobrazenie produktu, vloženie produktu do košíka a nákup produktu. Tabuľka obsahuje čas vykonania aktivity, ID používateľa, ID produktu a typ akcie. K dispozícii je aj súbor, ktorý obsahuje informácie o produktoch - kategóriu, dostupnosť a anonymizované atribúty produktu. Dátová množina obsahuje 2,7 milióna akcii, z toho bolo 69 tisíc pridaní do košíka a a 22 tisíc nákupov.

### Yoochoose ([link](https://2015.recsyschallenge.com/challenge.html),[analýza](https://github.com/LubomirKoprla/fiit_project_neural_networks/blob/master/notebooks/analysis-yoochoose.ipynb))
Táto dátová množina obsahuje dáta o kliknutiach a nákupoch používateľov v elektronickom obchode v rámci jedného sedenia. Je rozdelená do dvoch súborov, podľa akcie na kliknutia a nákupy. Tabuľka s kliknutiami obsahuje ID sedenia, čas akcie, ID produktu a kategóriu produktu. Tabuľka s nákupmi obsahuje ID sedenia, čas akcie, ID produktu, cenu produktu a množstvo nakúpeného produktu. Dátová sada obsahuje 9 miliónov sedení, 33 miliónov kliknutí a 1,1 milióna nákupov.


### Záver
Na základe analýzy dát sme sa rozhodli použiť dátovú množinu Yoochoose. Retailrocket sme vylúčili z dôvodu, že väčšina ľudí mala iba interakciu s jedným produktom. V dátovej množine Yoochoose bol priemerný počet produktov na sedenie 2,87 a obsahovala oveľa viac dát. Preto bolo možné odfiltrovať používateľov s nízkym počtom interakcií pri zachovaní dostatočného množstva dát na trénovanie modelu.

## Návrh
Projekt sme sa rozhodli riešiť rekurentnou neurónovou sieťou, konkrétne LSTM, vzhľadom na jej úspešnosť pri riešení podobného problému v iných doménach.
Vstup je teda tvorený sekvenciami interakcií používateľov s položkami (všetky typy udalostí: nákupy aj videnia) s fixnou dĺžkou s využitím post-paddingu. Pred samotnou LSTM vrstvou sa nachádza embedding vrstva, ktorej cieľom je zachytiť črty jednotlivých produktov vo forme latentných vektorov a tiež poskytnúť LSTM vrstve informácie o maskovaní. Za LSTM vrstvou je umiestnená Dense vrstva. Na výstupe je pre každú položku vypočítaná pravdepodobnosť, že sa bude nachádzať v nasledujúcich interakciách daného používateľa.

![Návrh modelu](model.png)


## Implementácia

### Predspracovanie dát
Pri predspracovaní dát sme si najprv spravili predvýber produktov a používateľov, s ktorými chceme pracovať. V prvom kroku sme si odfiltrovali produkty, ktoré majú menej ako 20 interakcií. V ďalšom kroku sme odfiltrovali používateľov, ktorí majú menej ako 20 a viac ako 50 interakcií. Takto vznikla dátová množina obsahujúca 23091 rôznych produktov a 35996 používateľov (= sekvencií) s priemerným počtom 27 interakcií na používateľa.
Následne sme interakcie každého používateľa rozdelili v pomere 80:20 (so zohľadnením časovej následnosti) na položky na základe, ktorých predikujeme a položky, ktoré chceme predikovať. Tzn. na základe 16-40 historických interakcií predikujeme nasledujúcich 4-10 interakcií.


### Spôsob trénovania
Trénovanie prebiehalo v prostredí Google Cloud na VM s 2xCPU, 24GB RAM, 1xTesla K80. Využili sme Docker kontajner založený na obraze `tensorflow/tensorflow:2.0.0-gpu-py3` doplnený o niektoré ďalšie knižnice.
Aplikovali sme náhodné prehľadávanie pre hľadanie optimálnych hyperparametrov modelu. Tie zahŕňali: dĺžku embeddingov, počet LSTM jednotiek, dropout, rýchlosť učenia, veľkosť dávky, aktivačné funkcie na každej vrstve a 3 parametre optimalizátora Adam (beta1, beta2, epsilon). V prvom kroku sa vyberie náhodných 20% používateľov (sekvencií), ktorí tvoria testovaciu množinu. Zvyšní používatelia sa delia na trénovaciu a validačnú množinu v pomere 80:20.

Maximálny počet epoch bol nastavený na 250, pričom sme využili tzv. EarlyStopping s trpezlivosťou 10 epoch pri monitorovaní metriky R@10 na validačnej časti dát. R@k predstavuje mieru pokrytia v top k výsledkoch. Počíta sa ako:

> | {relevantné produkty} ∩ {top k produkty} | / | {relevantné produkty} |

A vzhľadom na to, že maximálny počet relevantných produktov je tiež 10, teoretické maximum tejto metriky je 1 a priamo vyjadruje ako dobre model vie zoradiť produkty podľa relevantnosti.

Monitorovali sme však aj iné metriky (R@50, R@100, P@1, P@3, P@5, P@10). P@k, teda presnosť v top k sa počíta ako:

>  | {relevantné produkty} ∩ {top k produkty} | / k

V našom prípade je minimálny počet relevantných produktov 4, preto jej teoretické maximum dosahuje 1 iba v prípadoch kedy k<=4.

Model bol vyhodnocovaný na testovacej množine v každej konfigurácii hyperparametrov len raz, po nájdení finálnej epochy (spravidla určenej EarlyStopping-om). Podobne, ukladal sa vždy len najlepší model.
Využili sme tiež integráciu slacku na monitorovanie správneho chodu VM a trénovania.

## Výsledky
Z pohľadu pokrytia (R@10, R@50 a R@100) dosiahol model najlepšie výsledky na validačnej množine v epoche 91 pri nasledujúcich hyperparametroch:
```
emb_dim: 300
lstm_units: 40
lstm_activation: 'tanh'
lstm_recurrent_activation: 'sigmoid'
lstm_dropout: 0.2
lstm_recurrent_dropout: 0.2
dense_activation: 'softmax'
batch_size: 128
learning_rate: 0.001
adam_beta_1: 0.9
adam_beta_2: 0.999
adam_epsilon: 0.00000001

```

| množina | P@1 | P@3 | P@5 | P@10 | R@10 | R@50 | R@100 |
| -- | -- | -- | -- | -- | -- | -- | -- |
| val  | 5.7118% | 5.1678% | 4.7326% | 3.9722% | 7.9502% | 19.299% | 26.947% |
| test | 4.8750% | 4.6944% | 4.4056% | 3.6306% | 7.2316% | 18.660% | 26.060% |



Z pohľadu presnosti (P@1, P@3, P@5) dosiahol model najlepšie výsledky na validačnej množine v epoche 38 pri nasledujúcich hyperparametroch:
```
emb_dim: 250
lstm_units: 175
lstm_activation: 'sigmoid'
lstm_recurrent_activation: 'linear'
lstm_dropout: 0.3
lstm_recurrent_dropout: 0.1
dense_activation: 'tanh'
batch_size: 256
learning_rate: 0.0001
adam_beta_1: 0.975
adam_beta_2: 0.899
adam_epsilon: 0.000000001

```

| množina | P@1 | P@3 | P@5 | P@10 | R@10 | R@50 | R@100 |
| -- | -- | -- | -- | -- | -- | -- | -- |
| val  | 6.5625% | 5.8738% | 5.0208% | 3.7344% | 7.4802% | 14.494% | 18.455% |
| test | 6.5139% | 5.7361% | 4.8972% | 3.7292% | 7.4548% | 14.638% | 18.708% |

## Zdroje

[1] DEVOOGHT, Robin; BERSINI, Hugues. Collaborative filtering with recurrent neural networks. arXiv preprint arXiv:1608.07400, 2016.

[2] HE, Xiangnan, et al. Neural collaborative filtering. In: Proceedings of the 26th international conference on world wide web. International World Wide Web Conferences Steering Committee, 2017. p. 173-182.

[3] ZHENG, Yin, et al. A neural autoregressive approach to collaborative filtering. arXiv preprint arXiv:1605.09477, 2016.

[4] LI, Zhi, et al. Learning from history and present: Next-item recommendation via discriminatively exploiting user behaviors. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018. p. 1734-1743.
