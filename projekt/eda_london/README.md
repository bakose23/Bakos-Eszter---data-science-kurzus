# Londoni időjárási adatok feltáró elemzése

- Bemenet: `selected_8_places_2017_2023.csv`; az eredeti CSV-fájl nem módosult.
- Adatkör: 20,440 napi sor, 8 helyszín, 2017-01-01 és 2023-12-31 között.
- Minden helyszínhez 2,555 sor tartozik; pontos duplikált sorok száma: 0.
- Hiányzó értékek száma: 0; duplikált helyszín-dátum kulcsok száma: 0.
- Hiányzó dátum: 2020-02-29 minden helyszínnél hiányzik; ezt a napot nem interpoláltuk.
- A napi sorok 36.6%-án legalább 1 mm csapadék hullott; a csapadékos napok mediánja: 3.90 mm.
- A napi hőmérsékleti terjedelem átlaga: 6.50 °C; az átlagos maximum-hőmérséklet július hónapban volt a legmagasabb.
- A legnagyobb havi átlagos csapadék október hónapban fordult elő (3.08 mm/nap).
- A másnapi csapadék célváltozóhoz 20,424 érvényes pár tartozik; 16 sort kizártunk, a pozitív osztály aránya 36.6%.
- A helyszínenkénti csapadékosnap-arány (>=1 mm) 35.3%–42.7% között alakult.
- A legnagyobb Kruskal–Wallis-hatásméret ε²=0.7599; a csapadékosnap-arány tesztje χ²(7)=49.0, Cramér-féle V=0.049 (tájékoztató p=2.24e-08; az autokorreláció korlátozza a következtetést).
- Kimenetek: futtatott notebook, 15 nagy felbontású ábra, valamint CSV-audit- és statisztikai táblák a `figures/` és `tables/` mappákban.
- Korlátok: egy szökőnap hiányzik; a helyszínkiválasztás rögzített; a geokódolási/audit eredete, valamint a forrás mérési definíciói korlátozzák az értelmezést.
