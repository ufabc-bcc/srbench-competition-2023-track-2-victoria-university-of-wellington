import pandas as pd

task = 'Berri1'
weather_data = pd.read_csv('datasets/WeatherInfo.csv')
x0 = weather_data['Max Temp (°C)']
x1 = weather_data['Min Temp (°C)']
x2 = weather_data['Mean Temp (°C)']
x3 = weather_data['Total Rain (mm)']
x4 = weather_data['Total Snow (cm)']
x5 = weather_data['Total Precip (mm)']
x6 = weather_data['Snow on Grnd (cm)']
x7 = weather_data['Spd of Max Gust (km/h)']

if task == "Berri1":
    y = (4006.1884470699496 * x0 * (x3 + 1) + 895.93498936080078 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (227.41665975759217 * x0 ** 2 * (x3 - 1) + 311.87682016130872 * x0 ** 2
                                     - 492.40940134427135 * x0 - 492.40940134427135 * x2 - 492.40940134427135 * x3
                                     + 2622.6277944885887)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Boyer":
    y = (3327.335052998245 * x0 * (x3 + 1) + 767.60598361693462 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (241.6815430702618 * x0 ** 2 * (x3 - 1) + 414.73774180073804 * x0 ** 2
                                     - 410.0823713202159 * x0 - 410.0823713202159 * x2 - 410.0823713202159 * x3
                                     + 1884.585658776706)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Brébeuf":
    y = (4013.020959348472 * x0 * (x3 + 1) + 962.12613268036703 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (241.87503840788541 * x0 ** 2 * (x3 - 1) + 426.7581461459809 * x0 ** 2
                                     - 466.5861689907295 * x0 - 466.5861689907295 * x2 - 466.5861689907295 * x3
                                     + 2481.1831737879266)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "CSC (Côte Sainte-Catherine)":
    y = (1760.989684843327 * x0 * (x3 + 1) + 286.07766698084908 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (57.288230039707088 * x0 ** 2 * (x3 - 1) - 2.1643401866425148 * x0 ** 2
                                     - 186.47540403094915 * x0 - 186.47540403094915 * x2 - 186.47540403094915 * x3
                                     + 1113.473017513001)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Maisonneuve_2":
    y = (2767.8951122303687 * x0 * (x3 + 1) + 411.4364755422179 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (84.012340359532522 * x0 ** 2 * (1 - x3) + 79.352666409670279 * x0 ** 2
                                     - 71.690033764096371 * x0 - 71.690033764096371 * x2 - 71.690033764096371 * x3
                                     + 1855.2583015288686)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Maisonneuve_3":
    y = (1376.272566721658 * x0 * (x3 + 1) + 350.66698433482616 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (122.6032598033499 * x0 ** 2 * (x3 - 1) + 149.74404572282371 * x0 ** 2
                                     - 198.1899609772432 * x0 - 198.1899609772432 * x2 - 198.1899609772432 * x3
                                     + 934.38759566354166)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Notre-Dame":
    y = (1592.121556513564 * x0 * (x3 + 1) + 429.38363608519765 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (87.077644308054505 * x0 ** 2 * (x3 - 1) + 213.49024608491401 * x0 ** 2
                                     - 193.5594148710289 * x0 - 193.5594148710289 * x2 - 193.5594148710289 * x3
                                     + 938.20537375438427)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Parc":
    y = (2530.905169332978 * x0 * (x3 + 1) + 386.11949817642695 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (44.360434973494064 * x0 ** 2 * (x3 - 1) - 87.098539815268423 * x0 ** 2
                                     - 241.9026779532639 * x0 - 241.9026779532639 * x2 - 241.9026779532639 * x3
                                     + 1709.5377899819756)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "PierDup":
    y = (1208.8021968618153 * x0 * (x3 + 1) + 722.93439258689226 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (244.69044541391267 * x0 ** 2 * (x3 - 1) + 712.87815880486393 * x0 ** 2
                                     - 229.7212048301016 * x0 - 229.7212048301016 * x2 - 229.7212048301016 * x3
                                     + 695.45168591766173)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Rachel / Hôtel de Ville":
    y = (3061.4108954448401 * x0 * (x3 + 1) + 607.52026323758795 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (213.18379656420468 * x0 ** 2 * (x3 - 1) + 179.42389565879843 * x0 ** 2
                                     - 366.4280986191666 * x0 - 366.4280986191666 * x2 - 366.4280986191666 * x3
                                     + 2075.5006560553626)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Rachel / Papineau":
    y = (3777.4748696467641 * x0 * (x3 + 1) + 859.77813649715898 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (274.00448733839164 * x0 ** 2 * (x3 - 1) + 302.25612627127807 * x0 ** 2
                                     - 470.4457388410513 * x0 - 470.4457388410513 * x2 - 470.4457388410513 * x3
                                     + 2732.6560917381231)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "René-Lévesque":
    y = (1993.224951056265 * x0 * (x3 + 1) + 538.70189667343362 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (102.8581062472851 * x0 ** 2 * (x3 - 1) + 268.31010035855753 * x0 ** 2
                                     - 236.796748545691 * x0 - 236.796748545691 * x2 - 236.796748545691 * x3
                                     + 1219.4413364973905)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Saint-Antoine":
    y = (26.38414310912499 * x0 * (x3 + 1) + 127.85452823391213 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (18.832927261252481 * x0 ** 2 * (x3 - 1) + 77.990858788422464 * x0 ** 2
                                     - 20.25182964883686 * x0 - 20.25182964883686 * x2 - 20.25182964883686 * x3
                                     + 163.19098113449322)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Saint-Urbain":
    y = (1944.1625579449 * x0 * (x3 + 1) + 345.34434378138913 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (88.01474974365882 * x0 ** 2 * (x3 - 1) + 28.633833114786593 * x0 ** 2
                                     - 207.38312549624477 * x0 - 207.38312549624477 * x2 - 207.38312549624477 * x3
                                     + 1327.8622926890715)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Totem_Laurier":
    y = (2472.2004581562918 * x0 * (x3 + 1) + 495.22625945625754 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (133.63298577857819 * x0 ** 2 * (x3 - 1) + 191.15534301568599 * x0 ** 2
                                     - 255.4611362373323 * x0 - 255.4611362373323 * x2 - 255.4611362373323 * x3
                                     + 1722.7364195933516)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "University":
    y = (2581.4994852543992 * x0 * (x3 + 1) + 484.05295145494469 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (39.677710848382224 * x0 ** 2 * (x3 - 1) - 89.960713931285996 * x0 ** 2
                                     - 270.6070792053503 * x0 - 270.6070792053503 * x2 - 270.6070792053503 * x3
                                     + 1981.0614408067537)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Viger":
    y = (333.159518385166 * x0 * (x3 + 1) + 98.047890708534686 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (12.206097616552556 * x0 ** 2 * (x3 - 1) + 30.672361141960894 * x0 ** 2
                                     - 36.90158783498249 * x0 - 36.90158783498249 * x2 - 36.90158783498249 * x3
                                     + 260.64185406154223)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Maisonneuve_1":
    y = (320.29033568112828 * x0 * (x3 + 1) + 45.770842720594488 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (4.3877236792299522 * x0 ** 2 * (x3 - 1) - 10.767186034055725 * x0 ** 2
                                     - 42.34744490643222 * x0 - 42.34744490643222 * x2 - 42.34744490643222 * x3
                                     + 247.54288183753355)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Pont_Jacques_Cartier":
    y = (1966.9292522286531 * x0 * (x3 + 1) + 629.09962198278407 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (288.45004876445341 * x0 ** 2 * (x3 - 1) + 717.6450032203294 * x0 ** 2
                                     - 341.36714159859637 * x0 - 341.36714159859637 * x2 - 341.36714159859637 * x3
                                     + 1161.4757975153628)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Saint-Laurent U-Zelt Test":
    y = (4687.2625106068964 * x0 * (x3 + 1) - 149.7182766114505 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (954.2090501957726 * x0 ** 2 * (x3 - 1) - 1231.2376338964424 * x0 ** 2
                                     - 483.3615367154681 * x0 - 483.3615367154681 * x2 - 483.3615367154681 * x3
                                     + 4796.9867397740311)) \
        / ((x3 + 1) * (x0 * x2 + 1))
if task == "Parc U-Zelt Test":
    y = (933.1108822613878 * x0 * (x3 + 1) + 178.8817890307546 * x2 * (x0 * x2 + 1) +
         (x3 + 1) * (x0 * x2 + 1) * (455.18225603043742 * x0 ** 2 * (x3 - 1) - 1030.237473438544 * x0 ** 2
                                     - 106.49635824202948 * x0 - 106.49635824202948 * x2 - 106.49635824202948 * x3
                                     + 2225.6640807023284)) \
        / ((x3 + 1) * (x0 * x2 + 1))

print(y)
