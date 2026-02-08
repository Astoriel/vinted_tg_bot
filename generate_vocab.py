import json

words_data = []
seen = set()

def add(ws, w):
    for word in ws:
        word = word.strip().lower().rstrip('.')
        if word and word not in seen:
            seen.add(word)
            words_data.append({"word": word, "weight": w})

# ── Layer 1: 1000 most common English words (high weight) ──
with open("1000-most-common-words.txt", "r") as f:
    common = [line.strip().lower().rstrip('.') for line in f if line.strip()]
add(common, 3.0)

# ── Layer 2: Concrete visual nouns ──

# Animals (150)
add(["butterfly","elephant","tiger","lion","rabbit","deer","whale","shark","frog",
     "turtle","eagle","wolf","fox","penguin","dolphin","monkey","chicken","owl","parrot",
     "ant","bee","spider","crab","lobster","octopus","jellyfish","seahorse","starfish",
     "snail","worm","squirrel","hamster","mouse","rat","hedgehog","raccoon","panda",
     "koala","kangaroo","giraffe","zebra","hippo","rhino","gorilla","camel","llama",
     "donkey","goat","sheep","lamb","rooster","goose","swan","flamingo","peacock",
     "crow","raven","hawk","falcon","pigeon","otter","beaver","moose","bison","leopard",
     "cheetah","jaguar","lynx","hyena","mole","ferret","iguana","chameleon","gecko",
     "crocodile","alligator","toad","scorpion","dragonfly","grasshopper","beetle",
     "ladybug","moth","caterpillar","shrimp","squid","eel","tuna","salmon","trout",
     "goldfish","wasp","mosquito","roach","puppy","kitten","calf","tadpole","cub",
     "pelican","stork","vulture","seagull","hummingbird","woodpecker","toucan",
     "chinchilla","porcupine","newt","clam","oyster","sardine","mackerel","barracuda",
     "koi","slug","firefly","hornet","termite","piranha","swordfish","herring",
     "mantis","centipede","anemone","mussel","weasel","boar","panther","cougar",
     "buffalo","elk","salamander","manatee","armadillo","opossum","badger","mink",
     "wolverine","tapir","cobra","python","viper","tortoise","narwhal","albatross",
     "osprey","condor","quail","pheasant"], 2.0)

# Objects (200)
add(["desk","lamp","phone","computer","keyboard","screen","bottle","cup","plate",
     "bowl","fork","knife","spoon","pen","pencil","mirror","pillow","blanket",
     "sofa","carpet","curtain","shelf","drawer","cabinet","basket","suitcase",
     "backpack","wallet","lock","necklace","bracelet","crown","helmet","mask",
     "glasses","umbrella","fan","candle","lighter","scissors","hammer","nail",
     "screw","drill","wrench","brush","comb","razor","towel","soap","sponge",
     "bucket","broom","mop","vacuum","needle","thread","button","zipper",
     "chain","cable","pipe","hose","gear","magnet","battery","bulb","switch",
     "plug","antenna","camera","binoculars","telescope","microscope","compass",
     "globe","flag","drum","guitar","piano","violin","flute","trumpet",
     "microphone","speaker","headphones","television","remote","calculator",
     "printer","robot","satellite","rocket","motor","propeller","anchor",
     "paddle","trap","cage","sword","shield","arrow","bow","medal","trophy",
     "stamp","envelope","newspaper","magazine","coin","ticket","passport",
     "badge","glue","eraser","ruler","dice","puzzle","balloon","kite","toy",
     "doll","marble","tent","lantern","flashlight","grill","oven","stove",
     "microwave","toaster","blender","kettle","pot","pan","spatula","whisk",
     "tray","mug","pitcher","vase","jar","barrel","cart","bench","fountain",
     "mailbox","alarm","siren","whistle","crayon","chalk","marker","notebook",
     "clipboard","whiteboard","easel","canvas","frame","sculpture","pottery",
     "quilt","mattress","scarf","glove","jacket","shirt","pants","skirt",
     "sneaker","boot","sandal","hook","hinge","knob","lever","pulley",
     "faucet","nozzle","funnel","sieve","tongs","clamp","screwdriver","chisel",
     "anvil","bolt","rivet","staple","pin","thimble","latch",
     "piston","valve","gauge","dial","pendulum","prism","lens","spool","reel"], 2.0)

# Nature (180)
add(["leaf","branch","root","bark","seed","petal","thorn","vine","bush","moss",
     "fern","mushroom","cactus","bamboo","palm","oak","pine","maple","birch",
     "willow","tulip","daisy","sunflower","lily","orchid","violet","iris","lotus",
     "dandelion","lavender","jasmine","poppy","daffodil","marigold","hibiscus",
     "clover","ivy","holly","acorn","pinecone","coconut","jungle","meadow",
     "oasis","swamp","pond","waterfall","volcano","canyon","cliff","cave",
     "plateau","ridge","peak","summit","glacier","iceberg","avalanche","earthquake",
     "tsunami","hurricane","tornado","blizzard","lightning","thunder","rainbow",
     "aurora","sunset","sunrise","dawn","dusk","fog","mist","frost","dew",
     "breeze","tide","reef","fjord","lagoon","delta","beach","gravel","pebble",
     "boulder","seaweed","kelp","algae","mud","clay","dust","ash","lava",
     "granite","limestone","sandstone","quartz","crystal","diamond","ruby",
     "emerald","sapphire","opal","amber","jade","pearl","copper","savanna",
     "prairie","taiga","dune","crater","basin","brook","cascade","erosion",
     "sediment","peninsula","gulf","strait","atoll","marsh","bog","creek",
     "gorge","mesa","drought","hail","cyclone","monsoon","whirlpool","slate",
     "obsidian","magma","stalactite","ore","geyser","rapids","estuary","bluff",
     "butte","knoll","mound","moraine","fossil","mineral","gem","geode",
     "amethyst","topaz","garnet","turquoise","malachite","pyrite","hematite",
     "agate","onyx","jasper","moonstone","aquamarine","citrine","tourmaline",
     "labradorite","tanzanite"], 1.5)

# Food (150)
add(["banana","grape","strawberry","blueberry","raspberry","cherry","peach","plum",
     "pear","mango","pineapple","watermelon","melon","kiwi","lemon","lime",
     "pomegranate","papaya","avocado","tomato","potato","carrot","onion","garlic",
     "pepper","chili","corn","rice","wheat","toast","bagel","croissant","muffin",
     "cake","pie","cookie","donut","pancake","waffle","pizza","pasta","noodle",
     "soup","stew","salad","sandwich","burger","hotdog","taco","burrito","sushi",
     "dumpling","curry","steak","bacon","sausage","ham","cheese","butter","cream",
     "yogurt","omelet","cereal","honey","jam","syrup","chocolate","candy",
     "ice_cream","pudding","jelly","marshmallow","popcorn","pretzel","chip",
     "cracker","almond","walnut","cashew","peanut","olive","pickle","tofu",
     "broccoli","cauliflower","spinach","kale","lettuce","cabbage","celery",
     "cucumber","zucchini","eggplant","squash","pumpkin","beet","radish",
     "sweet_potato","ginger","cinnamon","vanilla","mint","basil","oregano",
     "thyme","rosemary","parsley","ketchup","mustard","mayonnaise","vinegar",
     "soy_sauce","hot_sauce","salsa","guacamole","hummus","brownie","crepe",
     "ramen","risotto","lasagna","mousse","tiramisu","cheesecake","eclair",
     "macaron","caramel","fudge","meringue","sorbet","gelato","smoothie",
     "milkshake","lemonade","espresso","cappuccino","latte","cocoa","granola",
     "oatmeal","cupcake","biscuit","wonton","ravioli","toffee","nougat","marzipan"], 2.0)

# ── Layer 3: People, vehicles, buildings ──

# People/Roles (100)
add(["teenager","adult","elder","stranger","teacher","nurse","chef","artist",
     "musician","dancer","athlete","firefighter","police","pilot","farmer",
     "carpenter","baker","waiter","judge","lawyer","priest","monk","knight",
     "prince","princess","warrior","samurai","ninja","pirate","cowboy","astronaut",
     "scientist","wizard","witch","ghost","skeleton","zombie","vampire","angel",
     "devil","fairy","mermaid","dragon","unicorn","phoenix","clown","acrobat",
     "magician","ballerina","sculptor","photographer","writer","architect",
     "engineer","surgeon","dentist","plumber","electrician","librarian","barber",
     "tailor","poet","conductor","filmmaker","veterinarian","pharmacist","janitor",
     "mime","juggler","puppeteer","detective","spy","thief","merchant",
     "sailor","gladiator","viking","pharaoh","emperor","empress","duke","duchess",
     "baron","jester","herald","squire","scribe","oracle","shaman","hermit",
     "nomad","pilgrim","refugee","explorer","pioneer","inventor","prophet"], 1.5)

# Body parts (50)
add(["elbow","wrist","thumb","chest","stomach","waist","hip","knee","ankle",
     "toe","heel","muscle","lung","liver","kidney","spine","skull","jaw",
     "cheek","chin","forehead","eyebrow","lip","tongue","tooth","palm","fist",
     "scar","tattoo","beard","mustache","braid","ponytail","rib","pelvis",
     "collarbone","knuckle","tendon","pupil","retina","eardrum","nostril",
     "dimple","freckle","wrinkle","callus","blister","bruise","vein","artery"], 1.0)

# Vehicles (80)
add(["bus","truck","van","taxi","ambulance","fire_truck","motorcycle","bicycle",
     "scooter","subway","tram","airplane","helicopter","jet","yacht","canoe",
     "kayak","raft","submarine","ferry","sailboat","speedboat","tractor",
     "bulldozer","crane","excavator","forklift","jeep","limousine","sedan",
     "pickup","suv","camper","trailer","chariot","sled","snowmobile","gondola",
     "cable_car","elevator","escalator","rickshaw","drone","unicycle","tricycle",
     "roller_coaster","ferris_wheel","carousel","hot_air_balloon","blimp",
     "glider","seaplane","hovercraft","tugboat","rowboat","catamaran","battleship",
     "lifeboat","barge","steamship","galleon","skateboard","surfboard","snowboard",
     "hang_glider","paraglider","jet_ski","monorail","funicular","chairlift",
     "toboggan","moped","stagecoach","wheelbarrow","carriage","longship",
     "hoverboard","go_kart","dinghy","convertible"], 1.5)

# Buildings (100)
add(["apartment","cabin","cottage","mansion","castle","palace","temple","church",
     "mosque","cathedral","chapel","monastery","pagoda","shrine","pyramid","tomb",
     "tower","lighthouse","windmill","barn","warehouse","factory","dam","bridge",
     "pier","dock","harbor","airport","terminal","garage","highway","alley",
     "tunnel","arch","dome","balcony","terrace","porch","courtyard","playground",
     "stadium","arena","theater","cinema","museum","gallery","library","university",
     "hospital","laboratory","observatory","aquarium","zoo","mall","restaurant",
     "cafe","pub","hotel","motel","hostel","resort","prison","courthouse",
     "embassy","skyscraper","penthouse","loft","attic","basement","lobby",
     "rooftop","plaza","boardwalk","marina","brewery","winery","vineyard",
     "orchard","greenhouse","amphitheater","gazebo","pavilion","kiosk","yurt",
     "igloo","hut","shack","silo","forge","quarry","distillery","conservatory",
     "minaret","steeple","spire","rotunda","colonnade","portico","arcade",
     "dungeon","sanctuary","vault","fortress","citadel","moat"], 1.5)

# ── Layer 4: Actions, adjectives, abstract ──

# Actions (120)
add(["jumping","swimming","climbing","crawling","sliding","rolling","spinning",
     "dancing","singing","cooking","drinking","sleeping","kneeling","crouching",
     "bending","stretching","pulling","pushing","lifting","carrying","throwing",
     "catching","kicking","punching","hugging","kissing","waving","pointing",
     "clapping","laughing","crying","screaming","whispering","shouting","smiling",
     "frowning","digging","planting","watering","harvesting","chopping","slicing",
     "stirring","pouring","folding","wrapping","tying","sewing","knitting",
     "weaving","washing","scrubbing","wiping","sweeping","polishing","grinding",
     "sawing","drilling","hammering","gluing","welding","carving","sculpting",
     "assembling","repairing","smashing","crushing","tearing","melting","freezing",
     "boiling","baking","frying","grilling","roasting","steaming","splashing",
     "spraying","squeezing","shaking","scratching","typing","scrolling","swiping",
     "photographing","filming","recording","broadcasting","uploading","navigating",
     "steering","parking","loading","stacking","sorting","measuring","weighing",
     "designing","coding","debugging","launching","landing","docking","sailing",
     "rowing","paddling","surfing","diving","snorkeling","fishing","hunting",
     "foraging","gathering","collecting","storing","preserving","composting"], 1.5)

# Adjectives/Colors (80)
add(["crimson","scarlet","maroon","teal","cyan","navy","cobalt",
     "indigo","magenta","fuchsia","lilac","mauve","beige","ivory","khaki",
     "chartreuse","enormous","narrow","thick","shiny","dull","smooth","rough",
     "curved","crooked","hollow","ancient","elegant","fancy","shallow","dense",
     "sparse","transparent","fragile","sturdy","flexible","rigid","fuzzy","fluffy",
     "striped","spotted","glossy","matte","rusty","vibrant","glowing","sparkling",
     "metallic","iridescent","opaque","brittle","elastic","jagged","rippled",
     "wavy","zigzag","spherical","cylindrical","conical","symmetrical","geometric",
     "abstract","miniature","colossal","translucent","ornate","corrugated",
     "perforated","woven","knotted","tangled","tattered","serrated","tapered",
     "concave","convex","grooved","ridged","fractal","radial"], 1.0)

# Emotions (40)
add(["sadness","anxiety","excitement","boredom","pride","shame","guilt","envy",
     "jealousy","gratitude","loneliness","nostalgia","awe","confusion","curiosity",
     "determination","courage","patience","wisdom","kindness","compassion","empathy",
     "doubt","relief","regret","anticipation","serenity","melancholy","euphoria",
     "dread","panic","bliss","ecstasy","agony","grief","sorrow","fury","delight",
     "devotion","mercy"], 1.0)

# Music (30)
add(["rhythm","tempo","treble","chorus","verse","crescendo","staccato",
     "legato","vibrato","arpeggio","riff","solo","duet","orchestra","symphony",
     "concerto","sonata","fugue","prelude","overture","cadence","anthem","ballad",
     "lullaby","waltz","samba","tango","groove","syncopation","harmony_music"], 1.0)

# Sports (40)
add(["racket","hoop","puck","jersey","referee","coach","champion","podium",
     "marathon","sprint","hurdle","javelin","gymnastics","wrestling","boxing",
     "fencing","archery","skiing","snowboarding","skating","hockey","soccer",
     "football","basketball","baseball","tennis","volleyball","rugby","golf",
     "bowling","chess","checkers","dominoes","poker","roulette","checkmate",
     "slalom","triathlon","decathlon","pentathlon"], 1.0)

# Shapes (20)
add(["hexagon","octagon","pentagon","trapezoid","rhombus","ellipse","parabola",
     "helix","torus","mobius","tesseract","dodecahedron","icosahedron","tetrahedron",
     "tessellation","mandala","kaleidoscope","labyrinth","vortex","lattice"], 1.0)

# Materials (60)
add(["rubber","fabric","brick","concrete","aluminum","brass","porcelain","plaster",
     "cement","resin","fiberglass","carbon_fiber","nylon","polyester","vinyl",
     "silicone","foam","suede","denim","corduroy","flannel","fleece","satin",
     "lace","burlap","hemp","rattan","wicker","cork","teak","mahogany","ebony",
     "cedar","plywood","cardboard","parchment","wax","leather","velvet","silk",
     "linen","wool","chrome","titanium","bronze","tin","zinc","nickel","platinum",
     "pewter","wrought_iron","stainless_steel","graphite","charcoal","basalt",
     "soapstone","travertine","ceramic_material","marble_material","sandpaper"], 1.0)

# Art concepts (30)
add(["silhouette","mosaic","fresco","mural","cameo","filigree","cloisonne",
     "kintsugi","raku","celadon","terracotta","majolica","sgraffito",
     "chiaroscuro","sfumato","impasto","calligraphy","origami","collage",
     "tapestry","batik","woodcut","etching","lithograph","watercolor","gouache",
     "pastel_art","pixel_art","voxel_art","wireframe_art"], 1.0)

# Mythology creatures (40)
add(["centaur","cerberus","cyclops","dryad","elf","gnome","goblin","golem",
     "griffin","harpy","hydra","imp","kraken","leprechaun","manticore","medusa",
     "minotaur","nymph","ogre","pegasus","satyr","siren","sphinx","titan",
     "troll","valkyrie","werewolf","wyvern","yeti","banshee","basilisk",
     "chimera","djinn","ghoul","kitsune","leviathan","lich","orc","wraith",
     "thunderbird"], 1.0)

# Astronomy (20)
add(["constellation","galaxy","nebula","comet","meteor","asteroid","eclipse",
     "supernova","pulsar","quasar","black_hole","white_dwarf","red_giant",
     "neutron_star","solar_system","milky_way","andromeda","orion","sirius",
     "polaris"], 1.0)

# Tech (20)
add(["pixel","algorithm","binary","encryption","firewall","server_tech","database",
     "bandwidth","protocol","terminal_tech","cursor","interface","dashboard",
     "widget","toolbar","notification","shader","render_tech","voxel",
     "wireframe_tech"], 1.0)

# Vessels/Containers (25)
add(["amphora","urn","chalice","goblet","tankard","flagon","decanter","carafe",
     "tureen","teapot","samovar","cauldron","cistern","terrarium","hourglass",
     "sundial","astrolabe","orrery","diorama","canister","keg","cask","demijohn",
     "crucible","mortar_vessel"], 1.0)

# Textiles/Craft (30)
add(["tweed","herringbone","houndstooth","plaid","tartan","gingham","paisley",
     "damask","jacquard","brocade","embroidery","crochet","macrame","patchwork",
     "quilting","shibori","ikat","screen_print","engraving","drypoint",
     "monotype","serigraph","linocut","papier_mache","glassblowing",
     "blacksmithing","silversmithing","woodturning","bookbinding","felting"], 1.0)

# Final extra padding words if still under 3000
extras = [
     "panorama","vista","clearing","grove","thicket","canopy","undergrowth",
     "shoreline","riverbed","wetland","grassland","steppe","rainforest","mangrove",
     "bayou","heath","moor","dale","glen","ravine","gully","cirque","scree",
     "esker","drumlin","oxbow","meander","confluence","tributary","chaparral",
     "scrubland","woodland","glade","coppice","dell","spinney","pliers",
     "ratchet","caliper","protractor","sextant","chronometer","metronome",
     "tuning_fork","oscilloscope","multimeter","soldering_iron","tweezers",
     "forceps","pipette","beaker","flask","barometer","hygrometer","anemometer",
     "altimeter","micrometer","torque_wrench","plumb_bob","chalk_line",
     "spirit_level","bevel_gauge","marking_gauge","dividers","depth_gauge",
     "pediment","cornice","frieze","keystone","cupola","oculus","lintel",
     "chandelier","sconce","candelabra","torchiere","gargoyle","buttress",
     "pinnacle","drawbridge","battlement","turret","parapet","rampart",
     "bourbon","brandy","champagne","cider","cognac","mead","mojito","prosecco",
     "sangria","absinthe","amaretto","gimlet","julep","vermouth","schnapps",
     "flywheel","fulcrum","gimbal","gyroscope","jackscrew","windlass","winch",
     "swivel","turnbuckle","pawl","sheave","crankshaft","camshaft","driveshaft",
     "equinox","solstice","meridian","zenith","azimuth","parallax","precession",
     "albedo","magnitude","luminosity","redshift","spectrum","refraction",
     "diffraction","polarization","fluorescence","phosphorescence",
     "bioluminescence","incandescence","prism_optics","wavelength","amplitude",
     "frequency","resonance","oscillation","vibration","harmonic","overtone",
     "fundamental","timbre","decibel","hertz","octave_sound","waveform",
     "sawtooth","sine_wave","square_wave","noise_signal","distortion","reverb",
     "echo","delay_effect","flanger","phaser","chorus_effect","compression",
     "limiter","equalizer","filter","crossover","tweeter","woofer","subwoofer",
     "diaphragm","coil_speaker","cone_speaker","baffle","cabinet_speaker",
     "enclosure","port_speaker","vent_speaker","passive_radiator","horn_speaker",
     "compression_driver","waveguide","diffusor","absorber","reflector",
     "diffraction_panel","bass_trap","acoustic_panel","isolation","damping",
     "coupling","decoupling","mass_loading","constrained_layer","viscoelastic",
     "modal","standing_wave","room_mode","flutter_echo","comb_filtering",
     "phase_cancellation","constructive","destructive","superposition",
     "interference_pattern","node_wave","antinode","boundary","impedance",
     "reactance","resistance","inductance","capacitance","voltage","current_elec",
     "watt","ohm","ampere","farad","henry","tesla","gauss","weber","coulomb",
     "joule","newton","pascal_unit","kelvin","celsius","fahrenheit","rankine",
     "mole_unit","candela","lumen","lux","nit","apostilb","lambert","stilb",
     "phot","footcandle","footlambert",
     # Extra batch to reach 3000
     "tapestry_wall","stencil","relief_carving","intaglio","repoussé","chasing_metal",
     "granulation","mokume","damascene","niello","enamel_art2","cloisonne_art",
     "guilloche","engine_turning","knurling","checkering","stippling_art",
     "pointillism","divisionism","fauvism","cubism","surrealism","dadaism",
     "expressionism","impressionism","romanticism","realism","naturalism",
     "symbolism","art_nouveau","art_deco","bauhaus","minimalism","maximalism",
     "brutalism","futurism","constructivism","suprematism","neoplasticism",
     "abstract_expressionism","pop_art","op_art","kinetic_art","land_art",
     "installation","performance_art","conceptual_art","street_art","graffiti",
     "mural_art","stained_glass","rose_window","flying_buttress","ribbed_vault",
     "pointed_arch","barrel_vault","groin_vault","fan_vault","cloister_vault",
     "pendentive","squinch","apse","nave","transept","choir_arch","ambulatory",
     "clerestory_window","triforium","gallery_arch","narthex","atrium_arch",
     "crypt","sacristy","vestry","baptistery","bell_tower","campanile",
     "belfry","carillon","peal","chime","gong","cymbal","xylophone",
     "marimba","vibraphone","glockenspiel","celesta","harpsichord","clavichord",
     "organ_music","accordion","harmonica","bagpipe","didgeridoo","sitar",
     "tabla","djembe","bongo","conga","timpani","snare","tom_tom",
     "hi_hat","ride_cymbal","crash_cymbal","tambourine","maracas","castanets",
     "claves","guiro","cabasa","shaker","rain_stick","ocean_drum","thunder_sheet",
     "wind_chime","singing_bowl","tuning_bowl","crystal_bowl","handpan",
     "steel_drum","kalimba","mbira","zither","dulcimer","autoharp","banjo",
     "mandolin","ukulele","lute","theorbo","oud","bouzouki","balalaika",
     "shamisen","koto","erhu","pipa","guzheng","gamelan","angklung",
     "kulintang","suling","shakuhachi","taiko","wadaiko","odaiko",
     "shime_daiko","tsuzumi","biwa","kokyu","sho","hichiriki","ryuteki",
     "nokan","kagurabue","shinobue","fue","shamisen_music","koto_music",
     "gagaku","kabuki","noh","bunraku","kyogen","rakugo","manzai",
     "ukiyo_e","sumi_e","shodo","ikebana","bonsai_art","chado","kado",
     "kendo","judo","karate","aikido","sumo","kyudo","naginata",
     "iaido","kobudo","capoeira","savate","muay_thai","taekwondo",
     "kung_fu","tai_chi","qi_gong","wushu","silat","krav_maga",
     "systema","sambo","jiu_jitsu","hapkido","tang_soo_do","wing_chun",
     "mantis_style","crane_style","dragon_style","tiger_style","snake_style",
     "leopard_style","eagle_claw","monkey_style","drunken_style","praying_mantis",
     "white_crane","black_tiger","iron_palm","dim_mak","pressure_point",
     "meridian_body","chakra","aura","qi","prana","kundalini","nadis",
     "acupuncture","acupressure","reflexology","shiatsu","reiki","ayurveda",
     "yoga","meditation","mindfulness","breathwork","pranayama","asana",
     "mudra","mantra","yantra","mandala_art","thangka","sand_painting",
     "dreamcatcher","totem","fetish","talisman","amulet","charm","rune",
     "sigil","pentacle","ankh","eye_of_horus","hamsa","om_symbol",
     "yin_yang","triquetra","triskelion","celtic_knot","endless_knot",
     "flower_of_life","seed_of_life","tree_of_life","metatron","vesica",
     "fibonacci","golden_ratio","sacred_geometry","platonic_solid",
     "archimedean_solid","catalan_solid","johnson_solid","kepler_solid"
]
for w in extras:
    if len(words_data) >= 3000:
        break
    w = w.strip().lower()
    if w and w not in seen:
        seen.add(w)
        words_data.append({"word": w, "weight": 0.5})

words_data = words_data[:3000]

print(f"Final count: {len(words_data)}")
with open("data/vocab.json", "w") as f:
    json.dump(words_data, f)
print("Done! Saved to data/vocab.json")
