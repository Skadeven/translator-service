from src.translator import translate_content
from mock import patch
import ipytest
from sentence_transformers import SentenceTransformer, util
import copy
from typing import Callable

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_unexpected_language(mocker):
  # we mock the model's response to return a random message
  mocker.return_value.text = "I don't understand your request"

  # TODO assert the expected behavior
  assert translate_content("Aquí está su primer ejemplo.") == (True, "Aquí está su primer ejemplo.")
  assert translate_content("") == (True, "")
  assert translate_content("勝つさ") == (True, "勝つさ")
  assert translate_content("Bruh") == (True, "Bruh")

  mocker.return_value.text = "I'm so sorry, but I can't translate this text into English."
  assert translate_content("Aquí está su primer ejemplo.") == (True, "Aquí está su primer ejemplo.")
  assert translate_content("") == (True, "")
  assert translate_content("勝つさ") == (True, "勝つさ")
  assert translate_content("Bruh") == (True, "Bruh")

  mocker.return_value.text = "I don't know what you are talking about."
  assert translate_content("Aquí está su primer ejemplo.") == (True, "Aquí está su primer ejemplo.")
  assert translate_content("") == (True, "")
  assert translate_content("勝つさ") == (True, "勝つさ")
  assert translate_content("Bruh") == (True, "Bruh")

  mocker.return_value.text = "Please do not ask me to translate gibberish."
  assert translate_content("Aquí está su primer ejemplo.") == (True, "Aquí está su primer ejemplo.")
  assert translate_content("") == (True, "")
  assert translate_content("勝つさ") == (True, "勝つさ")
  assert translate_content("Bruh") == (True, "Bruh")

def cosine_similarity(str1, str2):
  model = SentenceTransformer("all-MiniLM-L6-v2")

  # Sentences are encoded by calling model.encode()
  emb1 = model.encode(str1)
  emb2 = model.encode(str2)

  return util.cos_sim(emb1, emb2)

def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
  a, b = expected_answer
  c, d = llm_response
  w = 0.5
  return w * (a == c) + (1.0-w)*cosine_similarity(b, d)

def evaluate(query_fn: Callable[[str], str], eval_fn: Callable[[str, str], float], dataset) -> float:
  res = 0.0
  for x in dataset:
    res += eval_fn(query_fn(x["post"]),x["expected_answer"])
  return res / len(dataset)

temp = [
    {
        "post": "Aquí está su primer ejemplo.",
        "expected_answer": (False, "This is your first example.")
    },
    {
        "post": "Aquí está su primer ejemplo.",
        "expected_answer": (False, "Here is your first example.")
    },
    {
        "post": """ハンバーガーまたは単にハンバーガーは、スライスしたバンズまたはロールパン
                  の中に詰めた具材（通常はひき肉、通常は牛肉のパティ）で構成される食品です。
                  ハンバーガーには、チーズ、レタス、トマト、タマネギ、ピクルス、ベーコン、
                  またはチリが添えられることがよくあります。ケチャップ、マスタード、
                  マヨネーズ、レリッシュ、またはサウザンドアイランドドレッシングの
                  バリエーションである「特製ソース」などの調味料。ゴマ饅頭の上に乗せられる
                  ことが多いです。ハンバーガーのパティにチーズをトッピングしたものを
                  チーズバーガーと呼びます。""",
        "expected_answer": (False, """A hamburger or simply burger is a food consisting
                            of fillings—usually a patty of ground meat,
                            typically beef—placed inside a sliced bun or bread roll.
                            Hamburgers are often served with cheese, lettuce,
                            tomato, onion, pickles, bacon, or chilis; condiments
                            such as ketchup, mustard, mayonnaise, relish, or a "special sauce",
                            often a variation of Thousand Island dressing; and are frequently
                            placed on sesame seed buns. A hamburger patty topped with cheese is
                            called a cheeseburger.""")
    },
    {
        "post": "في مدينة نيويورك، يشعر جون ويك بالحزن على وفاة زوجته هيلين، التي رتبت له أن يحصل على جرو بيغل لمساعدته في التغلب على خسارته.",
        "expected_answer": (False, "In New York City, John Wick is grieving the death of his wife Helen, who had arranged for him to receive a beagle puppy to help cope with his loss.")
    },
    {
        "post": """Twierdzenie o nieskończonej małpie stwierdza, że ​​małpa uderzająca losowo w klawisze maszyny do pisania przez nieskończoną ilość czasu prawie na pewno napisze dowolny tekst, łącznie z wszystkimi dziełami Williama Szekspira. W rzeczywistości małpa prawie na pewno wpisałaby każdy możliwy, skończony tekst nieskończoną liczbę razy. Twierdzenie można uogólnić, stwierdzając, że dowolna sekwencja zdarzeń, dla której prawdopodobieństwo wystąpienia jest niezerowe, prawie na pewno wystąpi nieskończoną liczbę razy, biorąc pod uwagę nieskończoną ilość czasu lub wszechświat o nieskończonych rozmiarach.
                   W tym kontekście „prawie na pewno” to termin matematyczny oznaczający, że zdarzenie ma miejsce z prawdopodobieństwem 1, a „małpa” nie jest prawdziwą małpą, ale metaforą abstrakcyjnego urządzenia, które wytwarza nieskończoną losową sekwencję liter i symboli. Warianty twierdzenia obejmują wielu, a nawet nieskończenie wielu maszynistek, a tekst docelowy różni się od całej biblioteki do pojedynczego zdania.""",
        "expected_answer": (False, """The infinite monkey theorem states that a monkey hitting keys at random on a typewriter keyboard for an infinite amount of time will almost surely type any given text, including the complete works of William Shakespeare. In fact, the monkey would almost surely type every possible finite text an infinite number of times. The theorem can be generalized to state that any sequence of events that has a non-zero probability of happening will almost certainly occur an infinite number of times, given an infinite amount of time or a universe that is infinite in size.
                              In this context, "almost surely" is a mathematical term meaning the event happens with probability 1, and the "monkey" is not an actual monkey, but a metaphor for an abstract device that produces an endless random sequence of letters and symbols. Variants of the theorem include multiple and even infinitely many typists, and the target text varies between an entire library and a single sentence.""")
    },
    {
        "post": "Заливные угри — традиционное английское блюдо, зародившееся в 18 веке, преимущественно в лондонском Ист-Энде. Блюдо состоит из нарезанных угрей, отварных в бульоне со специями, которому дают остыть и застыть, образуя желе. Обычно его подают холодным.",
        "expected_answer": (False, "Jellied eels are a traditional English dish that originated in the 18th century, primarily in the East End of London. The dish consists of chopped eels boiled in a spiced stock that is allowed to cool and set, forming a jelly. It is usually served cold.")
    },
    {
        "post": """축구 전쟁(스페인어: La guerra del fútbol)은 축구 전쟁 또는 백시간 전쟁으로도 알려져 있으며, 1969년 엘살바도르와 온두라스 사이에 벌어진 짧은 군사적 충돌입니다. 1970년 FIFA 월드컵 예선. 1969년 7월 14일 엘살바도르 군대가 온두라스를 공격하면서 전쟁이 시작됐다. 미주기구(OAS)는 7월 18일 밤에 휴전 협상을 했고(따라서 "100시간 전쟁"), 이는 7월 20일에 발효되었습니다. 엘살바도르 군대는 8월 초에 철수했다. 전쟁은 짧았지만 양국 모두에게 큰 결과를 가져왔고 10년 후 엘살바도르 내전이 시작되는 주요 요인이 되었습니다.""",
        "expected_answer": (False,"""The Football War (Spanish: La guerra del fútbol), also known as the Soccer War or the Hundred Hours' War, was a brief military conflict fought between El Salvador and Honduras in 1969. Existing tensions between the two countries coincided with rioting during a 1970 FIFA World Cup qualifier. The war began on 14 July 1969 when the Salvadoran military launched an attack against Honduras. The Organization of American States (OAS) negotiated a cease-fire on the night of 18 July (hence "100 Hour War"), which took full effect on 20 July. Salvadoran troops were withdrawn in early August. The war, while brief, had major consequences for both countries and was a major factor in starting the Salvadoran Civil War a decade later.""")
    },
    {
        "post": "Tarjan 的强连通分量算法是图论中的一种算法，用于查找有向图的强连通分量 (SCC)。它以线性时间运行，与 Kosaraju 算法和基于路径的强分量算法等替代方法的时间限制相匹配。",
        "expected_answer": (False, "Tarjan's strongly connected components algorithm is an algorithm in graph theory for finding the strongly connected components (SCCs) of a directed graph. It runs in linear time, matching the time bound for alternative methods including Kosaraju's algorithm and the path-based strong component algorithm.")
    },
    {
        "post": "Casablanca ist ein US-amerikanisches romantisches Drama aus dem Jahr 1942 von Michael Curtiz mit Humphrey Bogart, Ingrid Bergman und Paul Henreid in den Hauptrollen. Gefilmt und angesiedelt während des Zweiten Weltkriegs, dreht sich der Film um einen amerikanischen Expatriate (Bogart), der sich zwischen seiner Liebe zu einer Frau (Bergman) und der Unterstützung ihres Mannes (Henreid), einem tschechoslowakischen Widerstandsführer, bei der Flucht aus der von Vichy kontrollierten Stadt entscheiden muss Casablanca setzt seinen Kampf gegen die Deutschen fort.",
        "expected_answer": (False, "Casablanca is a 1942 American romantic drama film directed by Michael Curtiz and starring Humphrey Bogart, Ingrid Bergman, and Paul Henreid. Filmed and set during World War II, it focuses on an American expatriate (Bogart) who must choose between his love for a woman (Bergman) and helping her husband (Henreid), a Czechoslovak resistance leader, escape from the Vichy-controlled city of Casablanca to continue his fight against the Germans.")
    },
    {
        "post": """कार्नेगी मेलॉन यूनिवर्सिटी (सीएमयू) पिट्सबर्ग, पेंसिल्वेनिया में एक निजी शोध विश्वविद्यालय है। संस्था की स्थापना मूल रूप से 1900 में एंड्रयू कार्नेगी द्वारा कार्नेगी टेक्निकल स्कूल के रूप में की गई थी। 1912 में, यह कार्नेगी इंस्टीट्यूट ऑफ टेक्नोलॉजी बन गया और चार साल की डिग्री देना शुरू कर दिया। 1967 में, यह मेलन इंस्टीट्यूट ऑफ इंडस्ट्रियल रिसर्च के साथ विलय के माध्यम से वर्तमान कार्नेगी मेलन विश्वविद्यालय बन गया, जिसकी स्थापना 1913 में एंड्रयू मेलन और रिचर्ड बी मेलन ने की थी और यह पहले पिट्सबर्ग विश्वविद्यालय का एक हिस्सा था।
                  विश्वविद्यालय में सात कॉलेज शामिल हैं, जिनमें इंजीनियरिंग कॉलेज, कंप्यूटर साइंस स्कूल और टेपर स्कूल ऑफ बिजनेस शामिल हैं। विश्वविद्यालय का मुख्य परिसर डाउनटाउन पिट्सबर्ग से 5 मील (8 किमी) दूर स्थित है। इसके छह महाद्वीपों में एक दर्जन से अधिक डिग्री देने वाले स्थान हैं, जिनमें कतर, सिलिकॉन वैली और किगाली, रवांडा (कार्नेगी मेलन विश्वविद्यालय अफ्रीका) के परिसर और राष्ट्रीय और वैश्विक स्तर पर विश्वविद्यालयों के साथ साझेदारी शामिल है। कार्नेगी मेलन ने अपने कई परिसरों में 15,818 छात्रों का नामांकन किया है। 117 देश और 1,400 से अधिक संकाय सदस्य कार्यरत हैं।
                  कार्नेगी मेलन अनुसंधान और अध्ययन के नए क्षेत्रों में अपनी प्रगति के लिए जाना जाता है, विशेष रूप से कंप्यूटर विज्ञान (पहले कंप्यूटर विज्ञान, मशीन लर्निंग और रोबोटिक्स विभागों सहित) में कई प्रथम पहलों का घर होने के नाते, प्रबंधन विज्ञान के क्षेत्र में अग्रणी होने के लिए, और संयुक्त राज्य अमेरिका में पहला नाटक कार्यक्रम। कार्नेगी मेलॉन एसोसिएशन ऑफ अमेरिकन यूनिवर्सिटीज़ के सदस्य हैं और उन्हें "आर1: डॉक्टोरल यूनिवर्सिटीज़ - बहुत उच्च अनुसंधान गतिविधि" में वर्गीकृत किया गया है।
                  कार्नेगी मेलन यूनिवर्सिटी एथलेटिक एसोसिएशन के संस्थापक सदस्य के रूप में एनसीएए डिवीजन III एथलेटिक्स में प्रतिस्पर्धा करते हैं। कार्नेगी मेलन ने टार्टन्स के रूप में आठ पुरुषों की टीमों और नौ महिलाओं की टीमों को मैदान में उतारा। विश्वविद्यालय के संकाय और पूर्व छात्रों में 20 नोबेल पुरस्कार विजेता और 13 ट्यूरिंग पुरस्कार विजेता शामिल हैं और उन्हें 142 एमी पुरस्कार, 52 टोनी पुरस्कार और 13 अकादमी पुरस्कार प्राप्त हुए हैं।""",
        "expected_answer": (False, """Carnegie Mellon University (CMU) is a private research university in Pittsburgh, Pennsylvania. The institution was originally established in 1900 by Andrew Carnegie as the Carnegie Technical Schools. In 1912, it became the Carnegie Institute of Technology and began granting four-year degrees. In 1967, it became the current-day Carnegie Mellon University through its merger with the Mellon Institute of Industrial Research, founded in 1913 by Andrew Mellon and Richard B. Mellon and formerly a part of the University of Pittsburgh.
                              The university consists of seven colleges, including the College of Engineering, the School of Computer Science, and the Tepper School of Business. The university has its main campus located 5 miles (8 km) from Downtown Pittsburgh. It also has over a dozen degree-granting locations in six continents, including campuses in Qatar, Silicon Valley, and Kigali, Rwanda (Carnegie Mellon University Africa) and partnerships with universities nationally and globally.Carnegie Mellon enrolls 15,818 students across its multiple campuses from 117 countries and employs more than 1,400 faculty members.
                              Carnegie Mellon is known for its advances in research and new fields of study, notably being home to many firsts in computer science (including the first computer science, machine learning, and robotics departments), pioneering the field of management science, and being home to the first drama program in the United States. Carnegie Mellon is a member of the Association of American Universities and is classified among "R1: Doctoral Universities – Very High Research Activity".
                              Carnegie Mellon competes in NCAA Division III athletics as a founding member of the University Athletic Association. Carnegie Mellon fields eight men's teams and nine women's teams as the Tartans. The university's faculty and alumni include 20 Nobel Prize laureates and 13 Turing Award winners and have received 142 Emmy Awards, 52 Tony Awards, and 13 Academy Awards. """)
    },
    {
        "post": "Robert Norman Ross (Oktoba 29, 1942 - 4 Julai 1995) alikuwa mchoraji na mkufunzi wa sanaa wa Kimarekani ambaye aliunda na kuandaa kipindi cha The Joy of Painting, kipindi cha mafunzo cha televisheni kilichorushwa hewani kuanzia 1983 hadi 1994 kwenye PBS nchini Marekani, CBC nchini Marekani. Kanada, na njia kama hizo huko Amerika Kusini, Ulaya na kwingineko.",
        "expected_answer": (False, "Robert Norman Ross (October 29, 1942 – July 4, 1995) was an American painter and art instructor who created and hosted the The Joy of Painting, an instructional television program that aired from 1983 to 1994 on PBS in the United States, CBC in Canada, and similar channels in Latin America, Europe and elsewhere.")
    },
    {
        "post": """Ann an leasachadh bathar-bog, tha cleachdaidhean sùbailte (uaireannan sgrìobhte “Agile”) a’ toirt a-steach riatanasan, lorg agus leasachadh fhuasglaidhean tro oidhirp cho-obrachail sgiobaidhean fèin-eagrachaidh agus tar-ghnìomhach leis an neach-ceannach / luchd-cleachdaidh / luchd-cleachdaidh deireannach.""",
        "expected_answer": (False, """In software development, agile practices (sometimes written "Agile") include requirements, discovery and solutions improvement through the collaborative effort of self-organizing and cross-functional teams with their customer(s)/end user(s).""")
    },
    {
        "post": "In mathematical analysis, Lipschitz continuity, named after German mathematician Rudolf Lipschitz, is a strong form of uniform continuity for functions. Intuitively, a Lipschitz continuous function is limited in how fast it can change: there exists a real number such that, for every pair of points on the graph of this function, the absolute value of the slope of the line connecting them is not greater than this real number; the smallest such bound is called the Lipschitz constant of the function (and is related to the modulus of uniform continuity). For instance, every function that is defined on an interval and has bounded first derivative is Lipschitz continuous.",
        "expected_answer": (False, "In analisi matematica, la continuità di Lipschitz, dal nome del matematico tedesco Rudolf Lipschitz, è una forma forte di continuità uniforme per le funzioni. Intuitivamente, una funzione continua di Lipschitz è limitata nella velocità con cui può cambiare: esiste un numero reale tale che, per ogni coppia di punti sul grafico di questa funzione, il valore assoluto della pendenza della linea che li collega non è maggiore di questo numero reale; il più piccolo limite di questo tipo è chiamato costante di Lipschitz della funzione (ed è correlato al modulo di continuità uniforme). Ad esempio, ogni funzione definita su un intervallo e con derivata prima limitata è continua Lipschitz.")
    },
    {
        "post": "சூப்பர்மேன் என்பது DC காமிக்ஸ் வெளியிட்ட அமெரிக்க காமிக் புத்தகங்களில் தோன்றும் ஒரு சூப்பர் ஹீரோ.",
        "expected_answer": (False, "Superman is a superhero who appears in American comic books published by DC Comics.")
    },
    {
        "post": "知らんけど（しらんけど）は、日本の近畿方言（関西弁）で使われる言葉の一つ。",
        "expected_answer": (False, """"I don't know" is a word used in Japan's Kinki dialect (Kansai dialect).""")
    },
    {
        "post": "勝つさ",
        "expected_answer": (False, "I'll win")
    }
]

complete_eval_set = copy.copy(temp)
for i in temp:
  a, b = i["expected_answer"]
  test = {
    "post": b,
    "expected_answer": (True, b)
  }
  complete_eval_set += [test]

  complete_eval_set += [
    {
        "post": "hbjagdfbhjakjsdfkhjafdjhkshkjfsad",
        "expected_answer": (True, "hbjagdfbhjakjsdfkhjafdjhkshkjfsad")
    },
    {
        "post": "Ȋ̴͈̭͗̏͋͝͠'̸̡̨̢̣̫̖̊̊́͆͊͑͜ḑ̶̼͇̣̜̼̲̊́̚͘ͅ ̵̬̺̍͂ͅw̷͇̗̰̭͕̿̋į̵̰̦̤̹̲͕̿͜ṇ̸̈́͗͋͝",
        "expected_answer": (True, "Ȋ̴͈̭͗̏͋͝͠'̸̡̨̢̣̫̖̊̊́͆͊͑͜ḑ̶̼͇̣̜̼̲̊́̚͘ͅ ̵̬̺̍͂ͅw̷͇̗̰̭͕̿̋į̵̰̦̤̹̲͕̿͜ṇ̸̈́͗͋͝")
    },
    {
        "post": """o:
                    푋 ∼
                    
                    Exp ( 푝 푝
                    0 w/prob 1 − 푝""",
        "expected_answer": (True, """o:
                                  푋 ∼
                                  
                                  Exp ( 푝 푝
                                  0 w/prob 1 − 푝""")
    },
    {
        "post": " ",
        "expected_answer": (True, " ")
    },
    {
        "post": """˙›Õ¸çà_ˇ∏h∂‚j›\ˇ¸~ˆı◊‚€77‚˜Àã8ä·_ùH""",
        "expected_answer": (True, """˙›Õ¸çà_ˇ∏h∂‚j›\ˇ¸~ˆı◊‚€77‚˜Àã8ä·_ùH""")
    }
]
  
def test_llm_response():
    eval_score = evaluate(translate_content, eval_single_response_complete, complete_eval_set)

    print(f"Evaluation Score: {eval_score}")

test_llm_response()