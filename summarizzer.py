import spacy
from collections import Counter
from heapq import nlargest

nlp = spacy.load("en_core_web_sm")

some_text = """
There is, however, one profound difference between AI computer-based creativity versus other machine-based image making technologies. Photography, and the similar media of film and video, are predicated on a reference to something outside of the machine, something in the natural world. They are technologies to capture elements of the world outside themselves as natural light on a plate or film, fixed with a chemical process to freeze light patterns in time and space. Computational imagery has no such referent in nature or to anything outside of itself. This is a profound difference that we believe should be given more attention. The lack of reference in nature has historical implications for how we understand something as art. Almost all human art creation has been inspired by something seen in the natural world. There, of course, may be many steps between the inspiration and the resulting work, such that the visual referent can be changed, abstracted or even erased by the final version. However, the process was always first instigated by the artist looking at something in the world, and photography, film, and video retained that first step of the art-making process through light encoding. The computer does not follow this primal pattern. It requires absolutely nothing from the natural world; instead, its “brain” and “eyes” (its internal apparatus for encoding imagery of any kind) consist only of receptors for numerical data. There are two preliminary points to elaborate here: The first relates to issues in contemporary art, the second to the distinction from human creativity.
First, the lack of referent in the natural world and the resulting freedom and range to create or not create any object as a result of the artistic inspiration aligns AI and all computational methods with conceptual art. Like with photography, the comparison with conceptual art has frequently been made for AI and computational methods in general. In conceptual art, the act of the creation of the art work is located in the mind of the artist, and its instantiation in any material form(s) in the world is, as Sol Lewitt (Lewitt 1967) famously declared, “a perfunctory affair. The idea becomes a machine that makes the art.” Thus, the making of an art object becomes simply optional. And although contemporary artists in the main have not stopped making objects, the principle that object making is optional and variable in relation to the art concept still remains. We believe this is at the heart of the usefulness of the comparison with conceptual art: The idea or concept is untethered from nature, being primarily located in the synapses of the brain and secondly disassociated from the dictates of the material world. Most AI systems use some form of a neural network, which is modeled on the neural complexity of the human brain. Therefore, AI and conceptual art coincide in locating the art act in the system network of the brain, rather than in the physical output. The physical act of an artist, either applying paint or carving marble, becomes optional. This removes the necessity of a human body (the artist) to make things and allows us to imagine that there could be more than one kind of artist, including other than human.
"""

doc = nlp(some_text)

# Capturing words that match the part of speech
pos_tag = ["PROPN", "ADJ", "NOUN", "VERB"]

capture_words = []
for token in doc:
    if token.pos_ in pos_tag:
        capture_words.append(token.text)

# Normalizing the data
max_frequency = Counter(capture_words).most_common(1)[0][1]

freq_counter = Counter(capture_words)
for key_word in freq_counter.keys():
    freq_counter[key_word] = freq_counter[key_word] / max_frequency

top_words = freq_counter.most_common(10)
print(top_words)

# Traversing the input text to identify sentences with the highest word frequencies score
sents_o_inter = {}
for sentence in doc.sents:
    for word in sentence:
        if word.text in freq_counter:
            if sentence in sents_o_inter:
                sents_o_inter[sentence] += freq_counter[word.text]
            else:
                sents_o_inter[sentence] = freq_counter[word.text]

print(sents_o_inter)

# Summarize the sentences with a max heap, ordering is based on sum values associated with each sentence
summarized_sentences = nlargest(6, sents_o_inter, key=sents_o_inter.get)

summarized_sentences = [span.text for span in summarized_sentences]

final_summary = ""
for i in summarized_sentences:
    final_summary += i + " "

print("=====================")
print(final_summary)
