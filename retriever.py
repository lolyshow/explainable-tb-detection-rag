def load_corpora():
    def read_file(path):
        try:
            return open(path, "r", encoding="utf-8").read()
        except:
            return open(path, "r", encoding="latin-1").read()

    who_text = read_file("9789241511506-eng.txt")
    statpearls_text = read_file("statpearls_corpus.txt")

    return who_text, statpearls_text


def retrieve_explanation(pred, who_text, statpearls_text):
    keyword = "tuberculosis" if pred == 1 else "normal"

    combined = who_text + "\n" + statpearls_text
    sentences = combined.split(".")

    relevant = [s for s in sentences if keyword in s.lower()]

    if len(relevant) == 0:
        return "No strong clinical reference found."

    return ". ".join(relevant[:5])