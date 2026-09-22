"""End-to-end pipeline test with a deterministic stub embedder (no model weights needed).

Exercises: HTML extraction (cp1250 fixture), chunking, index build, dense + BM25 + RRF
retrieval, mask filtering and neighbour de-dup. Run: `uv run pytest -q`.
"""

from __future__ import annotations

import hashlib
import itertools
import re

import numpy as np
import pytest

from stavby_rag import config, embed
from stavby_rag.chunk import chunk_pages
from stavby_rag.cli import EvalScore
from stavby_rag.crawl import Page, page_from_html
from stavby_rag.llm import system_prompt
from stavby_rag.pdf import PdfBook
from stavby_rag.retrieve import Hit, RetrievalConfig, Retriever, guess_lang
from stavby_rag.store import Store, tokenize

DIM = 64


def _stub_vec(text: str) -> np.ndarray:
    """Bag-of-hashed-words embedding: deterministic, and similar texts land close together."""
    v = np.zeros(DIM, dtype=np.float32)
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    return v / (np.linalg.norm(v) + 1e-9)


class _StubEmbedder:
    key = "stub:hash"

    def embed_texts(self, texts, **kw):
        return np.stack([_stub_vec(t) for t in texts])

    def embed_query(self, q):
        return _stub_vec(q)


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    monkeypatch.setattr(embed, "get_embedder", lambda *a, **k: _StubEmbedder())
    monkeypatch.setattr(embed, "get_reranker", lambda *a, **k: None)


FIXTURE_HTML = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<meta http-equiv="Content-Type" content="text/html; charset=windows-1250">
<html><head><title>Kapitola 4.1</title></head>
<body title="4.1 Stupně rozestavěnosti">
<p><h3>4.1 Stupně rozestavěnosti a technologické etapy</h3></p>
<p>Stavební objekt je technickým systémem. Technologická etapa je souhrn procesů.</p>
<p>Zemní práce, základy, hrubá spodní stavba, hrubá vrchní stavba, zastřešení.</p>
<IMG SRC="..\\sipka_vpred.GIF"><img src="obr41.jpg">
<a href="text42.html">další</a>
</body></html>"""


def _page(url: str, html: str) -> Page:
    p = page_from_html(url, config.RAW_DIR / "x.html", html)
    assert p is not None
    return p


def test_extract_metadata_and_text():
    p = _page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/text41.html", FIXTURE_HTML)
    assert (p.lang, p.edition, p.chapter, p.section) == ("cs", "full", 4, "4.1")
    assert p.title.startswith("4.1 Stupn")
    assert "Technologická etapa" in p.text
    assert p.images == ["https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/obr41.jpg"]  # arrows dropped
    assert p.links == ["https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/text42.html"]


def test_frameset_and_chrome_are_skipped():
    fs = "<html><frameset rows='8%,92%'><frame src='a.html'></frameset></html>"
    local = config.RAW_DIR / "x.html"
    assert page_from_html("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/frame4.html", local, fs) is None
    assert page_from_html("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/nadpis4.html", local, "<p>" + "x" * 500) is None
    for nav in ("obsah4.html", "obsah4small.html", "obrazky4.html", "nahledy10.html", "foto4ii.html"):
        assert page_from_html(f"https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/{nav}", local, "<p>" + "x" * 500) is None


def test_demo_edition_chapter_from_filename():
    p = _page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava-en-demo/text132.html",
              FIXTURE_HTML.replace("4.1", "13.2"))
    assert (p.lang, p.edition, p.chapter, p.section) == ("en", "demo", 13, "13.2")


def test_tokenizer_folds_and_stems():
    toks = tokenize("Rozestavěnosti ROZESTAVĚNOST 20250101 etapy")
    assert "rozestavenosti" in toks and "rozest" in toks and "rozestavenost" in toks
    assert "20250101" not in toks


def _build_store(tmp_path) -> Store:
    long = " ".join(f"Věta číslo {i} o harmonogramu výstavby a síťové analýze." for i in range(120))
    pages = [
        _page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/text41.html", FIXTURE_HTML),
        _page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap9/text91.html",
              FIXTURE_HTML.replace("4.1", "9.1").replace("Stavební objekt je technickým systémem.", long)),
        _page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava-en/kap4/text41.html",
              "<html><body title='4.1 Degrees of completion'><p>" + "Construction stages and technological phases of a building. " * 10 + "</p></body></html>"),
    ]
    chunks = chunk_pages(pages)
    assert len(chunks) >= 4  # long page split into several overlapping chunks
    assert all(c.text.startswith(("Učebnice", "Textbook")) for c in chunks)
    store = Store(tmp_path / "index")
    emb = embed.get_embedder()
    store.build(chunks, emb.embed_texts([c.text for c in chunks]), emb.key)
    loaded = Store(tmp_path / "index").load()
    assert loaded.meta["embed_backend"] == "stub" and loaded.meta["embed_model"] == "hash"
    return loaded


def test_hybrid_retrieval_and_filters(tmp_path):
    store = _build_store(tmp_path)
    cfg = RetrievalConfig(top_k=3, lang="cs", editions=("full",))
    hits = Retriever(store, cfg).search("technologické etapy rozestavěnosti")
    assert hits and hits[0].chunk.section == "4.1" and hits[0].chunk.lang == "cs"
    assert all(h.chunk.lang == "cs" for h in hits)
    assert hits[0].bm25_rank is not None  # lexical match contributed

    en = Retriever(store, RetrievalConfig(top_k=3, lang="en")).search("technological phases")
    assert en and en[0].chunk.lang == "en"

    # neighbouring overlapping chunks from the long page are de-duplicated
    long_hits = Retriever(store, RetrievalConfig(top_k=6, lang="cs")).search("harmonogram síťová analýza")
    ords = sorted(h.chunk.ordinal for h in long_hits if h.chunk.section == "9.1")
    assert all(b - a > 1 for a, b in itertools.pairwise(ords))


def test_chunk_ids_stable(tmp_path):
    p = _page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/text41.html", FIXTURE_HTML)
    a, b = chunk_pages([p]), chunk_pages([p])
    assert [c.id for c in a] == [c.id for c in b] and re.fullmatch(r"[0-9a-f]{16}", a[0].id)


def test_guess_lang_without_diacritics():
    assert guess_lang("Co jsou stupne rozestavenosti?") == "cs"
    assert guess_lang("Jaké jsou technologické etapy?") == "cs"
    assert guess_lang("What are the technological stages of a building?") == "en"
    assert guess_lang("CPM critical path") == "en"


def test_config_for_question():
    cfg = RetrievalConfig.for_question("Co je harmonogram?", top_k=3)
    assert (cfg.top_k, cfg.lang, cfg.editions, cfg.rerank) == (3, "cs", ("full",), None)
    assert RetrievalConfig.for_question("what is a schedule?", lang="cs", edition="all").editions == ("full", "demo")
    assert RetrievalConfig.for_question("what is a schedule?", lang="cs").lang == "cs"  # explicit lang wins


def test_reranker_score_aggregates_case_variants():
    from stavby_rag.embed import OllamaQwenReranker as R

    relevant = [{"token": "yes", "logprob": -0.1}, {"token": "Yes", "logprob": -3.4}, {"token": "no", "logprob": -6.2}, {"token": "YES", "logprob": -5.9}]
    weaker = [{"token": "yes", "logprob": -0.15}, {"token": "no", "logprob": -3.5}, {"token": "Yes", "logprob": -3.6}]
    irrelevant = [{"token": "no", "logprob": -0.05}, {"token": "yes", "logprob": -4.0}]
    s_rel, s_weak, s_irr = (R.score_from_top_logprobs(x) for x in (relevant, weaker, irrelevant))
    assert s_rel > s_weak > s_irr
    assert s_rel > 0.99 and s_irr < 0.05
    assert R.score_from_top_logprobs([{"token": "maybe", "logprob": -0.1}]) == 0.5  # neither token seen


def _hit(chapter: int, section: str) -> Hit:
    chunks = chunk_pages([_page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/text41.html", FIXTURE_HTML)])
    c = chunks[0]
    c.chapter, c.section = chapter, section
    return Hit(c, 0.0, None, None)


def test_eval_score_counts_chapters_and_sections():
    score = EvalScore(k=10)
    expected, got, ok = score.record({"chapter": 4, "sections": ["4.1"]}, [_hit(4, "4.1"), _hit(9, "9.1")])
    assert (expected, ok) == ("4 §4.1", True) and got.startswith("4/4.1 ")
    score.record({"chapter": 13, "sections": ["13.3"]}, [_hit(4, "4.1"), _hit(13, "13.3")])  # right chapter at rank 2
    assert score.summary() == [
        "chapter: hit@1 = 1/2 (50%)   hit@10 = 2/2 (100%)",
        "section: hit@1 = 1/2 (50%)   hit@3 = 2/2 (100%)",
    ]


# --------------------------------------------------------------------- PDF books


def _write_pdf(path) -> str:
    """A 4-page Czech PDF with a two-chapter outline; page 3 has a hard-wrapped word."""
    import pymupdf as fitz

    bodies = [
        "Prvni kapitola popisuje planovani stavby. Je to zaklad pripravy vystavby a rizeni prace na stavbe.",
        "Pokracovani prvni kapitoly: harmonogram, kritická cesta a sitová analýza jsou nástroje planovani.",
        "Druhá kapitola: technologie výstavby. Kazdá sta-\nvba má technologické etapy a stupne rozestavenosti.",
        "Základy jsou nosnou konstrukcí. Pro zakladani je nutné znát geologické podmínky na staveniti.",
    ]
    with fitz.open() as doc:
        for body in bodies:
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(40, 40, 550, 750), body + " " + "Text pokracuje dále. " * 4,
                                fontsize=11, fontname="helv")
        doc.set_toc([[1, "1 Úvod", 1], [1, "2 Technologie výstavby", 3], [2, "2.1 Základy", 4]])
        doc.save(str(path))
    return str(path)


def test_pdf_book_pages(tmp_path):
    pdf = _write_pdf(tmp_path / "kniha.pdf")
    pages = PdfBook(pdf).pages(log=lambda _m: None)

    assert len(pages) == 4
    assert {p.lang for p in pages} == {"cs"}
    assert {p.edition_key for p in pages} == {"pdf"} and {p.edition for p in pages} == {"full"}
    assert {p.book for p in pages} == {"kniha"}  # no --title, no PDF metadata title -> file stem
    assert [p.path for p in pages] == ["kniha.pdf"] * 4

    assert [(p.chapter, p.section) for p in pages] == [(1, None), (1, None), (2, None), (2, "2.1")]
    assert [p.title for p in pages] == ["1 Úvod", "1 Úvod", "2 Technologie výstavby", "2.1 Základy"]
    assert pages[3].url == f"file://{pdf}#page=4"

    assert "stavba" in pages[2].text and "sta- vba" not in pages[2].text  # de-hyphenated
    assert "\n" not in pages[0].text  # hard-wrapped lines joined


def test_pdf_book_explicit_title_and_lang(tmp_path):
    pages = PdfBook(_write_pdf(tmp_path / "kniha.pdf"), title="Priprava staveb", lang="en").pages(log=lambda _m: None)
    assert {(p.book, p.lang) for p in pages} == {("Priprava staveb", "en")}


def test_pdf_chunks_and_index_record_books(tmp_path):
    pages = PdfBook(_write_pdf(tmp_path / "kniha.pdf"), title="Priprava staveb").pages(log=lambda _m: None)
    chunks = chunk_pages(pages)
    assert chunks and all(c.text.startswith("Priprava staveb > ") for c in chunks)  # book in the breadcrumb

    emb = embed.get_embedder()
    store = Store(tmp_path / "index")
    store.build(chunks, emb.embed_texts([c.text for c in chunks]), emb.key, books=[p.book for p in pages])
    loaded = Store(tmp_path / "index").load()
    assert loaded.books == ["Priprava staveb"]
    assert "the following source(s): Priprava staveb" in system_prompt(loaded.books)

    hits = Retriever(loaded, RetrievalConfig(top_k=3, lang="cs")).search("kritická cesta a harmonogram")
    assert hits and hits[0].chunk.chapter == 1


def test_crawled_pages_keep_the_textbook_breadcrumb():
    """`book` defaults to "" so existing pages.jsonl chunks (and their ids) stay byte-identical."""
    p = _page("https://technologie.fsv.cvut.cz/aitom/podklady/online-priprava/kap4/text41.html", FIXTURE_HTML)
    assert p.book == ""
    chunk = chunk_pages([p])[0]
    assert chunk.text.startswith("Učebnice: Příprava a realizace staveb > Kapitola 4 > 4.1 Stupně rozestavěnosti\n\n")
    assert system_prompt().startswith("You are a study assistant for the CTU (ČVUT)")


def test_corpus_paths():
    assert config.Corpus().pages_file == config.DATA_DIR / "pages.jsonl"
    assert config.Corpus().index_dir == config.DATA_DIR / "index"
    assert config.Corpus("demo").pages_file == config.DATA_DIR / "corpora" / "demo" / "pages.jsonl"
    assert config.Corpus("demo").index_dir == config.DATA_DIR / "corpora" / "demo" / "index"
    assert (config.Corpus().label, config.Corpus("demo").label) == ("default", "demo")


def test_pdf_broken_ratio_and_headings():
    from stavby_rag.pdf import broken_ratio, headings_in

    assert broken_ratio("p\x00edepsané zkou⌃ky") > 0.05
    assert broken_ratio("předepsané zkoušky") == 0.0
    text = "3 Výrobní proces stavby a objektu\n\nÚvodní odstavec.\n\n3.1 Základní pojmy - prvky výrobního procesu\n\nText končí tečkou 3.2 ne.\n"
    assert headings_in(text) == [("3", "3 Výrobní proces stavby a objektu"), ("3.1", "3.1 Základní pojmy - prvky výrobního procesu")]


def test_pdf_reading_order_two_pages_spread_and_paragraphs():
    from stavby_rag.pdf import Line, reading_order

    def col(x0, texts, indents=()):
        return [Line(t, x0 + (0.02 if i in indents else 0.0), 0.10 + i * 0.03, x0 + 0.40, 0.12 + i * 0.03) for i, t in enumerate(texts)]

    left = col(0.05, ["L1 first", "L2 second", "L3 new para"], indents={2})
    right = col(0.55, ["R1 heading", "R2 body"], indents={1})
    right.append(Line("R3 after gap", 0.55, 0.40, 0.95, 0.42))
    text = reading_order(right + left)  # order of input is irrelevant
    assert text == "L1 first L2 second\n\nL3 new para\n\nR1 heading\n\nR2 body\n\nR3 after gap"
    assert reading_order([]) == ""
    single = col(0.1, ["a", "b"])
    single = [Line(ln.text, ln.x0, ln.y0, 0.9, ln.y1) for ln in single]  # full-width lines: one column
    assert reading_order(single) == "a b"
    hyph = [Line("Není-", 0.1, 0.1, 0.5, 0.12), Line("li objednatel", 0.1, 0.13, 0.5, 0.15)]
    assert reading_order(hyph) == "Není-li objednatel"
