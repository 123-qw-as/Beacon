from math_agent.rag.ingest import ingest_directory, _extract_pdf_text, _normalize_math_text


def test_ingest_directory_processes_md_files(mocker, workdir):
    corpus = workdir / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("段落一\n\n段落二", encoding="utf-8")
    (corpus / "b.md").write_text("内容 b", encoding="utf-8")

    captured = {"count": 0}

    def _fake_embed(texts, *, model, batch_size=64):
        captured["count"] += len(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    mocker.patch("math_agent.rag.ingest.embed_texts", side_effect=_fake_embed)

    db = workdir / "vec.db"
    report = ingest_directory(
        src_dir=corpus, db_path=db,
        embedding_model="m", dim=3,
        max_chars=200, overlap=20,
    )
    assert report.files_processed == 2
    assert report.chunks_added >= 2
    assert db.exists()
    assert captured["count"] == report.chunks_added


def test_ingest_directory_handles_pdf_via_pypdf(mocker, workdir):
    """PDF 文件用 pypdf 抽文本；这里只确认调用路径不抛。"""
    corpus = workdir / "corpus"
    corpus.mkdir()
    fake_pdf = corpus / "doc.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    mocker.patch("math_agent.rag.ingest._extract_pdf_text", return_value="pdf 内容")
    mocker.patch("math_agent.rag.ingest.embed_texts",
                 side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    rep = ingest_directory(src_dir=corpus, db_path=workdir / "v.db",
                           embedding_model="m", dim=3, max_chars=200, overlap=20)
    assert rep.files_processed == 1


def test_ingest_directory_skips_unreadable_files(mocker, workdir):
    corpus = workdir / "corpus"
    corpus.mkdir()
    bad = corpus / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    mocker.patch("math_agent.rag.ingest._extract_pdf_text",
                  side_effect=RuntimeError("corrupt"))
    mocker.patch("math_agent.rag.ingest.embed_texts",
                  side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    rep = ingest_directory(src_dir=corpus, db_path=workdir / "v.db",
                           embedding_model="m", dim=3, max_chars=200, overlap=20)
    assert rep.files_processed == 0
    assert len(rep.skipped) == 1


def test_extract_pdf_text_uses_pymupdf_no_box_chars(tmp_path):
    """Regression: pypdf produced box chars (□) for math symbols in CID-encoded
    fonts. pymupdf (fitz) decodes them correctly. This test creates a real PDF
    with math text via fitz and verifies _extract_pdf_text returns clean text."""
    import fitz

    pdf_path = tmp_path / "math_test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 插入含数学符号的文本
    math_text = "Integral: \u222bx dx  Sum: \u2211xi  Infinity: \u221e  Square: \u221ax"
    page.insert_text((72, 72), math_text, fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    result = _extract_pdf_text(pdf_path)
    # 不应包含方框字符
    assert "\u25a1" not in result
    assert "\ufffd" not in result


def test_extract_pdf_text_reconstructs_superscripts(tmp_path):
    """Regression: 2² was extracted as '22' (ambiguous). Now font-size analysis
    detects superscripts and wraps them as ^{...}, so 2² becomes '2^{2}'."""
    import fitz

    pdf_path = tmp_path / "sup_test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 用 insert_text 模拟上标：基线 size=12，上标 size=7 y 偏移小
    page.insert_text((72, 72), "2", fontsize=12)
    page.insert_text((80, 69), "2", fontsize=7)   # 上标，y 仅偏移 3
    page.insert_text((88, 72), " = 4", fontsize=12)
    page.insert_text((72, 100), "3", fontsize=12)
    page.insert_text((80, 97), "3", fontsize=7)   # 上标
    page.insert_text((88, 100), " = 27", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    result = _extract_pdf_text(pdf_path)
    # 2² 应还原为 2^{2}，而不是 22
    assert "2^{2}" in result
    # 3³ 应还原为 3^{3}，而不是 33
    assert "3^{3}" in result
    # 不应出现裸的 22= 或 33=
    assert "22=" not in result.replace(" ", "")
    assert "33=" not in result.replace(" ", "")


def test_normalize_math_text_fixes_neq_and_mu():
    """Regression: LaTeX \\neq outputs as combining slash U+0338 + '=', which
    is two chars. \\mu outputs as micro sign U+00B5, not greek mu U+03BC.
    _normalize_math_text should fix both."""
    # \neq: U+0338 + =
    raw_neq = "a \u0338= b"
    assert _normalize_math_text(raw_neq) == "a \u2260 b"

    # \mu: U+00B5 -> U+03BC
    raw_mu = "\u00b5 = 0.5"
    assert _normalize_math_text(raw_mu) == "\u03bc = 0.5"

    # other math symbols should not be affected
    other = "\u222b \u2211 \u221e \u221a \u03b1 \u03b2 \u03b3 \u2264 \u2265"
    assert _normalize_math_text(other) == other


def test_ingest_directory_is_idempotent_on_rerun(mocker, workdir):
    """同一 corpus 跑两次：第二次 chunks_added==0，DB 内 chunk 数不变。"""
    corpus = workdir / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("段落一\n\n段落二", encoding="utf-8")
    (corpus / "b.md").write_text("内容 b", encoding="utf-8")
    embed = mocker.patch(
        "math_agent.rag.ingest.embed_texts",
        side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts),
    )

    db = workdir / "vec.db"
    rep1 = ingest_directory(src_dir=corpus, db_path=db,
                            embedding_model="m", dim=3,
                            max_chars=200, overlap=20)
    assert rep1.chunks_added >= 2
    calls_after_first_run = embed.call_count

    # 第二次跑同样 corpus：应全部命中去重，新增 0
    rep2 = ingest_directory(src_dir=corpus, db_path=db,
                            embedding_model="m", dim=3,
                            max_chars=200, overlap=20)
    assert rep2.chunks_added == 0
    assert embed.call_count == calls_after_first_run  # 不重复消费 embedding API

    # DB 内 chunk 总数与第一次一致（用 search k=大数间接校验不翻倍）
    from math_agent.rag.store import VectorStore
    s = VectorStore.open(db, dim=3)
    hits = s.search([1.0, 0.0, 0.0], k=1000)
    assert len(hits) == rep1.chunks_added
    s.close()


def test_ingest_derives_source_type_paper_from_papers_dir(mocker, workdir):
    """路径含 papers/ → source_type='paper'。"""
    corpus = workdir / "corpus" / "papers"
    corpus.mkdir(parents=True)
    (corpus / "x.md").write_text("## 模型\n论文内容", encoding="utf-8")
    mocker.patch("math_agent.rag.ingest.embed_texts",
                  side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    db = workdir / "vec.db"
    ingest_directory(src_dir=workdir / "corpus", db_path=db,
                     embedding_model="m", dim=3, max_chars=200, overlap=20)
    from math_agent.rag.store import VectorStore
    s = VectorStore.open(db, dim=3)
    hits = s.search([1.0, 0.0, 0.0], k=10)
    s.close()
    assert hits and all(h.source_type == "paper" for h in hits)


def test_ingest_derives_source_type_model_lib_default(mocker, workdir):
    """路径含 models/（非 papers）→ source_type='model_lib'。"""
    corpus = workdir / "corpus" / "models"
    corpus.mkdir(parents=True)
    (corpus / "y.md").write_text("## 适用场景\n模型内容", encoding="utf-8")
    mocker.patch("math_agent.rag.ingest.embed_texts",
                  side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    db = workdir / "vec.db"
    ingest_directory(src_dir=workdir / "corpus", db_path=db,
                     embedding_model="m", dim=3, max_chars=200, overlap=20)
    from math_agent.rag.store import VectorStore
    s = VectorStore.open(db, dim=3)
    hits = s.search([1.0, 0.0, 0.0], k=10)
    s.close()
    assert hits and all(h.source_type == "model_lib" for h in hits)


def test_ingest_derives_source_type_model_lib_for_flat_corpus(mocker, workdir):
    """corpus 根目录直接放文件（无 papers 父目录）→ 'model_lib'。"""
    corpus = workdir / "corpus"
    corpus.mkdir()
    (corpus / "z.md").write_text("平铺内容", encoding="utf-8")
    mocker.patch("math_agent.rag.ingest.embed_texts",
                  side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    db = workdir / "vec.db"
    ingest_directory(src_dir=corpus, db_path=db,
                     embedding_model="m", dim=3, max_chars=200, overlap=20)
    from math_agent.rag.store import VectorStore
    s = VectorStore.open(db, dim=3)
    hits = s.search([1.0, 0.0, 0.0], k=10)
    s.close()
    assert hits and all(h.source_type == "model_lib" for h in hits)
