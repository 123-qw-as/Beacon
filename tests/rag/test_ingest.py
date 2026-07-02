from math_agent.rag.ingest import ingest_directory


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


def test_ingest_directory_is_idempotent_on_rerun(mocker, workdir):
    """同一 corpus 跑两次：第二次不新增 chunk，DB 内总数不变。"""
    corpus = workdir / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("段落一\n\n段落二", encoding="utf-8")
    (corpus / "b.md").write_text("内容 b", encoding="utf-8")
    mocker.patch("math_agent.rag.ingest.embed_texts",
                  side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    db = workdir / "vec.db"
    rep1 = ingest_directory(src_dir=corpus, db_path=db,
                            embedding_model="m", dim=3,
                            max_chars=200, overlap=20)
    rep2 = ingest_directory(src_dir=corpus, db_path=db,
                            embedding_model="m", dim=3,
                            max_chars=200, overlap=20)
    assert rep1.chunks_added > 0
    assert rep2.chunks_added == 0   # 全部命中去重

    # DB 内总数应等于第一次新增数，不翻倍
    from math_agent.rag.store import VectorStore
    s = VectorStore.open(db, dim=3)
    total = len(s.search([1.0, 0.0, 0.0], k=1000))
    s.close()
    assert total == rep1.chunks_added


def test_ingest_directory_is_idempotent_on_rerun(mocker, workdir):
    """同一 corpus 跑两次：第二次 chunks_added==0，DB 内 chunk 数不变。"""
    corpus = workdir / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("段落一\n\n段落二", encoding="utf-8")
    (corpus / "b.md").write_text("内容 b", encoding="utf-8")
    mocker.patch("math_agent.rag.ingest.embed_texts",
                  side_effect=lambda texts, **kw: [[1.0, 0.0, 0.0]] * len(texts))

    db = workdir / "vec.db"
    rep1 = ingest_directory(src_dir=corpus, db_path=db,
                            embedding_model="m", dim=3,
                            max_chars=200, overlap=20)
    assert rep1.chunks_added >= 2

    # 第二次跑同样 corpus：应全部命中去重，新增 0
    rep2 = ingest_directory(src_dir=corpus, db_path=db,
                            embedding_model="m", dim=3,
                            max_chars=200, overlap=20)
    assert rep2.chunks_added == 0

    # DB 内 chunk 总数与第一次一致（用 search k=大数间接校验不翻倍）
    from math_agent.rag.store import VectorStore
    s = VectorStore.open(db, dim=3)
    hits = s.search([1.0, 0.0, 0.0], k=100)
    assert len(hits) == rep1.chunks_added
    s.close()
