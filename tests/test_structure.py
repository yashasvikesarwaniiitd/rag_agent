"""Structure-aware chunking and the loaders that feed it headings."""
import pymupdf

from rag.adapters.loaders import FileLoader
from rag.domain.chunking import chunk_document
from rag.domain.models import Line, Page


def page(n, lines, title="Manual Ch 6"):
    lines = tuple(Line(t, heading=h) for t, h in lines)
    return Page("doc", "doc.pdf", n, "\n".join(l.text for l in lines), lines, title)


def test_heading_starts_a_new_chunk_and_labels_it():
    chunks = chunk_document([page(1, [("60.3 - Credentialing", True), ("Body one.", False),
                                      ("Recredentialing", True), ("Every 3 years.", False)])])
    assert [(c.context, c.text) for c in chunks] == [
        ("Manual Ch 6 > 60.3 - Credentialing", "Body one."),
        ("Manual Ch 6 > 60.3 - Credentialing > Recredentialing", "Every 3 years."),
    ]


def test_section_carries_across_page_breaks():
    chunks = chunk_document([page(12, [("60.3 - Credentialing", True), ("a", False)]),
                             page(13, [("continued text", False)])])
    assert chunks[1].page == 13 and chunks[1].context == "Manual Ch 6 > 60.3 - Credentialing"


def test_new_numbered_section_clears_the_old_sub_heading():
    chunks = chunk_document([page(1, [("60.3 - A", True), ("Sub", True), ("x", False),
                                      ("60.4 - B", True), ("y", False)])])
    assert chunks[-1].context == "Manual Ch 6 > 60.4 - B"


def test_paragraph_marker_is_a_section_in_a_regulation_but_a_sub_in_a_manual():
    reg = chunk_document([page(1, [("(a) General rule", True), ("x", False),
                                   ("(b) Basic requirements", True), ("y", False)],
                               title="§ 422.204")])
    assert reg[1].context == "§ 422.204 > (b) Basic requirements"
    manual = chunk_document([page(1, [("10.4.5 - Revalidations", True), ("(iii)", True),
                                      ("z", False)])])
    assert manual[0].context == "Manual Ch 6 > 10.4.5 - Revalidations > (iii)"


def test_label_is_kept_separate_from_the_quotable_text():
    chunk = chunk_document([page(1, [("60.3 - Credentialing", True), ("Every 3 years.", False)])])[0]
    assert chunk.text == "Every 3 years."
    assert chunk.context == "Manual Ch 6 > 60.3 - Credentialing"


def test_plain_text_pages_still_detect_numbered_headings():
    p = Page("d", "d.txt", 1, "10.1 - Intro\nHello.\n10.2 - Next\nWorld.")
    assert [c.context for c in chunk_document([p])] == ["10.1 - Intro", "10.2 - Next"]


def test_chunk_ids_stay_unique_per_page():
    chunks = chunk_document([page(1, [("A", True), ("x", False), ("B", True), ("y", False)])])
    assert len({c.id for c in chunks}) == len(chunks)


ECFR = """<div class="editorial-note"><h6>Editorial Note on Part 422</h6><p>Nomenclature
changes.</p></div><div class="section"><h4>§ 422.204 Provider selection and credentialing.</h4>
<p class="indent-1" data-title="422.204(a)"><span>(a)</span> <em>General rule.</em> An MA
organization must have written policies.</p>
<p class="indent-1" data-title="422.204(b)"><span>(b)</span> <em>Basic requirements.</em>
Recredentialing at least every 3 years.</p></div>"""


def test_ecfr_html_drops_editorial_note_and_labels_paragraphs(tmp_path):
    f = tmp_path / "ecfr_422.204.html"
    f.write_text(ECFR, encoding="utf-8")
    pages = FileLoader().load(f)
    assert pages[0].doc_title == "§ 422.204 Provider selection and credentialing"
    assert "Editorial" not in pages[0].text and "Nomenclature" not in pages[0].text
    chunks = chunk_document(pages)
    recred = next(c for c in chunks if "every 3 years" in c.text)
    assert recred.context == "§ 422.204 Provider selection and credentialing > (b) Basic requirements"


def test_pdf_large_font_line_becomes_a_heading_and_running_header_is_dropped(tmp_path):
    doc = pymupdf.open()
    topics = ["Enrollment", "Screening", "Credentialing", "Revalidation", "Termination", "Appeals"]
    for i, topic in enumerate(topics):
        p = doc.new_page()
        p.insert_text((72, 40), "Report to Congress on Medicaid", fontsize=8)  # every page
        p.insert_text((72, 90), f"{topic} rules", fontsize=16)
        p.insert_text((72, 120), f"What the {topic.lower()} section requires of plans.", fontsize=10)
        p.insert_text((72, 135), f"Longer {topic.lower()} body text so ten point is the body.",
                      fontsize=10)
    path = tmp_path / "report.pdf"
    doc.save(path)
    doc.close()

    pages = FileLoader().load(path)
    assert "Report to Congress" not in pages[2].text
    headings = [l.text for l in pages[2].lines if l.heading]
    assert headings == ["Credentialing rules"]
    assert chunk_document(pages)[2].context.endswith("Credentialing rules")
