from pathlib import Path

# создать директорию в которой потом по chat_id будут создаваться файлы 
def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True) 

# создать filename и записать байты data 
def save_bytes(path, data):
    ensure_dir(path.parent)
    path.write_bytes(data)

# чтение текста из файла с защитой на max_chars
def read_text_file(path, max_chars=200000):
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = read_text_txt_md(path)
    elif suffix == ".pdf":
        text = read_text_pdf(path)
    else:
        text = read_text_markitdown(path)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[обрезано]..."
    return text

def read_text_txt_md(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("cp1251", errors="replace")
    return text

def read_text_pdf(path):
    import fitz

    parts = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            page_text = page.get_text("text", sort=True)
            page_text = page_text.strip()
            if page_text:
                parts.append(f"\n\n--- page {i} ---\n{page_text}")

    text = "".join(parts).strip()
    if not text:
        text = "В этом pdf текст не найден"
    return text

def read_text_markitdown(path):
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(path))
        return result.text_content or ""
    except Exception as ex:
        return f"[Не удалось конвертировать файл: {ex}]"
