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
    if suffix not in {".txt", ".md"}:
        raise ValueError(f"Недоступный тип файла: {suffix}. Используйте .txt or .md")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except:
        text = raw.decode("cp1251", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[обрезано]..."
    return text
