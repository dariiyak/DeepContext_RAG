import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
import asyncio

from dotenv import load_dotenv

from utils import ensure_dir, save_bytes, read_text_file, chunk_text
from embeddings import embed_texts
from retrieval import retrieve_top_k
from llm import answer_with_gigachat

load_dotenv()

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN").strip()
if not TG_BOT_TOKEN:
    raise RuntimeError("Вы забыли вставить TG_BOT_TOKEN.")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()

bot = Bot(token=TG_BOT_TOKEN)
dp = Dispatcher()

MENU_KEYBOARD = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text="/status")], [KeyboardButton(text="/reset")]
    ], resize_keyboard=True)
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf",
    ".docx", ".pptx", ".xlsx",
    ".html", ".htm",
    ".csv", ".json"}

# словарь {chat_id: path}, содержит путь активного файла
LAST_DOC_PATH = {}
# словарь {chat_id: str}, содержит текст активного файла
LAST_DOC_TEXT = {}
# словарь {chat_id: list[str]}, содержит кэш чанков
LAST_DOC_CHUNKS = {}

@dp.message(CommandStart())
async def start(message):
    await message.answer(
        "Привет! Я RAG-бот.\n\n"
        "1) Пришли мне файл\n"
        "2) Задай интересующий вопрос по файлу.\n"
        "Команды:\n"
        "/status — активный файл\n"
        "/reset — сбросить активный файл", reply_markup=MENU_KEYBOARD)
    
@dp.message(F.document)
async def on_document(message):
    doc = message.document
    chat_id = message.chat.id

    filename = (doc.file_name or "document.txt").strip()
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        await message.answer(
            "Неподходящий формат файла.\n"
            "Поддерживаю: " + ", ".join(sorted(ALLOWED_SUFFIXES)))
        return
    
    chat_dir = DATA_DIR / "uploads" / str(chat_id)
    ensure_dir(chat_dir)
    save_path = chat_dir / filename

    file = await bot.get_file(doc.file_id)
    file_path = await bot.download_file(file.file_path)
    data = file_path.read()

    save_bytes(save_path, data)
    LAST_DOC_PATH[chat_id] = save_path

    try:
        text = read_text_file(save_path)
        LAST_DOC_TEXT[chat_id] = text
        chunks = chunk_text(text)
        LAST_DOC_CHUNKS[chat_id] = chunks
        await message.answer("Индексирую файл в Milvus...")
        embs = await asyncio.to_thread(embed_texts, chunks)
    except Exception as ex:
        await message.answer(f"Ошибка чтения файла: {ex}")
        return
    
    await message.answer(
        f"Файл сохранён: {filename}\n"
        f"Символов текста: {len(text)}\n\n",
        reply_markup=MENU_KEYBOARD,
    )

@dp.message(Command("status"))
async def status(message):
    chat_id = message.chat.id
    doc_path = LAST_DOC_PATH.get(chat_id)

    if not doc_path or not doc_path.exists():
        await message.answer(
            "Сейчас нет активного файла. Пришли файл, по которому есть вопросы!",
            reply_markup=MENU_KEYBOARD
            )
        return
    
    await message.answer(
        f"Активный файл: {doc_path.name}\n",
        reply_markup=MENU_KEYBOARD,
    )

@dp.message(Command("reset"))
async def reset(message):
    chat_id = message.chat.id

    if chat_id in LAST_DOC_PATH:
        del LAST_DOC_PATH[chat_id]
        LAST_DOC_TEXT.pop(chat_id, None)
        LAST_DOC_CHUNKS.pop(chat_id, None)
        LAST_DOC_EMBEDS.pop(chat_id, None)
        await message.answer(
            "Ок! Активный файл сброшен. Пришли новый файл!",
            reply_markup=MENU_KEYBOARD)
    else:
        await message.answer(
            "Нечего сбрасывать: активного файла нет. Пришли новый файл!",
            reply_markup=MENU_KEYBOARD)

@dp.message(F.text)
async def text_as_question(message):
    chat_id = message.chat.id
    text = (message.text or "").strip()

    if text.startswith("/"):
        return
    
    doc_path = LAST_DOC_PATH.get(chat_id)
    if not doc_path or not doc_path.exists():
        await message.answer(
            "Активного файла нет \n"
            "Пришлите его для разбора!",
        )
        return

    doc_text = LAST_DOC_TEXT.get(chat_id, "")
    chunks = LAST_DOC_CHUNKS.get(chat_id, [])
    chunks_embed = LAST_DOC_EMBEDS.get(chat_id)
    
    if (not doc_text) or (not chunks) or (chunks_embed is None):
        try:
            doc_text = read_text_file(doc_path)
            LAST_DOC_TEXT[chat_id] = doc_text
            chunks = chunk_text(doc_text)
            LAST_DOC_CHUNKS[chat_id] = chunks
            LAST_DOC_EMBEDS[chat_id] = await asyncio.to_thread(embed_texts, chunks)
            chunks_embed = LAST_DOC_EMBEDS[chat_id]
        except Exception as ex:
            await message.answer(f"Не смог перечитать файл после перезапуска: {ex}")
            return
    
    q = await asyncio.to_thread(embed_texts, [text])
    q = q[0]
    
    top = retrieve_top_k(q, chunks, chunks_embed, k=4)
    if not top:
        await message.answer("Не нашёл релевантных фрагментов в документе")
        return

    context = "\n\n---\n\n".join([ch for _, s, ch in top])
    
    await message.answer("Думаю над ответом c GigaChat...")

    final_answer = await asyncio.to_thread(answer_with_gigachat, text, context)
    await message.answer(final_answer[:3500])
    

async def main():
    ensure_dir(DATA_DIR)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())