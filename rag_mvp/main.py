import os
import sys
import asyncio
import logging
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv

from utils import ensure_dir, save_bytes, read_text_file, chunk_text
from embeddings import embed_texts
from llm import answer_with_gigachat
from milvus_store import MilvusStore

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("rag_bot")

# ---------------- ENV ----------------
load_dotenv()

TG_BOT_TOKEN = (os.getenv("TG_BOT_TOKEN") or "").strip()
if not TG_BOT_TOKEN:
    raise RuntimeError("Вы забыли вставить TG_BOT_TOKEN.")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()

# Таймауты (можно менять через .env)
EMBED_TIMEOUT_SEC = int(os.getenv("EMBED_TIMEOUT_SEC", "240"))   # эмбеддинги чанков
MILVUS_TIMEOUT_SEC = int(os.getenv("MILVUS_TIMEOUT_SEC", "60"))  # вставка/поиск
READ_TIMEOUT_SEC = int(os.getenv("READ_TIMEOUT_SEC", "60"))      # чтение/конвертация

# ---------------- BOT ----------------
bot = Bot(token=TG_BOT_TOKEN)
dp = Dispatcher()
store = MilvusStore()

MENU_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="/status")], [KeyboardButton(text="/reset")]],
    resize_keyboard=True,
)

ALLOWED_SUFFIXES = {
    ".txt", ".md", ".pdf",
    ".docx", ".pptx", ".xlsx",
    ".html", ".htm",
    ".csv", ".json"
}

# словари состояния (в памяти, до перезапуска)
LAST_DOC_PATH = {}
LAST_DOC_TEXT = {}
LAST_DOC_CHUNKS = {}
LAST_DOC_EMBEDS = {}


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет! Я RAG-бот.\n\n"
        "1) Пришли мне файл\n"
        "2) Задай интересующий вопрос по файлу.\n\n"
        "Команды:\n"
        "/status — активный файл\n"
        "/reset — сбросить активный файл",
        reply_markup=MENU_KEYBOARD
    )


@dp.message(F.document)
async def on_document(message: Message):
    doc = message.document
    chat_id = message.chat.id

    filename = (doc.file_name or "document.txt").strip()
    suffix = Path(filename).suffix.lower()

    logger.info("DOC: chat_id=%s filename=%s file_id=%s", chat_id, filename, doc.file_id)

    if suffix not in ALLOWED_SUFFIXES:
        await message.answer(
            "Неподходящий формат файла.\n"
            "Поддерживаю: " + ", ".join(sorted(ALLOWED_SUFFIXES)),
            reply_markup=MENU_KEYBOARD,
        )
        return

    chat_dir = DATA_DIR / "uploads" / str(chat_id)
    ensure_dir(chat_dir)
    save_path = chat_dir / filename

    # 1) скачиваем файл
    try:
        await message.answer("Получаю файл от Telegram...")
        logger.info("STEP 1: downloading file from Telegram")

        tg_file = await bot.get_file(doc.file_id)
        file_obj = await bot.download_file(tg_file.file_path)
        data = file_obj.read()

        save_bytes(save_path, data)
        LAST_DOC_PATH[chat_id] = save_path

        logger.info("STEP 1 DONE: saved to %s bytes=%d", save_path, len(data))

    except Exception:
        logger.exception("STEP 1 FAILED: download/save")
        await message.answer("Не смог скачать/сохранить файл (см. логи в терминале).")
        return

    # 2) читаем текст
    try:
        await message.answer("Читаю текст из файла...")
        logger.info("STEP 2: read_text_file(%s)", save_path)

        text = await asyncio.wait_for(
            asyncio.to_thread(read_text_file, save_path),
            timeout=READ_TIMEOUT_SEC,
        )

        LAST_DOC_TEXT[chat_id] = text
        logger.info("STEP 2 DONE: chars=%d", len(text))

    except asyncio.TimeoutError:
        logger.exception("STEP 2 TIMEOUT: reading/conversion too long")
        await message.answer("Чтение/конвертация файла заняли слишком много времени (таймаут).")
        return
    except Exception:
        logger.exception("STEP 2 FAILED: reading text")
        await message.answer("Ошибка чтения текста из файла (см. логи в терминале).")
        return

    # 3) режем на чанки
    try:
        await message.answer("Режу текст на фрагменты...")
        logger.info("STEP 3: chunk_text chars=%d", len(text))

        chunks = chunk_text(text)
        LAST_DOC_CHUNKS[chat_id] = chunks

        logger.info("STEP 3 DONE: chunks=%d", len(chunks))

    except Exception:
        logger.exception("STEP 3 FAILED: chunking")
        await message.answer("Ошибка разбиения текста на чанки (см. логи в терминале).")
        return

    # 4) эмбеддинги
    try:
        await message.answer("Считаю эмбеддинги (может занять 1–3 минуты на CPU)...")
        logger.info("STEP 4: embed_texts chunks=%d timeout=%ds", len(chunks), EMBED_TIMEOUT_SEC)

        embs = await asyncio.wait_for(
            asyncio.to_thread(embed_texts, chunks),
            timeout=EMBED_TIMEOUT_SEC,
        )
        LAST_DOC_EMBEDS[chat_id] = embs

        shape = getattr(embs, "shape", None)
        logger.info("STEP 4 DONE: embeddings shape=%s", shape)

    except asyncio.TimeoutError:
        logger.exception("STEP 4 TIMEOUT: embeddings too long")
        await message.answer(
            "Эмбеддинги считаются слишком долго (таймаут).\n"
            "Частые причины: модель не скачалась, интернет/диск, или torch подвис."
        )
        return
    except Exception:
        logger.exception("STEP 4 FAILED: embeddings")
        await message.answer(
            "Ошибка при вычислении эмбеддингов (см. логи в терминале).\n"
            "Если видишь про meta tensor — нужно поставить lock в embeddings.py."
        )
        return

    # 5) Milvus reset + insert
    try:
        await message.answer("Записываю фрагменты в Milvus...")
        logger.info("STEP 5: milvus reset_chat + upsert timeout=%ds", MILVUS_TIMEOUT_SEC)

        await asyncio.wait_for(
            asyncio.to_thread(store.reset_chat, chat_id),
            timeout=MILVUS_TIMEOUT_SEC,
        )
        await asyncio.wait_for(
            asyncio.to_thread(store.upsert_chunks, chat_id, str(save_path), chunks, embs),
            timeout=MILVUS_TIMEOUT_SEC,
        )

        logger.info("STEP 5 DONE: milvus upsert complete")

    except asyncio.TimeoutError:
        logger.exception("STEP 5 TIMEOUT: milvus too long")
        await message.answer(
            "Milvus отвечает слишком долго (таймаут).\n"
            "Проверь, что контейнер milvus healthy и доступен."
        )
        return
    except Exception:
        logger.exception("STEP 5 FAILED: milvus")
        await message.answer("Ошибка при записи в Milvus (см. логи в терминале).")
        return

    # готово
    await message.answer(
        f"Файл сохранён и проиндексирован: {filename}\n"
        f"Символов текста: {len(text)}\n"
        f"Фрагментов: {len(chunks)}\n",
        reply_markup=MENU_KEYBOARD,
    )


@dp.message(Command("status"))
async def status(message: Message):
    chat_id = message.chat.id
    doc_path = LAST_DOC_PATH.get(chat_id)

    if not doc_path or not doc_path.exists():
        await message.answer(
            "Сейчас нет активного файла. Пришли файл, по которому есть вопросы!",
            reply_markup=MENU_KEYBOARD
        )
        return

    await message.answer(
        f"Активный файл: {doc_path.name}",
        reply_markup=MENU_KEYBOARD
    )


@dp.message(Command("reset"))
async def reset(message: Message):
    chat_id = message.chat.id

    LAST_DOC_PATH.pop(chat_id, None)
    LAST_DOC_TEXT.pop(chat_id, None)
    LAST_DOC_CHUNKS.pop(chat_id, None)
    LAST_DOC_EMBEDS.pop(chat_id, None)

    try:
        await asyncio.wait_for(
            asyncio.to_thread(store.reset_chat, chat_id),
            timeout=MILVUS_TIMEOUT_SEC,
        )
    except Exception:
        logger.exception("RESET: failed to reset in Milvus (continuing)")

    await message.answer(
        "Ок! Активный файл сброшен. Пришли новый файл!",
        reply_markup=MENU_KEYBOARD
    )


@dp.message(F.text)
async def text_as_question(message: Message):
    chat_id = message.chat.id
    text = (message.text or "").strip()

    if not text or text.startswith("/"):
        return

    doc_path = LAST_DOC_PATH.get(chat_id)
    if not doc_path or not doc_path.exists():
        await message.answer("Активного файла нет. Пришли файл для разбора!")
        return

    # Если по какой-то причине кеш пустой — перечитаем и переэмбедим (с логами)
    doc_text = LAST_DOC_TEXT.get(chat_id, "")
    chunks = LAST_DOC_CHUNKS.get(chat_id, [])
    chunks_emb = LAST_DOC_EMBEDS.get(chat_id)

    if (not doc_text) or (not chunks) or (chunks_emb is None):
        logger.warning("CACHE MISS: rebuilding cache from disk chat_id=%s path=%s", chat_id, doc_path)
        await message.answer("Перечитываю файл и восстанавливаю индекс в памяти...")

        try:
            doc_text = await asyncio.wait_for(
                asyncio.to_thread(read_text_file, doc_path),
                timeout=READ_TIMEOUT_SEC,
            )
            chunks = chunk_text(doc_text)

            chunks_emb = await asyncio.wait_for(
                asyncio.to_thread(embed_texts, chunks),
                timeout=EMBED_TIMEOUT_SEC,
            )

            LAST_DOC_TEXT[chat_id] = doc_text
            LAST_DOC_CHUNKS[chat_id] = chunks
            LAST_DOC_EMBEDS[chat_id] = chunks_emb

            logger.info("CACHE REBUILT: chars=%d chunks=%d", len(doc_text), len(chunks))

        except Exception:
            logger.exception("CACHE REBUILD FAILED")
            await message.answer("Не смог восстановить кеш после перезапуска (см. логи в терминале).")
            return

    # эмбеддинг вопроса
    try:
        logger.info("Q: embedding question len=%d", len(text))
        q_embs = await asyncio.wait_for(
            asyncio.to_thread(embed_texts, [text]),
            timeout=60,
        )
        q = q_embs[0]
        logger.info("Q: embedding done dim=%d", len(q))

    except Exception:
        logger.exception("Q: failed to embed question")
        await message.answer("Не смог посчитать эмбеддинг вопроса (см. логи в терминале).")
        return

    # поиск в Milvus
    try:
        await message.answer("Ищу релевантные фрагменты в Milvus...")
        logger.info("SEARCH: chat_id=%s k=4", chat_id)

        hits = await asyncio.wait_for(
            asyncio.to_thread(store.search, chat_id, q, 4),
            timeout=MILVUS_TIMEOUT_SEC,
        )

        logger.info("SEARCH DONE: hits=%d", len(hits) if hits else 0)

    except Exception:
        logger.exception("SEARCH FAILED")
        await message.answer("Ошибка поиска в Milvus (см. логи в терминале).")
        return

    if not hits:
        await message.answer("Не нашёл релевантных фрагментов в базе (Milvus).")
        return

    context = "\n\n---\n\n".join(ch for score, ch in hits if ch)

    await message.answer("Думаю над ответом c GigaChat...")
    try:
        final_answer = await asyncio.wait_for(
            asyncio.to_thread(answer_with_gigachat, text, context),
            timeout=90,
        )
        await message.answer(final_answer[:3500])

    except asyncio.TimeoutError:
        logger.exception("GIGACHAT TIMEOUT")
        await message.answer("GigaChat отвечает слишком долго (таймаут).")
    except Exception:
        logger.exception("GIGACHAT FAILED")
        await message.answer("Ошибка при запросе к GigaChat (см. логи в терминале).")


async def main():
    ensure_dir(DATA_DIR)
    logger.info("Starting bot. DATA_DIR=%s MILVUS_URI=%s", DATA_DIR, os.getenv("MILVUS_URI"))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())