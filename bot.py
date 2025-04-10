from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.utils.request import Request
from sentence_transformers import SentenceTransformer, util
import os
import time
import re

# Загружаем текст
with open("kol_dogovor.txt", "r", encoding="utf-8") as f:
    pdf_text = f.read()

# Разделяем на статьи и разделы
sections = re.split(r"(РАЗДЕЛ\s+\d+\..*|СТАТЬЯ\s+\d+\..*)", pdf_text, flags=re.IGNORECASE)
articles = []

for i in range(1, len(sections), 2):
    header = sections[i].strip()
    body = sections[i+1].strip() if i+1 < len(sections) else ""
    articles.append(header + "\n" + body)

# Инициализация модели
print("🔍 Загрузка модели...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
article_embeddings = model.encode(articles, convert_to_tensor=True)

def find_article(question):
    question_embedding = model.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, article_embeddings)
    best_idx = similarities.argmax()
    return articles[best_idx]

def start(update: Update, context: CallbackContext):
    update.message.reply_text("👋 Задайте вопрос по коллективному договору.")

def handle_question(update: Update, context: CallbackContext):
    q = update.message.text
    result = find_article(q)
    update.message.reply_text(result[:3000])

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    request = Request(connect_timeout=10.0, read_timeout=10.0, con_pool_size=8)
    bot = Bot(token=token, request=request)

    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_question))

    print("⏳ Ожидание перед запуском polling...")
    time.sleep(3)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()