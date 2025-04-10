from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.utils.request import Request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# Загрузка текста
with open("kol_dogovor.txt", "r", encoding="utf-8") as f:
    pdf_text = f.read()

# Разделение на абзацы
articles = [p.strip() for p in pdf_text.split("\n\n") if len(p.strip()) > 30]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(articles)

def find_article(question):
    q_vec = vectorizer.transform([question])
    similarity = cosine_similarity(q_vec, X)
    best_idx = similarity.argmax()
    return articles[best_idx]

def start(update: Update, context: CallbackContext):
    update.message.reply_text("👋 Задайте вопрос по коллективному договору.")

def handle_question(update: Update, context: CallbackContext):
    q = update.message.text
    result = find_article(q)
    update.message.reply_text(result[:3000])

def main():
    token = os.getenv("TELEGRAM_TOKEN")

    # Увеличенные таймауты и контроль соединения
    request = Request(
        connect_timeout=10.0,
        read_timeout=10.0,
        con_pool_size=8
    )
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