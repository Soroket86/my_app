from telegram import Bot, Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.utils.request import Request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import re

# === Загрузка текста ===
with open("kol_dogovor.txt", "r", encoding="utf-8") as f:
    pdf_text = f.read()

# === Разделение на статьи/разделы ===
sections = re.split(r"(РАЗДЕЛ\s+\d+\..*|СТАТЬЯ\s+\d+\..*)", pdf_text, flags=re.IGNORECASE)
articles = []
for i in range(1, len(sections), 2):
    header = sections[i].strip()
    body = sections[i+1].strip() if i+1 < len(sections) else ""
    articles.append(header + "\n" + body)

# === Векторизация ===
vectorizer = TfidfVectorizer()

# если текст не распарсился, подставим примеры вручную
if not articles or len(articles) < 3:
    articles = [
        "РАЗДЕЛ 1. Общие положения\nНастоящий коллективный договор регулирует трудовые отношения на предприятии.",
        "СТАТЬЯ 5. Отпуска\nРаботникам предоставляется ежегодный отпуск продолжительностью не менее 28 дней.",
        "СТАТЬЯ 7. Молодые специалисты\nМолодым специалистам предоставляются дополнительные льготы и выплаты."
    ]
X = vectorizer.fit_transform(articles)

def find_article(question):
    q_vec = vectorizer.transform([question])
    similarity = cosine_similarity(q_vec, X)
    best_idx = similarity.argmax()
    return articles[best_idx]

# === Команды ===
def start(update: Update, context: CallbackContext):
    keyboard = [["🟢 Задать вопрос", "📄 Инструкция"], ["📬 Связаться с профсоюзом"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    update.message.reply_text("👋 Добро пожаловать! Выберите действие ниже или напишите вопрос:", reply_markup=reply_markup)

def handle_menu(update: Update, context: CallbackContext):
    text = update.message.text
    if "Инструкция" in text:
        update.message.reply_text("📄 Просто напишите свой вопрос, например:\n\n«Сколько дней отпуска у работников с ненормированным графиком?»")
    elif "Связаться" in text:
        update.message.reply_text("📬 Вы можете обратиться в профсоюз по email: profcom@avtovaz.ru")
    elif "Задать вопрос" in text:
        update.message.reply_text("✍️ Напишите свой вопрос в следующем сообщении.")
    else:
        result = find_article(text)
        update.message.reply_text(result[:3000])  # Telegram ограничивает длину сообщений

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    request = Request(connect_timeout=10.0, read_timeout=10.0, con_pool_size=8)
    bot = Bot(token=token, request=request)

    updater = Updater(bot=bot, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_menu))

    print("⏳ Бот запускается...")
    time.sleep(2)
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
