from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import fitz  # PyMuPDF

# --- Загрузка и преобразование PDF в текст ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Проверяем, если текст уже извлечен
if not os.path.exists("kol_dogovor.txt"):
    pdf_text = extract_text_from_pdf("kol_dogovor.pdf")
    with open("kol_dogovor.txt", "w", encoding="utf-8") as f:
        f.write(pdf_text)
else:
    with open("kol_dogovor.txt", "r", encoding="utf-8") as f:
        pdf_text = f.read()

# Разделяем текст на абзацы (по двойному переносу строки)
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
    updater = Updater(token)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_question))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()