from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, User
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
import warnings
from telegram.warnings import PTBUserWarning
import os
import re
from pathlib import Path

from apis.danelfin import DanelfinAPI, DanelfinScores
from apis.tipranks import MyTipRanks, TipRanksScores
from apis.bridgewise import BridgewiseAPI
from yfinance import Ticker
from contextlib import suppress
import pandas as pd
from tabulate import tabulate
from logger import alfred_logger
from dataclasses import dataclass
import toml


AUTHORIZED_USERS = ("eld4d", "absdotan")
CSV_TABLE_FILE = "stock_analysis.csv"
AUTHORIZED_IDS = (1734405151,)
PICK_STOCKS = 1
CHOOSE_OUTPUT = 2
TABLE_ROWS = (
    "Danelfin General",
    "Danelfin Sentiment",
    "Danelfin Technical",
    "Danelfin Fundamental",
    "Bridgewise Score",
    "Current Price",
    "Price Target",
    "Best Price Target",
    "Consensus",
    "PE Ratio",
    "Beta",
    "1 Month Gain",
    "3 Months Gain",
    "6 Months Gain",
    "YTD Gain",
)
API_KEYS_FILE = "api_keys.toml"

warnings.filterwarnings("ignore", category=PTBUserWarning)
api_keys = toml.loads(Path(API_KEYS_FILE).read_text())

danelfin = DanelfinAPI(api_keys["danelfin"])
alfred_logger.info("Danelfin API Successfully initalized")

bridgewise = BridgewiseAPI(api_keys["bridgewise"])
companies_from_scanner = bridgewise.get_companies_from_scanner()
alfred_logger.info("Bridgewise API Successfully initalized")

tr_username, tr_password = api_keys["tipranks"].split(":")
tipranks = MyTipRanks(tr_username, tr_password)
alfred_logger.info("TipRanks API Successfully initalized")


@dataclass
class CombinedScores:
    ticker: str
    danelfin: DanelfinScores
    bridgewise: int
    tipranks: TipRanksScores

    @classmethod
    def from_ticker(cls, ticker: str):
        danelfin_scores = danelfin.get_ticker_ai_scores(ticker)
        bridgewise_score = bridgewise.get_ticker_ai_score(
            companies_from_scanner, ticker
        )

        tipranks_scores = tipranks.get_analyst_projection(ticker)
        return cls(ticker, danelfin_scores, bridgewise_score, tipranks_scores)


def is_valid_user(user: User) -> bool:
    return user.username in AUTHORIZED_USERS or user.id in AUTHORIZED_IDS


def generate_stock_info_table(df: pd.DataFrame, combined_scores: CombinedScores):

    df.at["Danelfin General", combined_scores.ticker] = combined_scores.danelfin.general
    df.at["Danelfin Sentiment", combined_scores.ticker] = (
        combined_scores.danelfin.sentiment
    )
    df.at["Danelfin Technical", combined_scores.ticker] = (
        combined_scores.danelfin.technical
    )
    df.at["Danelfin Fundamental", combined_scores.ticker] = (
        combined_scores.danelfin.fundamental
    )
    df.at["Bridgewise Score", combined_scores.ticker] = combined_scores.bridgewise
    df.at["Current Price", combined_scores.ticker] = combined_scores.tipranks.price
    df.at["Price Target", combined_scores.ticker] = (
        combined_scores.tipranks.price_target
    )
    df.at["Best Price Target", combined_scores.ticker] = (
        combined_scores.tipranks.best_price_target
    )
    df.at["Consensus", combined_scores.ticker] = combined_scores.tipranks.consensus
    df.at["PE Ratio", combined_scores.ticker] = combined_scores.tipranks.pe_ratio
    try:
        df.at["Beta", combined_scores.ticker] = Ticker(combined_scores.ticker).info[
            "beta"
        ]
    except KeyError:
        df.at["Beta", combined_scores.ticker] = None
    df.at["1 Month Gain", combined_scores.ticker] = (
        combined_scores.tipranks.one_month_gain
    )
    df.at["3 Months Gain", combined_scores.ticker] = (
        combined_scores.tipranks.three_months_gain
    )
    df.at["6 Months Gain", combined_scores.ticker] = (
        combined_scores.tipranks.six_months_gain
    )
    df.at["YTD Gain", combined_scores.ticker] = combined_scores.tipranks.ytd_gain


def generate_stock_info_message(combined_scores: CombinedScores) -> str:

    response = f"📈 *Stock Analysis for {re.escape(combined_scores.tipranks.company_name)} \\({combined_scores.ticker}\\)*\n"
    response += "\n💶 *Danelfin*\n"
    response += f"• General: {combined_scores.danelfin.general}\n"
    response += f"• Sentiment: {combined_scores.danelfin.sentiment}\n"
    response += f"• Technical: {combined_scores.danelfin.technical}\n"
    response += f"• Fundamental: {combined_scores.danelfin.fundamental}\n"
    response += "\n💷 *Bridgewise*\n"
    response += f"• Score: {re.escape(str(combined_scores.bridgewise))}\n"
    response += "\n💸 *General Info*\n"
    response += f"• Current Price: {re.escape(str(combined_scores.tipranks.price))}\n"
    response += f"• Price Target: {re.escape(str(combined_scores.tipranks.price_target))} \\(Best: {re.escape(str(combined_scores.tipranks.best_price_target))}\\)\n"
    response += f"• Analyst Consensus: {combined_scores.tipranks.consensus}\n"
    response += f"• PE Ratio: {re.escape(str(combined_scores.tipranks.pe_ratio))}\n"
    with suppress(KeyError):
        response += (
            f"• Beta: {re.escape(str(Ticker(combined_scores.ticker).info['beta']))}\n"
        )
    response += "\n🚀 *Performance*\n"
    response += f"• 1 Month: {re.escape(combined_scores.tipranks.one_month_gain)}\n"
    response += f"• 3 Months: {re.escape(combined_scores.tipranks.three_months_gain)}\n"
    response += f"• 6 Months: {re.escape(combined_scores.tipranks.six_months_gain)}\n"
    response += f"• Year To Date: {re.escape(combined_scores.tipranks.ytd_gain)}\n"

    return response


async def start_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:

    if not is_valid_user(update.message.from_user):
        alfred_logger.warning(f"Unauthozired access from {update.message.from_user}")
        await update.message.reply_text("You are not authorized to use Alfred")
        return ConversationHandler.END

    alfred_logger.info(f"Converstaion started with {update.message.from_user}")

    await update.message.reply_text(
        'Which stocks would you like me to analyze? 🕵️‍♂️\n\nI can analyze a single ticker, such as: "NVDA", or multiple tickers separated by commas, for example: "AMZN, MSFT, TSLA"'
    )
    return PICK_STOCKS


async def receive_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Store user stocks input
    context.user_data["stocks"] = [
        x.strip().upper() for x in update.message.text.split(",")
    ]

    alfred_logger.info(f"""Stocks picked: {context.user_data["stocks"]}""")
    await update.message.reply_text(
        f"""Stocks received: {', '.join(context.user_data["stocks"])} 🔥"""
    )

    # Present options for output format
    keyboard = [
        [InlineKeyboardButton("Table 📖", callback_data="table")],
        [InlineKeyboardButton("Message 💬", callback_data="message")],
        [InlineKeyboardButton("File 📁", callback_data="file")],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "How would you like to receive the output?", reply_markup=reply_markup
    )
    return CHOOSE_OUTPUT


async def choose_output(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "message":
        try:
            await analyze_stocks_message(update, context)
        except Exception as e:
            alfred_logger.error(e)
            raise e

    elif query.data == "table":
        try:
            context.args = ["table"]
            await analyze_stocks_table(update, context)
        except Exception as e:
            alfred_logger.error(e)
            raise e

    elif query.data == "file":
        try:
            context.args = ["file"]
            await analyze_stocks_table(update, context)
        except Exception as e:
            alfred_logger.error(e)
            raise e

    alfred_logger.info("Stocks analyzed successfully")
    return ConversationHandler.END


async def analyze_stocks_table(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    file_response = context.args[0] == "file"

    stock_tickers = context.user_data["stocks"]
    alfred_logger.info(
        f"Generating Table response for {stock_tickers}, {file_response = }"
    )
    await query.message.reply_text("Working on it... 📝")

    df = pd.DataFrame(columns=stock_tickers, index=TABLE_ROWS)
    for ticker in stock_tickers:
        alfred_logger.info(f"Analyzing {ticker}")
        combined_scores = CombinedScores.from_ticker(ticker)
        if combined_scores.tipranks._raw_data:
            generate_stock_info_table(df, combined_scores)
        else:
            alfred_logger.error(f"Received unknown ticker: {ticker}")
            continue

    if file_response:
        try:
            with Path(CSV_TABLE_FILE).open("w"):
                df.to_csv(CSV_TABLE_FILE, index=True)

            with Path(CSV_TABLE_FILE).open("rb") as document:
                await query.message.reply_document(document)
        finally:
            os.remove(CSV_TABLE_FILE)

    elif context.args[0] == "table":
        table = f'<pre>{tabulate(df, headers="keys", tablefmt="grid")}</pre>'
        await query.message.reply_text(table, parse_mode=ParseMode.HTML)


async def analyze_stocks_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    stock_tickers = context.user_data["stocks"]
    alfred_logger.info(f"Generating Message response for {stock_tickers}")

    await query.message.reply_text("Working on it... 📝")

    for ticker in stock_tickers:
        alfred_logger.info(f"Analyzing {ticker}")

        combined_scores = CombinedScores.from_ticker(ticker)

        if combined_scores.tipranks._raw_data:
            stock_info_response = generate_stock_info_message(combined_scores)
        else:
            stock_info_response = f"I didn't find any information about {ticker} 😔"
            alfred_logger.error(f"Received unknown ticker: {ticker}")

        await query.message.reply_text(
            stock_info_response, parse_mode=ParseMode.MARKDOWN_V2
        )


def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    update.message.reply_text("Conversation cancelled.")
    return ConversationHandler.END


conv_handler = ConversationHandler(
    entry_points=[
        CommandHandler("analyze", start_analyze)
    ],  # Command to start the conversation
    states={
        PICK_STOCKS: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, receive_stocks)
        ],  # Input handling state
        CHOOSE_OUTPUT: [CallbackQueryHandler(choose_output)],
    },
    fallbacks=[CommandHandler("cancel", cancel)],  # Fallback for cancellation
)
app = ApplicationBuilder().token(api_keys["telegram"]).build()

app.add_handler(conv_handler)

app.run_polling()
