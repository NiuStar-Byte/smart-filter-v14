# KuCoin Futures Auto Signal Bot

## Description
This bot fetches KuCoin futures market data for selected tokens and timeframes, applies a smart filter strategy to identify trading signals, and sends alerts via Telegram.

## Setup

1. Configure `BOT_TOKEN` and `CHAT_ID` environment variables or update `telegram_alert.py` directly.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the bot:
    ```
    python main.py
    ```
## Notes
- The bot runs continuously with cooldown timers for each timeframe.
- Signals include score and passed stacks for reference.
