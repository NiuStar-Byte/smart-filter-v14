import time
from kucoin_data import fetch_ohlcv
from smart_filter import analyze
from telegram_alert import send_telegram_alert

TOKEN_LIST = [
    "SPARK-USDT", "BID-USDT", "SKATE-USDT", "LA-USDT", "SPK-USDT", 
    "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT", 
    "FUN-USDT", "SNT-USDT", "X-USDT", "BANK-USDT", "RAY-USDT", 
    "REX-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACT-USDT"
]

TIMEFRAMES = ["3min", "5min"]

def main():
    while True:
        for tf in TIMEFRAMES:
            alert_messages = []
            for idx, token in enumerate(TOKEN_LIST, start=1):
                ohlcv = fetch_ohlcv(token, tf)
                if not ohlcv:
                    continue
                analysis_result = analyze(ohlcv)
                if analysis_result["score"] >= 10:  # example threshold
                    msg = (
                        f"{idx}. {token} ({tf}) {analysis_result['signal']} Signal\n"
                        f"💰 Price: {analysis_result['price']}\n"
                        f"✅ Score: {analysis_result['score']}/18\n"
                        f"📌 Passed: {analysis_result['passed']}/12"
                    )
                    alert_messages.append(msg)
            for message in alert_messages:
                send_telegram_alert(message)
        time.sleep(60)  # check every 60 seconds

if __name__ == "__main__":
    main()
