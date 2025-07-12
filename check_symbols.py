import ccxt

# Your current TOKENS list
TOKENS = [
    "SPARK-USDT", "BID-USDT", "SKATE-USDT", "LA-USDT", "SPK-USDT",
    "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT",
    "FUN-USDT", "SNT-USDT", "X-USDT", "BANK-USDT", "RAY-USDT",
    "REX-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACT-USDT",
    "CROSS-USDT", "KNC-USDT", "TANSSI-USDT", "ARK-USDT", "PORTAL-USDT",
    "SKYAI-USDT", "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT"
]

# Load market symbols from each exchange
ku = ccxt.kucoin()
bn = ccxt.binance()
ku_markets = set(ku.load_markets().keys())
bn_markets = set(bn.load_markets().keys())

# Check each token
for tok in TOKENS:
    hyphen = tok
    slash  = tok.replace('-', '/')
    nobar  = tok.replace('-', '')
    ok_ku  = (slash in ku_markets) or (hyphen in ku_markets)
    ok_bn  = nobar in bn_markets
    print(f"{tok:12} → KuCoin? {ok_ku}  Binance? {ok_bn}")
