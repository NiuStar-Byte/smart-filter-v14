import ccxt

# Your current TOKENS list
TOKENS = [
    "ENA-USDT", "DOGE-USDT", "CFX-USDT", "PENGU-USDT", "VINE-USDT",
    "EIGEN-USDT", "SUI-USDT", "PEPE-USDT", "HYPE-USDT", "UNI-USDT",
    "TON-USDT", "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "SAHARA-USDT",
    "VIRTUAL-USDT", "ZORA-USDT", "ERA-USDT", "AAVE-USDT", "GALA-USDT",
    "TREE-USDT", "SPK-USDT", "SKATE-USDT", "LA-USDT",  "ZKJ-USDT"
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
