import ccxt

# Your current TOKENS list
TOKENS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
    "AVAX-USDT", "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT",
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT",
    "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT",
    "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT",
    "DUCK-USDT", "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT",
    "VOXEL-USDT", "ALU-USDT", "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT", 
    "DOGE-USDT", "ALGO-USDT", "DOT-USDT", "NEWT-USDT", "SAHARA-USDT",
    "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", "ENA-USDT",
    "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT" 
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
